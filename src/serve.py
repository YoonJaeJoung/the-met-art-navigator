"""
Backend API Server.

FastAPI application that serves:
- POST /search/text   — text-to-artwork search
- POST /search/image  — image-to-artwork search
- POST /train         — trigger model training (background)
- GET  /status        — training status
- GET  /gallery-map   — gallery coordinate data

Usage:
    .venv/bin/python src/serve.py [--port 8000]
"""

import argparse
import asyncio
import json
import os
import threading
from io import BytesIO
from pathlib import Path

# Prevent OpenMP multiple initialization crash on MacOS when bringing faiss and torch together
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from lit_model import ContrastiveModel, MetDataModule
from telemetry import TelemetryCallback

import uvicorn

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
MAP_DIR = Path("map")

app = FastAPI(title="Met Art Navigator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve map images as static files (only locally where map dir exists)
if MAP_DIR.exists():
    app.mount("/map", StaticFiles(directory=str(MAP_DIR)), name="map")

# ─── Global State ────────────────────────────────────────────────────────────

_state = {
    "index_visual": None,
    "index_semantic": None,
    "mapping_visual": None,
    "mapping_semantic": None,
    "metadata": None,
    "gallery_coords": None,
    "model": None,
    "text_tokenizer": None,
    "text_model": None,
    "image_processor": None,
    "image_model": None,
    "device": torch.device("cpu"),
    "training_status": "idle",  # idle | training | complete | error
    "training_error": None,
}


# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Load models and FAISS index on server start."""
    # Force CPU for inference serving to prevent Apple Silicon (MPS) segfaults with Nomic
    device = torch.device("cpu")
    _state["device"] = device
    print(f"Device: {device}")

    # Load contrastive model if checkpoint exists
    ckpt_path = CKPT_DIR / "contrastive_final.ckpt"
    if ckpt_path.exists():
        _state["model"] = ContrastiveModel.load_from_checkpoint(str(ckpt_path), map_location=device)
        _state["model"].eval()
        print("✓ Contrastive model loaded.")

    # Load metadata FIRST
    metadata_path = DATA_DIR / "met_final.parquet"
    if not metadata_path.exists():
        metadata_path = DATA_DIR / "met_enriched.parquet"
    if metadata_path.exists():
        _state["metadata"] = pd.read_parquet(metadata_path)
        print(f"✓ Metadata loaded: {len(_state['metadata'])} records")

        # Create mappings based on description availability
        metadata = _state["metadata"]
        has_desc = metadata["description"].notna() & (metadata["description"] != "")
        
        _state["mapping_semantic"] = np.where(has_desc)[0]
        _state["mapping_visual"] = np.where(~has_desc)[0]
        print(f"  - Semantic artworks: {len(_state['mapping_semantic'])}")
        print(f"  - Visual artworks: {len(_state['mapping_visual'])}")

    # Load FAISS indices
    projected_path = DATA_DIR / "images_projected.pt"
    if projected_path.exists() and _state["mapping_visual"] is not None:
        import faiss
        embeddings = torch.load(projected_path, map_location="cpu").numpy().astype(np.float32)
        vis_emb = embeddings[_state["mapping_visual"]]
        dim = vis_emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vis_emb)
        _state["index_visual"] = index
        print(f"✓ FAISS Visual index built: {index.ntotal} vectors, dim={dim}")

    text_unprojected_path = DATA_DIR / "text_unprojected.pt"
    if text_unprojected_path.exists() and _state["mapping_semantic"] is not None:
        import faiss
        embeddings_raw = torch.load(text_unprojected_path, map_location="cpu")
        # Ensure L2 normalization for cosine similarity
        embeddings_norm = F.normalize(embeddings_raw, p=2, dim=1).numpy().astype(np.float32)
        sem_emb = embeddings_norm[_state["mapping_semantic"]]
        dim = sem_emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(sem_emb)
        _state["index_semantic"] = index
        print(f"✓ FAISS Semantic (High-Fidelity) index built: {index.ntotal} vectors, dim={dim}")

    # Load dynamic map geolocations to override stale parquet mappings
    coords_path = DATA_DIR / "gallery_coords.json"
    if coords_path.exists():
        with open(coords_path, "r") as f:
            data = json.load(f)
            _state["gallery_coords"] = data.get("galleries", data)
        print("✓ Live Gallery mapping coordinates loaded")

    # Load embedding models for online inference
    print("Loading DINOv2-small for image queries...")
    _state["image_processor"] = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    _state["image_model"] = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    _state["image_model"].eval()

    print("Loading Nomic Embed Text for text queries...")
    _state["text_tokenizer"] = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    _state["text_model"] = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)
    _state["text_model"].eval()

    print("✓ Server ready.")


# ─── Search Endpoints ────────────────────────────────────────────────────────

class TextQuery(BaseModel):
    query: str
    top_k: int = 10


def search_faiss(query_embedding: np.ndarray, index_type: str = "visual", top_k: int = 10) -> list[dict]:
    """Search FAISS index and return matched metadata."""
    import faiss
    if index_type == "visual":
        index = _state.get("index_visual")
        mapping = _state.get("mapping_visual")
    else:
        index = _state.get("index_semantic")
        mapping = _state.get("mapping_semantic")

    metadata = _state["metadata"]
    if index is None or metadata is None or mapping is None:
        return []

    # Request extra candidates to account for filtered out items
    fetch_k = min(top_k * 5, index.ntotal)
    if fetch_k == 0:
        return []
        
    distances, indices = index.search(query_embedding, fetch_k)
    results = []
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(mapping):
            continue
            
        real_idx = mapping[idx]
        row = metadata.iloc[real_idx]
        
        # Filter out "The Cloisters"
        dept = str(row.get("department", ""))
        g_num = str(row.get("GalleryNumber", ""))
        if "Cloister" in dept or g_num.startswith("00"):
            continue

        # Override hardcoded parquet locations with live Apple Vision coordinates
        live_coords = {}
        if _state.get("gallery_coords") and g_num in _state["gallery_coords"]:
            live_coords = _state["gallery_coords"][g_num]

        results.append({
            "rank": len(results) + 1,
            "score": float(dist),
            "objectID": int(row.get("objectID", 0)),
            "title": str(row.get("title", "")),
            "artistDisplayName": str(row.get("artistDisplayName", "")),
            "medium": str(row.get("medium", "")),
            "department": str(row.get("department", "")),
            "GalleryNumber": g_num,
            "primaryImage": str(row.get("primaryImage", "")),
            "primaryImageSmall": str(row.get("primaryImageSmall", "")),
            "description": str(row.get("description", "")) if pd.notna(row.get("description")) else "",
            "culture": str(row.get("culture", "")) if pd.notna(row.get("culture")) else "",
            "period": str(row.get("period", "")) if pd.notna(row.get("period")) else "",
            "objectURL": str(row.get("objectURL", "")),
            "floor": live_coords.get("floor", str(row.get("floor", ""))),
            "map_file": live_coords.get("map_file", str(row.get("map_file", ""))),
            "x_pct": float(live_coords.get("x_pct", row.get("x_pct", 0))) if live_coords.get("x_pct") or pd.notna(row.get("x_pct")) else None,
            "y_pct": float(live_coords.get("y_pct", row.get("y_pct", 0))) if live_coords.get("y_pct") or pd.notna(row.get("y_pct")) else None,
        })
        
        if len(results) >= top_k:
            break
            
    return results


@app.post("/search/text")
async def search_text(query: TextQuery):
    """Search by text query."""
    model = _state["model"]
    if model is None or (_state["index_semantic"] is None and _state["index_visual"] is None):
        return {"error": "Model not loaded. Train the model first.", "results": {"semantic": [], "visual": []}}

    device = _state["device"]
    tokenizer = _state["text_tokenizer"]
    text_model = _state["text_model"]

    # Embed twice: once for Semantic (query-style), once for Visual (document-style for projector)
    doc_query = f"search_query: {query.query}"
    doc_project = f"search_document: {query.query}"
    
    inputs_query = tokenizer([doc_query], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    inputs_doc = tokenizer([doc_project], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_query = text_model(**inputs_query)
        out_doc = text_model(**inputs_doc)

    def pool_emb(outputs, attention_mask):
        token_emb = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return torch.sum(token_emb * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

    text_emb_query = pool_emb(out_query, inputs_query["attention_mask"])
    text_emb_doc = pool_emb(out_doc, inputs_doc["attention_mask"])

    # 1. Semantic Search (768-d)
    # L2 normalize raw query embedding for semantic cosine similarity
    text_emb_norm = F.normalize(text_emb_query, p=2, dim=1)
    query_vec_raw = text_emb_norm.cpu().detach().numpy().astype(np.float32)
    results_semantic = search_faiss(query_vec_raw, "semantic", query.top_k)

    # 2. Visual Search (512-d)
    # Project document-style embedding through Contrastive Model
    text_projected = F.normalize(model.text_projector(text_emb_doc.to(device)), p=2, dim=1)
    query_vec_proj = text_projected.cpu().detach().numpy().astype(np.float32)
    results_visual = search_faiss(query_vec_proj, "visual", query.top_k)

    return {"query": query.query, "results": {"semantic": results_semantic, "visual": results_visual}}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = 10):
    """Search by uploaded image."""
    model = _state["model"]
    if model is None or (_state["index_semantic"] is None and _state["index_visual"] is None):
        return {"error": "Model not loaded. Train the model first.", "results": {"semantic": [], "visual": []}}

    device = _state["device"]
    processor = _state["image_processor"]
    image_model = _state["image_model"]

    # Read and process the image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = image_model(**inputs)
    image_emb = outputs.last_hidden_state[:, 0, :]  # CLS token (raw DINOv2)

    # Search categories:
    # For image search, Semantic (image query vs text index) is still cross-modal but uses projected space.
    # Visual (image query vs image index) is direct image-to-image but in projected space for consistency.

    # Project image through contrastive model
    img_projected = F.normalize(model.image_projector(image_emb.to(device)), p=2, dim=1)
    query_vec_proj = img_projected.cpu().detach().numpy().astype(np.float32)

    # Note: Semantic search with image query uses text_projected.pt index if we want cross-modal.
    # But we previously updated index_semantic to use text_unprojected (768).
    # This means image query (384) cannot directly search 768 semantic index without project/reshape.
    # So for image queries, we MUST use the projected space (512) for semantic search if we want it.
    
    # Actually, for image search, let's keep it simple: Search Visual Match using projected.
    results_visual = search_faiss(query_vec_proj, "visual", top_k)
    
    # If the user wants semantic match for images, we'd need another index for text_projected.
    # We'll stick to Visual for image uploads as it's the most common mental model.
    return {"results": {"semantic": [], "visual": results_visual}}


# ─── Training Endpoint ───────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int = 50
    batch_size: int = 512
    lr: float = 5e-4
    joint_dim: int = 512


@app.post("/train")
async def start_training(req: TrainRequest):
    """Trigger model training in a background thread."""
    if _state["training_status"] == "training":
        return {"status": "already_training"}

    _state["training_status"] = "training"
    _state["training_error"] = None

    def train_worker():
        try:
            dm = MetDataModule(batch_size=req.batch_size)
            model = ContrastiveModel(lr=req.lr, d_joint=req.joint_dim)

            telemetry = TelemetryCallback()
            from lightning.pytorch.callbacks import EarlyStopping
            early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.001)

            trainer = L.Trainer(
                max_epochs=req.epochs,
                accelerator="auto",
                devices=1,
                default_root_dir="checkpoints",
                log_every_n_steps=1,
                callbacks=[telemetry, early_stop],
            )

            trainer.fit(model, dm)
            trainer.save_checkpoint(str(CKPT_DIR / "contrastive_final.ckpt"))

            # Re-project embeddings
            model.eval()
            device = next(model.parameters()).device
            all_images = torch.load(DATA_DIR / "images_unprojected.pt", map_location="cpu")
            all_texts = torch.load(DATA_DIR / "text_unprojected.pt", map_location="cpu")
            with torch.no_grad():
                img_proj = F.normalize(model.image_projector(all_images.to(device)), p=2, dim=1).cpu()
                txt_proj = F.normalize(model.text_projector(all_texts.to(device)), p=2, dim=1).cpu()
            torch.save(img_proj, DATA_DIR / "images_projected.pt")
            torch.save(txt_proj, DATA_DIR / "text_projected.pt")

            # Rebuild FAISS indices
            import faiss
            if _state["mapping_visual"] is not None:
                vis_emb = img_proj.numpy().astype(np.float32)[_state["mapping_visual"]]
                index_visual = faiss.IndexFlatIP(vis_emb.shape[1])
                index_visual.add(vis_emb)
                _state["index_visual"] = index_visual
                
            if _state["mapping_semantic"] is not None:
                # Use raw text embeddings for semantic index, normalized
                sem_emb_raw = all_texts.to(device)
                sem_emb_norm = F.normalize(sem_emb_raw, p=2, dim=1).cpu().numpy().astype(np.float32)
                sem_emb = sem_emb_norm[_state["mapping_semantic"]]
                index_semantic = faiss.IndexFlatIP(sem_emb.shape[1])
                index_semantic.add(sem_emb)
                _state["index_semantic"] = index_semantic
                
            _state["model"] = model.cpu()

            _state["training_status"] = "complete"
        except Exception as e:
            _state["training_status"] = "error"
            _state["training_error"] = str(e)

    thread = threading.Thread(target=train_worker, daemon=True)
    thread.start()
    return {"status": "training_started", "epochs": req.epochs}


@app.get("/status")
async def get_status():
    return {
        "training_status": _state["training_status"],
        "training_error": _state["training_error"],
        "index_visual_size": _state["index_visual"].ntotal if _state["index_visual"] else 0,
        "index_semantic_size": _state["index_semantic"].ntotal if _state["index_semantic"] else 0,
        "metadata_size": len(_state["metadata"]) if _state["metadata"] is not None else 0,
    }


# ─── Gallery Map Data ─────────────────────────────────────────────────────────

@app.get("/gallery")
async def get_gallery(page: int = 1, page_size: int = 50, gallery: str = None):
    """Return paginated processed artworks."""
    metadata = _state["metadata"]
    if metadata is None:
        return {"results": [], "total": 0, "page": page, "pages": 0}

    # Filter out "The Cloisters"
    mask = ~metadata['department'].fillna("").astype(str).str.contains("Cloister", na=False) & \
           ~metadata['GalleryNumber'].fillna("").astype(str).str.startswith("00")
           
    if gallery:
        # Strip trailing decimal zeros safely and strip whitespace
        metadata_gallery_str = metadata['GalleryNumber'].fillna("").astype(str).str.strip()
        query_str = str(gallery).strip()
        
        # Handle cases where pandas casted "899" as "899.0" by slicing `.0` out 
        metadata_gallery_str = metadata_gallery_str.str.replace(r'\.0$', '', regex=True)
        query_str = query_str.replace('.0', '')
        
        mask = mask & (metadata_gallery_str == query_str)

    filtered_df = metadata[mask]
    total = len(filtered_df)

    start = (page - 1) * page_size
    end = start + page_size
    page_df = filtered_df.iloc[start:end]

    results = []
    for _, row in page_df.iterrows():
        results.append({
            "objectID": int(row.get("objectID", 0)),
            "title": str(row.get("title", "")),
            "artistDisplayName": str(row.get("artistDisplayName", "")),
            "medium": str(row.get("medium", "")),
            "department": str(row.get("department", "")),
            "GalleryNumber": str(row.get("GalleryNumber", "")),
            "primaryImage": str(row.get("primaryImage", "")),
            "primaryImageSmall": str(row.get("primaryImageSmall", "")),
            "description": str(row.get("description", "")) if pd.notna(row.get("description")) else "",
            "culture": str(row.get("culture", "")) if pd.notna(row.get("culture")) else "",
            "period": str(row.get("period", "")) if pd.notna(row.get("period")) else "",
            "objectURL": str(row.get("objectURL", "")),
            "floor": str(row.get("floor", "")) if pd.notna(row.get("floor")) else "",
            "map_file": str(row.get("map_file", "")) if pd.notna(row.get("map_file")) else "",
            "x_pct": float(row.get("x_pct", 0)) if pd.notna(row.get("x_pct")) else None,
            "y_pct": float(row.get("y_pct", 0)) if pd.notna(row.get("y_pct")) else None,
        })
    pages = (total + page_size - 1) // page_size if page_size > 0 else 0
    return {"results": results, "total": total, "page": page, "pages": pages}


@app.get("/gallery-map")
async def get_gallery_map():
    """Return the gallery coordinate mapping."""
    coords_path = DATA_DIR / "gallery_coords.json"
    if coords_path.exists():
        with open(coords_path) as f:
            data = json.load(f)
        return data["galleries"]
    return {}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
