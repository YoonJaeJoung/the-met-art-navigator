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
    "faiss_index": None,
    "metadata": None,
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

    # Load FAISS index
    projected_path = DATA_DIR / "images_projected.pt"
    if projected_path.exists():
        import faiss
        embeddings = torch.load(projected_path, map_location="cpu").numpy().astype(np.float32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        _state["faiss_index"] = index
        print(f"✓ FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Load metadata
    metadata_path = DATA_DIR / "met_final.parquet"
    if not metadata_path.exists():
        metadata_path = DATA_DIR / "met_enriched.parquet"
    if metadata_path.exists():
        _state["metadata"] = pd.read_parquet(metadata_path)
        print(f"✓ Metadata loaded: {len(_state['metadata'])} records")

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


def search_faiss(query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
    """Search FAISS index and return matched metadata."""
    import faiss
    index = _state["faiss_index"]
    metadata = _state["metadata"]
    if index is None or metadata is None:
        return []

    # Request extra candidates to account for filtered out items
    fetch_k = min(top_k * 5, len(metadata))
    distances, indices = index.search(query_embedding, fetch_k)
    results = []
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata.iloc[idx]
        
        # Filter out "The Cloisters"
        dept = str(row.get("department", ""))
        g_num = str(row.get("GalleryNumber", ""))
        if "Cloister" in dept or g_num.startswith("00"):
            continue
            
        results.append({
            "rank": len(results) + 1,
            "score": float(dist),
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
            "floor": str(row.get("floor", "")),
            "map_file": str(row.get("map_file", "")),
            "x_pct": float(row.get("x_pct", 0)) if pd.notna(row.get("x_pct")) else None,
            "y_pct": float(row.get("y_pct", 0)) if pd.notna(row.get("y_pct")) else None,
        })
        
        if len(results) >= top_k:
            break
            
    return results


@app.post("/search/text")
async def search_text(query: TextQuery):
    """Search by text query."""
    model = _state["model"]
    if model is None or _state["faiss_index"] is None:
        return {"error": "Model not loaded. Train the model first.", "results": []}

    device = _state["device"]
    tokenizer = _state["text_tokenizer"]
    text_model = _state["text_model"]

    # Embed the query
    doc = f"search_query: {query.query}"
    inputs = tokenizer([doc], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    attention_mask = inputs["attention_mask"]
    token_emb = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    text_emb = torch.sum(token_emb * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

    # Project through contrastive model
    text_projected = F.normalize(model.text_projector(text_emb.to(device)), p=2, dim=1)
    query_vec = text_projected.cpu().detach().numpy().astype(np.float32)

    results = search_faiss(query_vec, query.top_k)
    return {"query": query.query, "results": results}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = 10):
    """Search by uploaded image."""
    model = _state["model"]
    if model is None or _state["faiss_index"] is None:
        return {"error": "Model not loaded. Train the model first.", "results": []}

    device = _state["device"]
    processor = _state["image_processor"]
    image_model = _state["image_model"]

    # Read and process the image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = image_model(**inputs)
    image_emb = outputs.last_hidden_state[:, 0, :]  # CLS token

    # Project through contrastive model
    img_projected = F.normalize(model.image_projector(image_emb.to(device)), p=2, dim=1)
    query_vec = img_projected.cpu().detach().numpy().astype(np.float32)

    results = search_faiss(query_vec, top_k)
    return {"results": results}


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

            # Rebuild FAISS index
            import faiss
            embeddings = img_proj.numpy().astype(np.float32)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            _state["faiss_index"] = index
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
        "index_size": _state["faiss_index"].ntotal if _state["faiss_index"] else 0,
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
        mask = mask & (metadata['GalleryNumber'].fillna("").astype(str) == str(gallery))

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
