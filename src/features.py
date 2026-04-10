"""
Stage 4: Feature Extraction.

Extracts image embeddings (DINOv2-small, dim=384) and text embeddings
(Nomic Embed Text v1.5, dim=768) in mini-batches. Saves unprojected
tensors to disk.

Usage:
    .venv/bin/python src/features.py [--batch-size 64] [--device auto]
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
PROGRESS_FILE = Path("progress.md")


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def update_progress(batch_num: int, images_done: int, texts_done: int):
    """Append a row to the Stage 4 table in progress.md."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = f"| {batch_num} | {images_done} | {texts_done} | {ts} |"

    content = PROGRESS_FILE.read_text()
    marker_stage4 = "| — | — | — | — |"
    # Find the LAST occurrence (Stage 4 table) since Stage 3 also has a 4-column table
    parts = content.rsplit(marker_stage4, 1)
    if len(parts) == 2:
        content = parts[0] + row + "\n" + marker_stage4 + parts[1]
    PROGRESS_FILE.write_text(content)


def build_text_document(row: pd.Series) -> str:
    """Concatenate metadata fields into a single text document for embedding."""
    parts = []
    for field in ["title", "artistDisplayName", "medium", "department", "culture", "period", "classification", "description"]:
        val = str(row.get(field, "")).strip()
        if val:
            parts.append(val)
    tags = str(row.get("tags", "")).strip()
    if tags:
        parts.append(tags.replace("|", ", "))
    return " | ".join(parts)


def extract_image_embeddings(df: pd.DataFrame, device: torch.device, batch_size: int) -> tuple[torch.Tensor, list[bool]]:
    """Extract DINOv2 embeddings for all images. Returns (embeddings, valid_mask)."""
    print("Loading DINOv2-small model...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    model.eval()

    checkpoint_path = DATA_DIR / "images_unprojected_checkpoint.pt"
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path)
        all_embeddings = ckpt["embeddings"]
        valid_mask = ckpt["valid_mask"]
        print(f"[Resume] Found checkpoint for image embeddings with {len(valid_mask)} processed.")
    else:
        all_embeddings = []
        valid_mask = []

    start_idx = len(valid_mask)
    batch_num = start_idx // batch_size + 1

    for i in tqdm(range(start_idx, len(df), batch_size), desc="Image embeddings"):
        batch_df = df.iloc[i : i + batch_size]
        images = []
        batch_valid = []

        for _, row in batch_df.iterrows():
            img_path = IMAGES_DIR / f"{row['objectID']}.jpg"
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    batch_valid.append(True)
                except Exception:
                    batch_valid.append(False)
            else:
                batch_valid.append(False)

        valid_mask.extend(batch_valid)

        if images:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]  # (N, 384)
            all_embeddings.append(embeddings.cpu())

        update_progress(batch_num, sum(valid_mask), 0)
        batch_num += 1

        # Checkpoint every batch
        torch.save({
            "embeddings": all_embeddings,
            "valid_mask": valid_mask
        }, checkpoint_path)

    if all_embeddings:
        image_tensor = torch.cat(all_embeddings, dim=0)
    else:
        image_tensor = torch.zeros(0, 384)

    return image_tensor, valid_mask


def extract_text_embeddings(df: pd.DataFrame, device: torch.device, batch_size: int) -> torch.Tensor:
    """Extract Nomic text embeddings for all artworks."""
    print("Loading Nomic Embed Text v1.5 model...")
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)
    model.eval()

    # Build text documents
    documents = [build_text_document(row) for _, row in df.iterrows()]
    # Nomic recommends prefixing with task type
    documents = [f"search_document: {doc}" for doc in documents]

    checkpoint_path = DATA_DIR / "text_unprojected_checkpoint.pt"
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path)
        all_embeddings = ckpt["embeddings"]
        print(f"[Resume] Found checkpoint for text embeddings with {len(all_embeddings)*batch_size} approx processed.")
    else:
        all_embeddings = []

    start_batch_idx = len(all_embeddings)
    start_idx = start_batch_idx * batch_size

    for i in tqdm(range(start_idx, len(documents), batch_size), desc="Text embeddings"):
        batch_docs = documents[i : i + batch_size]
        inputs = tokenizer(batch_docs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling over non-padding tokens
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        all_embeddings.append(embeddings.cpu())

        # Checkpoint every batch
        torch.save({
            "embeddings": all_embeddings
        }, checkpoint_path)

    if all_embeddings:
        text_tensor = torch.cat(all_embeddings, dim=0)
    else:
        text_tensor = torch.zeros(0, 768)

    return text_tensor


def main(batch_size: int = 64, device_str: str = "auto"):
    device = get_device(device_str)
    print(f"Using device: {device}")

    # Load enriched metadata
    parquet_path = DATA_DIR / "met_enriched.parquet"
    if not parquet_path.exists():
        parquet_path = DATA_DIR / "met_verified.parquet"
    if not parquet_path.exists():
        print("Error: No metadata Parquet found. Run previous pipeline stages first.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} artwork records.")

    # Image embeddings
    print("\n=== Image Embeddings (DINOv2-small) ===")
    image_embeddings, valid_mask = extract_image_embeddings(df, device, batch_size)
    print(f"Image embeddings shape: {image_embeddings.shape}")

    # Text embeddings (only for valid images to keep alignment)
    print("\n=== Text Embeddings (Nomic Embed Text v1.5) ===")
    # Filter to rows with valid images
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    text_embeddings = extract_text_embeddings(df_valid, device, batch_size)
    print(f"Text embeddings shape: {text_embeddings.shape}")

    assert image_embeddings.shape[0] == text_embeddings.shape[0], \
        f"Shape mismatch: images={image_embeddings.shape[0]}, texts={text_embeddings.shape[0]}"

    # Save tensors
    torch.save(image_embeddings, DATA_DIR / "images_unprojected.pt")
    torch.save(text_embeddings, DATA_DIR / "text_unprojected.pt")

    # Save the valid-only metadata for alignment
    df_valid.to_parquet(DATA_DIR / "met_final.parquet", index=False)

    # Clean checkpoints
    for f in [DATA_DIR / "images_unprojected_checkpoint.pt", DATA_DIR / "text_unprojected_checkpoint.pt"]:
        if f.exists():
            f.unlink()

    print(f"\n✓ Saved {image_embeddings.shape[0]} paired embeddings:")
    print(f"  data/images_unprojected.pt  ({image_embeddings.shape})")
    print(f"  data/text_unprojected.pt    ({text_embeddings.shape})")
    print(f"  data/met_final.parquet      ({len(df_valid)} rows)")

    # Update progress
    content = PROGRESS_FILE.read_text()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = content.replace(
        "## Stage 4: Feature Extraction (`features.py`)\n**Status**: Not started",
        f"## Stage 4: Feature Extraction (`features.py`)\n**Status**: ✅ Complete — {image_embeddings.shape[0]} pairs ({ts})",
    )
    PROGRESS_FILE.write_text(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image & text embeddings")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda/mps")
    args = parser.parse_args()
    main(batch_size=args.batch_size, device_str=args.device)
