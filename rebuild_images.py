import torch
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import time
from pathlib import Path
from tqdm import tqdm
import os

DATA_DIR = Path("data")
df = pd.read_parquet(DATA_DIR / "met_final.parquet")
print(f"Loaded {len(df)} artworks from met_final.parquet")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
model.eval()

batch_size = 256
all_embeddings = []

for i in tqdm(range(0, len(df), batch_size), desc="Image Embeddings"):
    batch_df = df.iloc[i : i + batch_size]
    images = []
    
    for _, row in batch_df.iterrows():
        img_path = DATA_DIR / f"images/{row['objectID']}.jpg"
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # (N, 384)
    all_embeddings.append(embeddings.cpu())

image_tensor = torch.cat(all_embeddings, dim=0)
print(f"Final Image Tensor Shape: {image_tensor.shape}")
torch.save(image_tensor, DATA_DIR / "images_unprojected.pt")
print("✅ Successfully rebuilt images_unprojected.pt!")
