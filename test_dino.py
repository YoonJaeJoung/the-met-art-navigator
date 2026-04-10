import torch
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import time
from pathlib import Path

DATA_DIR = Path("data")
df = pd.read_parquet(DATA_DIR / "met_final.parquet").head(256)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to("mps")

images = []
for _, row in df.iterrows():
    img = Image.open(DATA_DIR / f"images/{row['objectID']}.jpg").convert("RGB")
    images.append(img)

start = time.time()
inputs = processor(images=images, return_tensors="pt").to("mps")
with torch.no_grad():
    outputs = model(**inputs)
print(f"Time for 256 images: {time.time() - start:.2f}s")
