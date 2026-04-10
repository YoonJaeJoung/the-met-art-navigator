import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import time
from pathlib import Path

DATA_DIR = Path("data")
df = pd.read_parquet(DATA_DIR / "met_final.parquet").head(256)
docs = ["search_document: " + str(r.get("title","")) for _, r in df.iterrows()]

device = "mps"
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)

start = time.time()
inputs = tokenizer(docs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
print(f"Time for 256 text docs: {time.time() - start:.2f}s")
