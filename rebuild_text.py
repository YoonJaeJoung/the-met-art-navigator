import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import time
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
df = pd.read_parquet(DATA_DIR / "met_final.parquet")
print(f"Loaded {len(df)} artworks from met_final.parquet")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)
model.eval()

def build_text_document(row):
    parts = []
    for field in ["title", "artistDisplayName", "medium", "department", "culture", "period", "classification", "description"]:
        val = str(row.get(field, "")).strip()
        if val:
            parts.append(val)
    tags = str(row.get("tags", "")).strip()
    if tags:
        parts.append(tags.replace("|", ", "))
    return "search_document: " + " | ".join(parts)

documents = [build_text_document(row) for _, row in df.iterrows()]

batch_size = 256
all_embeddings = []

for i in tqdm(range(0, len(documents), batch_size), desc="Text Embeddings"):
    batch_docs = documents[i : i + batch_size]
    # Restrict max_length to 256 for significantly faster embedding, usually enough for these short art docs
    inputs = tokenizer(batch_docs, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    all_embeddings.append(embeddings.cpu())

text_tensor = torch.cat(all_embeddings, dim=0)
print(f"Final Text Tensor Shape: {text_tensor.shape}")
torch.save(text_tensor, DATA_DIR / "text_unprojected.pt")
print("✅ Successfully rebuilt text_unprojected.pt!")
