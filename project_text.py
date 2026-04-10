import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, str(Path("src")))
from lit_model import ContrastiveModel

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")

def main():
    device = torch.device("cpu")
    ckpt_path = CKPT_DIR / "contrastive_final.ckpt"
    model = ContrastiveModel.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()

    text_unproj = torch.load(DATA_DIR / "text_unprojected.pt", map_location=device)
    print(f"Loaded text embeddings: {text_unproj.shape}")

    batch_size = 1024
    projected_chunks = []
    
    with torch.no_grad():
        for i in range(0, text_unproj.shape[0], batch_size):
            chunk = text_unproj[i:i+batch_size]
            proj = F.normalize(model.text_projector(chunk), p=2, dim=1)
            projected_chunks.append(proj)
    
    text_projected = torch.cat(projected_chunks, dim=0)
    print(f"Projected text embeddings: {text_projected.shape}")
    torch.save(text_projected, DATA_DIR / "text_projected.pt")
    print("Done!")

if __name__ == "__main__":
    main()
