import sys
import os
sys.path.append(os.path.abspath('src'))
import torch
import torch.nn.functional as F
import lightning as L
from collections import namedtuple
from transformers import AutoTokenizer, AutoModel

from serve import search_text, TextQuery, _state, DATA_DIR, CKPT_DIR, search_faiss
from lit_model import ContrastiveModel
import pandas as pd
import numpy as np

# Mock the setup
device = torch.device('cpu')
os.chdir('src') # To make paths relative to where serve.py expects, actually paths inside serve.py are relative to CWD. serve.py is run from ROOT usually.
os.chdir('..')

_state["device"] = device
_state["text_tokenizer"] = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
_state["text_model"] = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)
_state["metadata"] = pd.read_parquet("data/met_final.parquet")

model = ContrastiveModel.load_from_checkpoint("checkpoints/contrastive_final.ckpt", map_location=device)
model.eval()

_state["model"] = model.cpu()

import faiss
all_texts = torch.load("data/text_projected.pt", map_location="cpu")
embeddings = all_texts.numpy().astype(np.float32)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
_state["faiss_index"] = index

import asyncio
try:
    res = asyncio.run(search_text(TextQuery(query="impressionist landscape")))
    print("Success:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
