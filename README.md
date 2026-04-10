# MET Guide
## General Information

This project, as a part of the NYU Machine Learning course, aims to learn and use cross modal retrieval method to retrieve artworks from the MET museum dataset.

I've used professor's project code [the-met-retrieval](https://github.com/kyunghyuncho/the-met-retrieval/tree/main) as a reference to build this project.

But in detailed use case - I targeted students from modern art course, so they can visit the MET museum and use this app to find artworks related to what they've learned in class.

---

## 🚀 Pipeline Status & Performance

The data pipeline for this project was built to handle The Met's massive collection efficiently. Due to the high computational cost of text embeddings (Nomic BERT), the extraction was offloaded to a remote GPU Studio.

| Metric | Result |
|--------|--------|
| **Collection Scan** | 501,094 object IDs scanned |
| **Verified Artworks** | 33,368 on-view artworks identified |
| **Final Dataset** | 27,539 artworks (with valid high-res images & embeddings) |
| **Metadata Ingestion** | ~1.5 hours (local) |
| **Feature Extraction** | **9 minutes** (Accelerated via Lightning AI T4 GPU) |
| **Local Fallback ETA** | ~16 hours (CPU/MPS) |

> [!TIP]
> **Cloud Acceleration**: Stage 4 text embedding was migrated to a **Lightning AI T4 Studio**. By using an Nvidia Tesla T4 GPU, I processed 27.5k BERT embeddings in a single 9-minute sweep, bypassing the 16-hour bottleneck on local M-series silicon.

---

## How to Run (Step-by-Step)

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- A machine with a GPU is recommended for feature extraction and training (CPU fallback is supported but slow)
- `uv` package manager for Python

### 1. Set up the Python environment
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run the offline data pipeline (in order)

Each script processes data in mini-batches and updates `progress.md` with status.

```bash
# Stage 1 — Ingest metadata from The Met API & cross-check on-view status
.venv/bin/python src/ingest_met.py

# Stage 2 (OPTIONAL) — Map gallery numbers to floor-plan pixel coordinates
# Note: The highly calibrated 'data/gallery_coords.json' mapping file is already included 
# in this repository out-of-the-box so you can skip this stage entirely!
# 
# If you want to rebuild it from scratch yourself, this script automates the process using:
# 1. ocrmac (Apple Vision framework) to extract text bounding boxes from floor maps
# 2. Targeted Affine Transformations to project detailed screenshots using trusted anchor points
# 3. An Interactive FastAPI WebApp to manually pin remaining galleries cleanly 
# .venv/bin/python src/gallery_mapper.py

# Stage 3 — Download artwork images
.venv/bin/python src/download_images.py

# Stage 4 — Extract image & text embeddings
# Accelerated using Remote GPU (Nvida T4) for Nomic-BERT
.venv/bin/python src/features.py
```

After each stage completes, check `progress.md` for batch-level summaries and any errors.

### 3. Start the backend server
```bash
.venv/bin/python src/serve.py
```
The FastAPI server starts on `http://localhost:8000`. The training telemetry WebSocket runs on `ws://localhost:8765`.

### 4. Start the frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

### 5. Using the app
1. **Search by text**: Type a style, period, or keyword (e.g. "impressionist landscape") and press Search.
2. **Search by image**: Drag-and-drop or upload an artwork image and press Search.
3. **View results on the map**: Matched artworks appear as pins on the Met floor plan. Switch floors using the tab selector.
4. **Train the model**: Click the "Train" button to start contrastive model training. A live loss chart updates in real-time via WebSocket. 
   > [!NOTE]
   > **Early Stopping**: The trainer includes an automated "Early Stopping" monitor. If the validation loss stops improving for 5 epochs, the training will finish early to prevent overfitting and save the best model weights.
5. **Explore the Gallery**: Browse the entire processed collection with high-performance **pagination**.
6. **View Artwork Details**: Click any artwork in the search results or gallery to see a detailed modal with descriptions, medium, culture, and a direct link to the Met website.

---

## Machine Learning — What & Why

### Overview
This project uses **cross-modal retrieval** to bridge the gap between text descriptions and artwork images. Given a text query (e.g. "cubist still life") or an uploaded image, the system finds the most visually or semantically similar artworks in The Met's on-view collection and shows their physical location in the museum.

### Foundation Models Used

| Model | Role | Output Dim | Source |
|-------|------|-----------|--------|
| **DINOv2-small** | Image feature extraction | 384 | Meta AI — self-supervised vision transformer trained on 142M images. Captures rich visual semantics (texture, color, composition) without task-specific labels. |
| **Nomic Embed Text v1.5** | Text feature extraction | 768 | Nomic AI — long-context text embedding model. Encodes artwork metadata (title, artist, medium, culture, period, tags) into dense semantic vectors. |

### Contrastive Learning (InfoNCE / CLIP-style)
The two foundation models produce embeddings in **different** vector spaces (384-d for images, 768-d for text). A custom two-tower projection network aligns them into a **shared 512-d latent space**:

1. **Image tower**: `Linear(384 → 512)` → `BatchNorm` → `GELU` → `Dropout(0.3)` → `Linear(512 → 512)` → L2-normalize
2. **Text tower**: `Linear(768 → 512)` → `BatchNorm` → `GELU` → `Dropout(0.3)` → `Linear(512 → 512)` → L2-normalize
3. **Loss**: InfoNCE contrastive loss — for each batch of (image, text) pairs, compute the cosine similarity matrix and minimize cross-entropy so that matching pairs score highest. A learnable temperature parameter scales the logits.

### Advanced Training Features
*   **Regularization**: Integrated **Dropout** and **Batch Normalization** to ensure the model generalizes well to unseen art styles.
*   **Early Stopping**: Automatically halts training when the validation loss plateaus, ensuring you always get the most optimized version of the model.
*   **Synchronized Metrics**: A clean epoch-by-epoch summary is saved to **`data/metrics_summary.csv`**, allowing for easy comparison of training vs. validation loss in Excel/Sheets.

This is trained with **PyTorch Lightning**, which handles data loading, gradient descent, checkpointing, and distributed training abstractions.

### Vector Search at Inference
Once trained, all artwork embeddings are projected into the shared 512-d space and indexed with **FAISS** (`IndexFlatIP` — exact inner-product search). Because vectors are L2-normalized, inner product equals cosine similarity. A user query (text or image) is embedded, projected, and the top-K nearest neighbors are retrieved in milliseconds.

### Key ML Libraries

| Library | Purpose |
|---------|---------|
| `torch` / `torchvision` | Tensor operations, image transforms, neural network layers, gradient descent |
| `lightning` | Structured training loops, data modules, callbacks, checkpointing |
| `transformers` | Pre-trained DINOv2 weights, image processors, tokenizers |
| `faiss-cpu` | High-performance vector similarity search index |
| `pandas` / `fastparquet` | Tabular data storage and manipulation for the data pipeline |
| `websockets` | Real-time training metric streaming to the frontend |
