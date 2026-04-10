---
title: Met Art Navigator API
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

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
   - **Semantic Matches**: Highly accurate text-to-text matching using raw Nomic embeddings.
   - **Visual Matches**: Cross-modal retrieval matching your query against visual image features.
2. **Search by image**: Drag-and-drop or upload an artwork image and press Search.
3. **Toggle Results**: Switch between **Semantic** and **Visual** result categories using the tabs in the sidebar.
4. **View results on the map**: Matched artworks appear as pins on the Met floor plan. Switch floors using the tab selector.
   - **Responsive Scaling**: Pins scale proportionally with the map wide, ensuring precision on both desktop and mobile devices.
   - **Mobile View**: On smaller screens, the layout automatically stacks the map above the results for easier navigation.
5. **Explore the Gallery**: Browse the entire processed collection with high-performance **pagination**.
6. **View Artwork Details**: Click any artwork in the search results or gallery to see a detailed modal with descriptions, medium, culture, and a direct link to the Met website.

---

## Machine Learning — What & Why

The system uses a **Dual Search Engine** strategy:

1. **Semantic Search (High-Fidelity)**: Targeted at artworks **with** detailed descriptions. It uses raw 768-d Nomic text embeddings to preserve full semantic nuances without projection-based loss.
2. **Visual Search (Projected)**: Targeted at artworks **without** descriptions. It uses cross-modal learning to find artworks based on their visual features by projecting queries into a shared 512-d space.

### Advanced Infererence Logic
User queries are embedded once and branched:
- **Query -> RAW Nomic (768-d)**: Matched against the High-Fidelity Semantic Index.
- **Query -> Projected Space (512-d)**: Matched against the Project Visual Index.

### Vector Search at Inference
Artwork embeddings are indexed with **FAISS** (`IndexFlatIP` — exact inner-product search). A user query (text or image) is embedded, branched, and the top nearest neighbors from both semantic and visual pools are retrieved.

### Key ML Libraries

| Library | Purpose |
|---------|---------|
| `torch` / `torchvision` | Tensor operations, image transforms, neural network layers, gradient descent |
| `lightning` | Structured training loops, data modules, callbacks, checkpointing |
| `transformers` | Pre-trained DINOv2 weights, image processors, tokenizers |
| `faiss-cpu` | High-performance vector similarity search index |
| `pandas` / `fastparquet` | Tabular data storage and manipulation for the data pipeline |
| `websockets` | Real-time training metric streaming to the frontend |
