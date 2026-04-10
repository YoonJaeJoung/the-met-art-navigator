# The Met Art Navigator — Implementation Plan

## Goal

Build a cross-modal art retrieval system for The Metropolitan Museum of Art. Users can search by uploading an image or typing an artwork style. Matched artworks are displayed on actual Met floor maps with gallery pin markers. The system includes a "Train" button with a live training-progress dashboard streamed via WebSockets.

---

## 1. Data Engineering Pipeline (Offline — Mini-Batch)

All API calls and heavy I/O are processed in **configurable mini-batches** (e.g. 50–200 objects per batch). A persistent `progress.md` file is updated after every batch to report counts, errors, and resumability.

### Stage 1: Metadata Ingestion (`src/ingest_met.py`)

| Step | Detail |
|------|--------|
| 1a | Fetch the full object-ID list from `GET /public/collection/v1/objects`. |
| 1b | In mini-batches, fetch each object via `GET /public/collection/v1/objects/[objectID]`. Filter for `hasImages=true` and non-empty `GalleryNumber` (on-view artworks). |
| 1c | **Cross-Check**: Also query the search endpoint with `isOnView=true` to obtain a second independent list of on-view object IDs. Compare the two lists: <br>• Objects in both lists → **verified on-view dataset** (saved as `data/met_verified.parquet`). <br>• Objects that exist in one list but not the other → **mismatch log** (saved as `data/met_mismatches.csv` with columns: `objectID`, `source`, `GalleryNumber`, `reason`). |
| 1d | Serialize the verified dataset to Parquet with columns: `objectID`, `title`, `artistDisplayName`, `medium`, `department`, `culture`, `period`, `classification`, `GalleryNumber`, `primaryImage`, `primaryImageSmall`, `tags`, `objectURL`. |

### Stage 2: Gallery-to-Map Coordinate Mapping (`src/gallery_mapper.py`)

Since the Met API does **not** provide lat/long or pixel coordinates — only `GalleryNumber` — we build a static mapping JSON (`data/gallery_coords.json`) that associates each gallery number with:

```json
{
  "gallery_number": "521",
  "floor": "1",
  "map_file": "map/floor1.png",
  "x_pct": 0.48,
  "y_pct": 0.32
}
```

- `x_pct` / `y_pct` are **percentage offsets** (0.0–1.0) on the floor image, extracted by visually reading gallery positions from the downloaded map screenshots.
- The script reads the verified Parquet, looks up each artwork's `GalleryNumber` in this JSON, and appends `floor`, `map_file`, `x_pct`, `y_pct` columns. Artworks with unmapped galleries are flagged in `progress.md`.

**Gallery → Floor assignment rules** (derived from map inspection):
- **Floor 1**: galleries 100–169, 300–380, 500–556, 700–746, 900–962
- **Floor 1 Mezzanine**: galleries 170–172, 707, 773–774, 915–916
- **Floor 2**: galleries 173–176, 200–253, 400–464, 535, 600–640, 680–684, 706–722, 750–772, 800–830, 850–851, 899, 917–925, 999
- **Floor 3**: galleries 219–222, 251–253, 708–714

### Stage 3: Image Downloading (`src/download_images.py`)

- Download `primaryImageSmall` URLs in mini-batches (e.g. 200 at a time, 8 concurrent threads).
- Save to `data/images/{objectID}.jpg`.
- Produce manifest `data/images_manifest.parquet` with columns: `objectID`, `filepath`, `success`.
- Update `progress.md` after each batch with download counts.

### Stage 4: Feature Extraction (`src/features.py`)

- **Image embeddings**: `dinov2-small` (dim=384) via HuggingFace `transformers`.
- **Text embeddings**: `nomic-embed-text-v1.5` (dim=768) over concatenated metadata string (`title | artistDisplayName | medium | department | culture | period | tags`).
- Output: `data/images_unprojected.pt`, `data/text_unprojected.pt`.
- Mini-batch GPU processing (batch_size=64). Update `progress.md` after each batch.

---

## 2. Machine Learning Architecture

### Contrastive Model (`src/lit_model.py`)

| Component | Detail |
|-----------|--------|
| Framework | PyTorch Lightning |
| Image projector | `nn.Linear(384, 512)` → L2 normalize |
| Text projector  | `nn.Linear(768, 512)` → L2 normalize |
| Loss | InfoNCE / CLIP-style contrastive loss with learnable temperature |
| Data Module | `MetDataModule` — loads paired (image, text) tensors, filters missing images via boolean mask |

### Training Telemetry (`src/telemetry.py`)

A PyTorch Lightning `Callback` that:
1. On each `on_train_batch_end`, serializes `{"epoch", "step", "loss", "lr"}` to JSON.
2. Sends the JSON packet over a WebSocket server (port 8765).
3. The React frontend connects to this WebSocket and plots a live loss curve.

---

## 3. Inference & Online Serving

### Backend API (`src/serve.py`)

- **Framework**: FastAPI + uvicorn
- **Vector Index**: `faiss-cpu` `IndexFlatIP` (inner-product = cosine similarity on L2-normalized vectors)
- **Endpoints**:
  - `POST /search/text` — accept text query, embed with Nomic, project, search FAISS, return top-K results with gallery coords.
  - `POST /search/image` — accept uploaded image, embed with DINOv2, project, search FAISS, return top-K results with gallery coords.
  - `POST /train` — trigger model training in a background task; telemetry streams via the WebSocket.
  - `GET /status` — return training status (idle / running / complete).

---

## 4. Frontend (React + Vite)

### Search Panel
- **Text input** for style/keyword queries.
- **Image uploader** (drag-and-drop) for reverse-image search.
- Submit button fires the appropriate `/search/*` endpoint.

### Results & Map View
- A **results sidebar** showing matched artwork cards (thumbnail, title, artist, gallery#).
- A **floor-plan viewer** rendering the appropriate `map/floorX.png` image.
- **Pin markers** overlaid at the `(x_pct, y_pct)` position for each matched artwork's gallery.
- Floor tab selector (Floor 1 / 1M / 2 / 3).

### Training Dashboard
- A **"Train Model"** button that POSTs to `/train`.
- A real-time **line chart** (using Chart.js or Recharts) plotting loss vs. step, fed by the WebSocket.
- Status indicator (idle → training → complete).

---

## 5. File Structure

```
the-met-art-navigator/
├── README.md
├── REFERENCE.md
├── plan.md
├── progress.md                  # auto-updated batch progress log
├── map/
│   ├── floor1.png
│   ├── floor1M.png
│   ├── floor2.png
│   └── floor3.png
├── data/
│   ├── gallery_coords.json      # gallery_number → floor + (x_pct, y_pct)
│   ├── met_verified.parquet     # verified on-view metadata
│   ├── met_mismatches.csv       # cross-check mismatches
│   ├── images/                  # cached artwork JPEGs
│   ├── images_manifest.parquet
│   ├── images_unprojected.pt
│   └── text_unprojected.pt
├── src/
│   ├── ingest_met.py
│   ├── gallery_mapper.py
│   ├── download_images.py
│   ├── features.py
│   ├── lit_model.py
│   ├── telemetry.py
│   └── serve.py
├── frontend/                    # React + Vite app
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── SearchPanel.jsx
│   │   │   ├── ResultsSidebar.jsx
│   │   │   ├── MapViewer.jsx
│   │   │   └── TrainingDashboard.jsx
│   │   └── ...
│   └── ...
└── pyproject.toml / requirements.txt
```

---

## 6. Primary Dependencies

| Category | Libraries |
|----------|-----------|
| ML Core | `torch`, `torchvision`, `lightning` |
| Embeddings | `transformers` (DINOv2), `nomic-embed-text` (Nomic) |
| Vector Search | `faiss-cpu` |
| Backend | `fastapi`, `uvicorn`, `websockets` |
| Data | `pandas`, `fastparquet`, `aiohttp` |
| Frontend | React, Vite, Recharts (or Chart.js) |

---

## 7. Verification Plan

### Automated
- `ingest_met.py` produces `met_verified.parquet` with >0 rows and valid `GalleryNumber` in every row.
- `met_mismatches.csv` is generated with clear `reason` column.
- Tensor shapes: `images_unprojected.pt` → `(N, 384)`, `text_unprojected.pt` → `(N, 768)`.

### Manual
- Press "Train" in the UI → live loss chart updates in real time.
- Search for "impressionist painting" → results appear pinned on floor plan at correct gallery locations.
- Upload an image → similar artworks returned with map pins.
