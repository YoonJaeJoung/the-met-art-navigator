# Data Handling and ML Techniques

This document outlines how data is managed, processed, and utilized in the Cross-Modal Antiquities Retrieval System. It also details the machine learning tools and techniques employed to achieve cross-modal (text-to-image and image-to-image) retrieval. The architecture relies on robust offline data engineering and a high-performance serving layer.

---

## 1. Data Engineering Pipeline (Offline)

The backend data engineering pipeline is structured into four sequential stages, decoupling I/O-bound tasks from compute-bound tasks to maximize hardware utilization.

### Stage 1: Metadata Ingestion (`ingest.py`)
- **Source:** Art Institute of Chicago (AIC) Open Access API.
- **Process:** Asynchronously streams and paginates through the AIC REST API.
- **Data Handling:** Raw artifact metadata is downloaded and serialized into a structured tabular format for downstream processing.

### Stage 2: Geocoding (`geocode.py`)
- **Source:** Nominatim / OpenStreetMap.
- **Process:** Parses the "place-of-origin" or related geographic strings from the ingested metadata.
- **Tooling:** Uses `geopy` to resolve text to spatial coordinates (latitude and longitude), later used by the `Deck.gl` frontend mapping visualizations. Data is stored in `data/aic_geocoded.parquet`.

### Stage 3a: Image Downloading (`download_images.py`)
- **Process:** I/O-bound process that downloads raw JPEG images to a local cache (`data/images/`).
- **Parallelism:** Runs heavily parallelized with multiple HTTP threads (e.g., 32 workers) to saturate network bandwidth.
- **Fault-Tolerance:** Outputs a manifest (`data/images_manifest.parquet`) that records successful downloads, ensuring the subsequent GPU-bound stages can gracefully handle missing or corrupt images with zero-tensor fallbacks.

### Stage 3b: Feature Extraction (`features.py`)
- **Process:** Compute-bound GPU process that runs base foundational models over the downloaded data to extract high-dimensional semantic embeddings.
- **Image Embeddings:** Uses `dinov2-small` (via Hugging Face `transformers`) to extract raw visual representations.
- **Text Embeddings:** Uses `nomic-embed-text-v1.5` to extract semantic text embeddings from the serialized document metadata.
- **Output:** Stores the raw, "unprojected" dense vector representations directly to disk as PyTorch tensors (`images_unprojected.pt` and `text_unprojected.pt`).

---

## 2. Machine Learning Architecture

The system utilizes a custom, two-tower projection architecture to align the distinct feature spaces of vision and text.

### The Contrastive Model (`lit_model.py`)
- **Framework:** PyTorch Lightning (`lightning.pytorch`).
- **Objective:** Map the unprojected DINOv2 ($d_{image}=384$) and Nomic ($d_{text}=768$) embeddings into a unified joint latent space of dimension $d_{joint}=512$.
- **Architecture:** Uses simple linear projections (`nn.Linear`) followed by $L_2$ normalization (`F.normalize(..., p=2, dim=1)`).
- **Loss Function:** Employs an **InfoNCE (CLIP-style) contrastive loss**. It calculates the dot-product similarity (cosine similarity, since vectors are $L_2$ normalized) between image and text embeddings in a batch, scaled by a learnable temperature parameter. The model minimizes the cross-entropy loss across both the image-to-text and text-to-image similarity matrices.
- **Data Loading:** Employs a custom `LightningDataModule` (`MetDataModule`) that filters out artifacts lacking a valid image using a boolean mask, ensuring only genuine image-text pairs are passed to the InfoNCE loss function.

---

## 3. Inference and Search (Online)

### Vector Search Engine
- **Tooling:** `faiss-cpu`.
- **Index Type:** `IndexFlatIP` (Exact Inner Product Search).
- **Handling:** Because the PyTorch output tensors are strictly $L_2$ normalized during the forward pass, inner product (IP) search is mathematically equivalent to cosine similarity. This allows for hyper-fast, exact nearest-neighbor lookups in the unified $512$-dimensional joint latent space.

### Asynchronous Telemetry & Serving
- **Backend App:** Built on `FastAPI` and `uvicorn`, communicating asynchronously.
- **Model Training Telemetry:** `lit_model.py` utilizes a custom PyTorch Lightning Callback (`TelemetryCallback`) that streams training metrics loss via WebSockets (using `websockets` library) directly to the Vite/React frontend in real-time.

---

## 4. Primary Machine Learning Dependencies
*   **`torch` / `torchvision`:** Tensor operations, dataset transforms, and model gradient descent.
*   **`lightning`:** Structured model compilation, training loops, data modules, and callbacks.
*   **`transformers`:** Pre-trained weights and tokenizers/processors for DINOv2 and Nomic.
*   **`faiss-cpu`:** High-performance vector index for deployment text-to-image retrieve tasks.
*   **`pandas` / `fastparquet`:** High-performance columnar storage and tabular data manipulation for offline data engineering.