#!/bin/bash
export PYTHONUNBUFFERED=1

export VENV_PYTHON="./.venv/bin/python"

echo "Starting data pipeline..."
START_TIME=$(date +%s)

echo "Running Stage 1: Ingestion..."
$VENV_PYTHON src/ingest_met.py --batch-size 200

echo "Running Stage 2: Gallery Mapping..."
$VENV_PYTHON src/gallery_mapper.py

echo "Running Stage 3: Image Downloading..."
$VENV_PYTHON src/download_images.py --batch-size 100 --workers 5

echo "Running Stage 4: Feature Extraction..."
$VENV_PYTHON src/features.py

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

TIME_STR=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

echo -e "\n\nTotal Pipeline Duration: $TIME_STR" >> progress.md
echo "Finished!"
