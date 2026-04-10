"""
Stage 2: Gallery Coordinate Mapper.

Reads the verified Parquet and the gallery_coords.json, appends floor/map
coordinate columns to each artwork record.

Usage:
    .venv/bin/python src/gallery_mapper.py
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
PROGRESS_FILE = Path("progress.md")


def main():
    # Load verified dataset
    parquet_path = DATA_DIR / "met_verified.parquet"
    if not parquet_path.exists():
        print("Error: data/met_verified.parquet not found. Run ingest_met.py first.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} verified artworks.")

    # Load gallery coordinate mapping
    coords_path = DATA_DIR / "gallery_coords.json"
    with open(coords_path) as f:
        coords_data = json.load(f)
    gallery_map = coords_data["galleries"]
    print(f"Loaded {len(gallery_map)} gallery coordinate mappings.")

    # Map each artwork's GalleryNumber to floor info
    floors = []
    map_files = []
    x_pcts = []
    y_pcts = []
    unmapped = []

    for _, row in df.iterrows():
        gn = str(row["GalleryNumber"]).strip()
        if gn in gallery_map:
            info = gallery_map[gn]
            floors.append(info["floor"])
            map_files.append(info["map_file"])
            x_pcts.append(info["x_pct"])
            y_pcts.append(info["y_pct"])
        else:
            floors.append("")
            map_files.append("")
            x_pcts.append(None)
            y_pcts.append(None)
            unmapped.append({"objectID": row["objectID"], "GalleryNumber": gn})

    df["floor"] = floors
    df["map_file"] = map_files
    df["x_pct"] = x_pcts
    df["y_pct"] = y_pcts

    # Save the enriched dataset
    df.to_parquet(DATA_DIR / "met_enriched.parquet", index=False)
    mapped_count = df[df["floor"] != ""].shape[0]
    print(f"✓ Enriched dataset saved: {mapped_count} mapped, {len(unmapped)} unmapped → data/met_enriched.parquet")

    # Report unmapped galleries
    if unmapped:
        unmapped_galleries = sorted(set(u["GalleryNumber"] for u in unmapped))
        print(f"  Unmapped gallery numbers: {unmapped_galleries}")

    # Update progress.md
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = PROGRESS_FILE.read_text()
    stage2_status = f"**Status**: ✅ Complete — {mapped_count} mapped, {len(unmapped)} unmapped ({ts})"
    content = content.replace(
        "## Stage 2: Gallery Coordinate Mapping (`gallery_mapper.py`)\n**Status**: Not started",
        f"## Stage 2: Gallery Coordinate Mapping (`gallery_mapper.py`)\n{stage2_status}",
    )
    PROGRESS_FILE.write_text(content)


if __name__ == "__main__":
    main()
