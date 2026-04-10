"""
Stage 3: Image Downloader.

Downloads primaryImageSmall URLs in mini-batches with concurrent threads.
Produces an image manifest Parquet.

Usage:
    .venv/bin/python src/download_images.py [--batch-size 200] [--workers 8]
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
PROGRESS_FILE = Path("progress.md")


def download_image(object_id: int, url: str, dest: Path) -> dict:
    """Download a single image. Returns a manifest record."""
    if dest.exists():
        return {"objectID": object_id, "filepath": str(dest), "success": True}
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return {"objectID": object_id, "filepath": str(dest), "success": True}
    except Exception as e:
        return {"objectID": object_id, "filepath": str(dest), "success": False}


def update_progress(batch_num: int, attempted: int, succeeded: int, failed: int):
    """Append a row to the Stage 3 table in progress.md."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = f"| {batch_num} | {attempted} | {succeeded} | {failed} | {ts} |"

    content = PROGRESS_FILE.read_text()
    marker = "| — | — | — | — | — |"
    if marker in content:
        content = content.replace(marker, row + "\n" + marker, 1)
    PROGRESS_FILE.write_text(content)


def main(batch_size: int = 200, workers: int = 8):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load the enriched metadata
    parquet_path = DATA_DIR / "met_enriched.parquet"
    if not parquet_path.exists():
        # Fallback to verified
        parquet_path = DATA_DIR / "met_verified.parquet"
    if not parquet_path.exists():
        print("Error: No metadata Parquet found. Run ingest_met.py and gallery_mapper.py first.")
        return

    df = pd.read_parquet(parquet_path)
    # Filter to rows with a valid image URL
    df = df[df["primaryImageSmall"].astype(str).str.startswith("http")]
    print(f"Will download images for {len(df)} artworks.")

    manifest_records = []
    total_success = 0
    total_fail = 0

    # Load existing manifest for resume
    manifest_path = DATA_DIR / "images_manifest.parquet"
    already_downloaded = set()
    if manifest_path.exists():
        df_manifest = pd.read_parquet(manifest_path)
        already_downloaded = set(df_manifest[df_manifest["success"]]["objectID"].tolist())
        manifest_records = df_manifest.to_dict("records")
        print(f"[Resume] {len(already_downloaded)} images already downloaded.")

    remaining = df[~df["objectID"].isin(already_downloaded)]
    print(f"Remaining to download: {len(remaining)}")

    batch_num = 1
    for i in range(0, len(remaining), batch_size):
        batch = remaining.iloc[i : i + batch_size]
        batch_results = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for _, row in batch.iterrows():
                oid = row["objectID"]
                url = row["primaryImageSmall"]
                dest = IMAGES_DIR / f"{oid}.jpg"
                futures[pool.submit(download_image, oid, url, dest)] = oid

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_num}"):
                result = future.result()
                batch_results.append(result)

        succeeded = sum(1 for r in batch_results if r["success"])
        failed = len(batch_results) - succeeded
        total_success += succeeded
        total_fail += failed
        manifest_records.extend(batch_results)

        # Save manifest checkpoint
        pd.DataFrame(manifest_records).to_parquet(manifest_path, index=False)

        update_progress(batch_num, len(batch_results), succeeded, failed)
        remaining_count = max(0, len(remaining) - (i + len(batch)))
        print(f"  Batch {batch_num}: {succeeded}/{len(batch_results)} succeeded, remaining: {remaining_count}")
        batch_num += 1

    # Final status
    content = PROGRESS_FILE.read_text()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = content.replace(
        "## Stage 3: Image Downloading (`download_images.py`)\n**Status**: Not started",
        f"## Stage 3: Image Downloading (`download_images.py`)\n**Status**: ✅ Complete — {total_success + len(already_downloaded)} images ({ts})",
    )
    PROGRESS_FILE.write_text(content)
    print(f"\n✓ Download complete: {total_success + len(already_downloaded)} total images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Met artwork images")
    parser.add_argument("--batch-size", type=int, default=200, help="Images per batch")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download threads")
    args = parser.parse_args()
    main(batch_size=args.batch_size, workers=args.workers)
