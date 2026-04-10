"""
Stage 1: Metadata Ingestion from The Met Collection API.

Fetches object metadata in mini-batches, cross-checks with isOnView search
results, produces a verified Parquet dataset and a mismatches CSV.

Usage:
    .venv/bin/python src/ingest_met.py [--batch-size 100] [--max-objects 0]
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
DATA_DIR = Path("data")
PROGRESS_FILE = Path("progress.md")

# Rate-limit: Met API aggressively rate-limits concurrent requests.
# Use fully sequential fetching with a slight delay safely under 80 RPS.
REQUEST_DELAY = 1.0  # Super slow Redemption batch
MAX_RETRIES = 10


def update_progress(batch_num: int, fetched: int, on_view: int, mismatches: int, errors: int):
    """Append a row to the Stage 1 table in progress.md."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = f"| {batch_num} | {fetched} | {on_view} | {mismatches} | {errors} | {ts} |"

    content = PROGRESS_FILE.read_text()
    # Find the Stage 1 table sentinel row and replace it
    marker = "| — | — | — | — | — | — |"
    if marker in content:
        content = content.replace(marker, row + "\n" + marker)
    else:
        # Table already has data; insert before the Stage 2 section
        stage2_marker = "## Stage 2:"
        content = content.replace(stage2_marker, row + "\n\n" + stage2_marker)
    content = content.replace("**Status**: Not started", "**Status**: In progress", 1)
    PROGRESS_FILE.write_text(content)


async def fetch_all_object_ids(session: aiohttp.ClientSession) -> list[int]:
    """Fetch the full list of object IDs from the Met API."""
    url = f"{BASE_URL}/objects"
    async with session.get(url) as resp:
        data = await resp.json()
    return data.get("objectIDs", [])


async def fetch_on_view_ids(session: aiohttp.ClientSession) -> set[int]:
    """Fetch object IDs that are currently on view via the search endpoint."""
    # The Met search API requires a query — use '*' as wildcard
    url = f"{BASE_URL}/search?isOnView=true&hasImages=true&q=*"
    async with session.get(url) as resp:
        data = await resp.json()
    ids = data.get("objectIDs", []) or []
    return set(ids)


async def fetch_description(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch and scrape the description from the object URL HTML."""
    if not url:
        return ""
    try:
        # Met tends to block requests without a User-Agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, 'lxml')
                divs = soup.find_all('div')
                for div in divs:
                    cls = div.get('class', [])
                    for c in cls:
                        # Match the specific class requested by user, robust to hash changes
                        if 'object-overview-module' in c and 'label' in c:
                            return div.get_text(separator=' ', strip=True)
            return ""
    except Exception:
        return ""


async def fetch_object_sequential(session: aiohttp.ClientSession, object_id: int) -> dict | None:
    """Fetch a single object's metadata sequentially with retry."""
    url = f"{BASE_URL}/objects/{object_id}"
    for attempt in range(MAX_RETRIES):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) MetArtNavigator/1.0'}
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    obj_data = await resp.json()
                    
                    # Also fetch description from the artwork webpage
                    if "objectURL" in obj_data and obj_data["objectURL"]:
                        # Scrape description
                        desc = await fetch_description(session, obj_data["objectURL"])
                        obj_data["description"] = desc
                    else:
                        obj_data["description"] = ""
                        
                    return obj_data
                elif resp.status == 429 or resp.status == 403:
                    wait = 2 ** (attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                return None
        except (asyncio.TimeoutError, aiohttp.ClientError):
            wait = 2 ** (attempt + 1)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait)
                continue
            return None
    return None


async def fetch_batch_sequential(session: aiohttp.ClientSession, object_ids: list[int]) -> list[dict]:
    """Fetch a batch of objects one at a time (no concurrency) but with lower delay."""
    results = []
    for oid in object_ids:
        obj = await fetch_object_sequential(session, oid)
        if obj is not None:
            results.append(obj)
        await asyncio.sleep(REQUEST_DELAY)
    return results


def extract_record(obj: dict) -> dict:
    """Extract the fields we care about from a raw API response."""
    tags = obj.get("tags") or []
    tag_terms = [t.get("term", "") for t in tags if isinstance(t, dict)]

    return {
        "objectID": obj.get("objectID"),
        "title": obj.get("title", ""),
        "artistDisplayName": obj.get("artistDisplayName", ""),
        "medium": obj.get("medium", ""),
        "department": obj.get("department", ""),
        "culture": obj.get("culture", ""),
        "period": obj.get("period", ""),
        "classification": obj.get("classification", ""),
        "GalleryNumber": obj.get("GalleryNumber", ""),
        "primaryImage": obj.get("primaryImage", ""),
        "primaryImageSmall": obj.get("primaryImageSmall", ""),
        "description": obj.get("description", ""),
        "tags": "|".join(tag_terms),
        "objectURL": obj.get("objectURL", ""),
        "isPublicDomain": obj.get("isPublicDomain", False),
    }


async def main(batch_size: int = 100, max_objects: int = 0):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Checkpoint: resume from previous partial downloads
    checkpoint_path = DATA_DIR / "ingest_checkpoint.json"
    processed_ids: set[int] = set()
    all_records: list[dict] = []

    if (DATA_DIR / "met_verified.parquet").exists() and not checkpoint_path.exists():
        print("[Resume] Rebuilding queue from met_verified.parquet because checkpoint was cleaned...")
        df_ver = pd.read_parquet(DATA_DIR / "met_verified.parquet")
        processed_ids = set(df_ver["objectID"].tolist())
        all_records = df_ver.to_dict("records")

    if checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text())
        processed_ids = set(ckpt.get("processed_ids", []))
        # Load existing records
        parquet_partial = DATA_DIR / "met_partial.parquet"
        if parquet_partial.exists():
            df_partial = pd.read_parquet(parquet_partial)
            all_records = df_partial.to_dict("records")
        print(f"[Resume] Found checkpoint with {len(processed_ids)} already processed objects.")

    async with aiohttp.ClientSession() as session:
        # Step 1: Get all object IDs
        print("[1/4] Fetching all object IDs...")
        all_ids = await fetch_all_object_ids(session)
        print(f"       Total objects in Met collection: {len(all_ids)}")

        # Step 2: Get on-view IDs via search
        print("[2/4] Fetching on-view object IDs via search endpoint...")
        on_view_ids = await fetch_on_view_ids(session)
        print(f"       On-view objects (search): {len(on_view_ids)}")

        # Filter to only objects that are on-view (for efficiency)
        # We'll fetch all on-view objects first since that's our target set
        target_ids = sorted(on_view_ids)

        # Remove already processed
        remaining_ids = [oid for oid in target_ids if oid not in processed_ids]
        
        if max_objects > 0:
            remaining_ids = remaining_ids[:max_objects]
        print(f"[3/4] Fetching metadata for {len(remaining_ids)} on-view objects...")

        semaphore = None  # Not used — sequential fetching
        total_errors = 0
        batch_num = len(processed_ids) // batch_size + 1

        for i in range(0, len(remaining_ids), batch_size):
            batch_ids = remaining_ids[i : i + batch_size]
            results = await fetch_batch_sequential(session, batch_ids)

            batch_errors = len(batch_ids) - len(results)
            total_errors += batch_errors

            for obj in results:
                rec = extract_record(obj)
                all_records.append(rec)
                processed_ids.add(rec["objectID"])

            # Save checkpoint
            checkpoint_path.write_text(json.dumps({
                "processed_ids": sorted(processed_ids),
                "timestamp": datetime.now().isoformat(),
            }))
            # Save partial parquet
            pd.DataFrame(all_records).to_parquet(DATA_DIR / "met_partial.parquet", index=False)

            update_progress(batch_num, len(results), 0, 0, batch_errors)
            batch_num += 1

            remaining_count = max(0, len(remaining_ids) - (i + len(batch_ids)))
            print(f"       Batch {batch_num-1}: fetched {len(results)}/{len(batch_ids)}, total: {len(all_records)}, remaining: {remaining_count}")
            await asyncio.sleep(REQUEST_DELAY)

    # Step 4: Cross-check & produce verified dataset vs mismatches
    print("[4/4] Cross-checking gallery numbers vs on-view status...")
    df = pd.DataFrame(all_records)

    # Verified: has a non-empty GalleryNumber AND is in the on-view search set
    df["in_on_view_search"] = df["objectID"].isin(on_view_ids)
    df["has_gallery"] = df["GalleryNumber"].astype(str).str.strip().ne("")

    # Verified dataset: both flags true
    verified_mask = df["in_on_view_search"] & df["has_gallery"]
    df_verified = df[verified_mask].drop(columns=["in_on_view_search", "has_gallery"])
    df_verified.to_parquet(DATA_DIR / "met_verified.parquet", index=False)
    print(f"       ✓ Verified on-view+gallery: {len(df_verified)} artworks → data/met_verified.parquet")

    # Mismatches: in one list but not the other
    mismatch_mask = df["in_on_view_search"] != df["has_gallery"]
    df_mismatch = df[mismatch_mask].copy()
    df_mismatch["reason"] = df_mismatch.apply(
        lambda r: "on_view_but_no_gallery" if r["in_on_view_search"] and not r["has_gallery"]
                  else "has_gallery_but_not_in_search",
        axis=1,
    )
    df_mismatch = df_mismatch[["objectID", "title", "GalleryNumber", "reason"]]
    df_mismatch.to_csv(DATA_DIR / "met_mismatches.csv", index=False)
    print(f"       ✗ Mismatches: {len(df_mismatch)} artworks → data/met_mismatches.csv")

    # Clean up temporary files
    for f in [DATA_DIR / "met_partial.parquet", checkpoint_path]:
        if f.exists():
            f.unlink()

    # Update progress with final status
    content = PROGRESS_FILE.read_text()
    content = content.replace("**Status**: In progress", f"**Status**: ✅ Complete — {len(df_verified)} verified, {len(df_mismatch)} mismatches", 1)
    PROGRESS_FILE.write_text(content)

    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Met Collection metadata")
    parser.add_argument("--batch-size", type=int, default=100, help="Objects per API batch")
    parser.add_argument("--max-objects", type=int, default=0, help="Max objects to fetch (0=all on-view)")
    args = parser.parse_args()
    asyncio.run(main(batch_size=args.batch_size, max_objects=args.max_objects))
