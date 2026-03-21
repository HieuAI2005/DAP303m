#!/usr/bin/env python3
"""
download_didemo_videos.py
=========================
Download DiDeMo videos from Flickr using the existing dl_links.
Supports resume (skip already-downloaded), parallel workers, and progress tracking.

Usage:
    python scripts/download_didemo_videos.py              # Download all splits
    python scripts/download_didemo_videos.py --limit 50   # Test with 50 videos
    python scripts/download_didemo_videos.py --splits train val
    python scripts/download_didemo_videos.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Config ────────────────────────────────────────────────────────────────────

DIDEMO_REPO = PROJECT_ROOT / "data" / "didemo_repo"
DIDEMO_VIDEOS = PROJECT_ROOT / "data" / "didemo_videos"
DIDEMO_VIDEOS.mkdir(parents=True, exist_ok=True)

SPLIT_FILES = {
    "train": DIDEMO_REPO / "data" / "train_data.json",
    "val":   DIDEMO_REPO / "data" / "val_data.json",
    "test":  DIDEMO_REPO / "data" / "test_data.json",
}

DEFAULT_WORKERS = 8
MAX_RETRIES = 3
RETRY_DELAY = 5   # seconds
REQUEST_TIMEOUT = 60  # seconds


# ── Stats (thread-safe) ────────────────────────────────────────────────────────

stats_lock = Lock()
stats = {"total": 0, "done": 0, "skipped": 0, "failed": 0, "errors": []}


def update_stats(**kwargs):
    with stats_lock:
        for k, v in kwargs.items():
            stats[k] = stats.get(k, 0) + v


# ── Video ID extraction ───────────────────────────────────────────────────────

def extract_flickr_id(url: str) -> str | None:
    """Extract Flickr photo/video ID from dl_link."""
    import re
    m = re.search(r'[?&]id=(\d+)', url)
    return m.group(1) if m else None


# ── Downloader ────────────────────────────────────────────────────────────────

def download_single(args_tuple) -> dict:
    """Download one video. Returns dict with status."""
    entry, split, idx, total = args_tuple
    video_file = entry.get("video", "")
    dl_link = entry.get("dl_link", "")
    flickr_id = extract_flickr_id(dl_link)

    if not flickr_id:
        update_stats(failed=1)
        return {"status": "error", "reason": "no_flickr_id", "video": video_file}

    out_name = f"{flickr_id}.mp4"
    out_path = DIDEMO_VIDEOS / out_name

    # Resume: skip if already downloaded and non-empty
    if out_path.exists() and out_path.stat().st_size > 10_000:
        update_stats(skipped=1)
        return {"status": "skipped", "path": str(out_path), "flickr_id": flickr_id}

    # Build output filename from original video name (for reference)
    orig_ext = Path(video_file).suffix.lstrip(".") or "mp4"
    out_path = DIDEMO_VIDEOS / f"{flickr_id}.mp4"
    if out_path.exists() and out_path.stat().st_size > 10_000:
        update_stats(skipped=1)
        return {"status": "skipped", "path": str(out_path), "flickr_id": flickr_id}

    # Download from Flickr
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.flickr.com/",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(dl_link, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = resp.read()

            if len(data) < 10_000:
                update_stats(failed=1)
                return {
                    "status": "error", "reason": "file_too_small",
                    "size": len(data), "flickr_id": flickr_id
                }

            with open(out_path, "wb") as f:
                f.write(data)

            update_stats(done=1)
            return {"status": "done", "path": str(out_path), "size": len(data), "flickr_id": flickr_id}

        except Exception as e:
            if attempt == MAX_RETRIES:
                update_stats(failed=1)
                err_msg = str(e)[:100]
                print(f"    ❌ [{idx+1}/{total}] {flickr_id}: {err_msg}")
                return {"status": "error", "reason": str(e), "flickr_id": flickr_id}
            time.sleep(RETRY_DELAY * attempt)

    update_stats(failed=1)
    return {"status": "error", "reason": "max_retries", "flickr_id": flickr_id}


# ── Progress reporter ──────────────────────────────────────────────────────────

def report_progress(futures, total):
    """Print progress bar as futures complete."""
    done = 0
    last_pct = -1
    while done < total:
        completed = sum(1 for f in futures if f.done())
        pct = int(100 * completed / total)
        if pct != last_pct and pct % 5 == 0:
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}% ({completed}/{total})", end="", flush=True)
            last_pct = pct
        import time; time.sleep(0.5)
        done = sum(1 for f in futures if f.done())
    bar = "█" * 20
    print(f"\r  [{bar}] 100% ({total}/{total})", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download DiDeMo videos from Flickr")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"],
                        help="Which splits to download (default: all)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit total downloads per split (0=all, for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if file exists")
    args = parser.parse_args()

    # Collect all entries
    all_entries = []
    for split in args.splits:
        split_path = SPLIT_FILES[split]
        if not split_path.exists():
            print(f"⚠️  Split file not found: {split_path}")
            continue
        with open(split_path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            entry["_split"] = split
        all_entries.extend(data)

    if not all_entries:
        print("❌ No entries found. Run download_process first.")
        sys.exit(1)

    total = len(all_entries)
    print(f"\n{'='*60}")
    print(f"  📥 DiDeMo Video Downloader")
    print(f"{'='*60}")
    print(f"  Total entries: {total:,}")
    print(f"  Workers:       {args.workers}")
    print(f"  Output:        {DIDEMO_VIDEOS}")
    print(f"  Limit:         {args.limit if args.limit else 'all'}")
    print(f"{'='*60}\n")

    # Check already downloaded
    already_done = 0
    for entry in all_entries:
        flickr_id = extract_flickr_id(entry.get("dl_link", ""))
        if flickr_id:
            p = DIDEMO_VIDEOS / f"{flickr_id}.mp4"
            if p.exists() and p.stat().st_size > 10_000:
                already_done += 1

    print(f"  Already downloaded: {already_done:,}/{total:,}")
    print(f"  Remaining:          {total - already_done:,}\n")

    if already_done == total:
        print("✅ All videos already downloaded!")
        return

    # Download
    args_limited = all_entries[:args.limit] if args.limit else all_entries

    print(f"  Starting download ({len(args_limited)} entries)...\n")

    # Prepare tasks
    tasks = [
        (entry, entry.get("_split", "unknown"), i, len(args_limited))
        for i, entry in enumerate(args_limited)
    ]

    # Filter out already-downloaded unless --force
    if not args.force:
        def should_download(t):
            entry = t[0]
            fid = extract_flickr_id(entry.get("dl_link", ""))
            if not fid:
                return True
            p = DIDEMO_VIDEOS / f"{fid}.mp4"
            return not (p.exists() and p.stat().st_size > 10_000)
        tasks = [t for t in tasks if should_download(t)]
        if not tasks:
            print("✅ All videos already downloaded!")
            return
        print(f"  Filtering to {len(tasks)} not-yet-downloaded entries\n")

    stats["total"] = len(tasks)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single, t): t for t in tasks}
        report_progress(futures, len(tasks))

        for future in as_completed(futures):
            result = future.result()
            # Silence successful/skipped (already counted in progress)
            if result["status"] == "error":
                with stats_lock:
                    stats["errors"].append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  ✅ Done:    {stats['done']:,}")
    print(f"  ⏭️  Skipped: {stats['skipped']:,}")
    print(f"  ❌ Failed:  {stats['failed']:,}")
    if stats["errors"]:
        err_file = DIDEMO_VIDEOS / "download_errors.json"
        with open(err_file, "w", encoding="utf-8") as f:
            json.dump(stats["errors"], f, indent=2, ensure_ascii=False)
        print(f"  📄 Errors:  {err_file}")

    total_on_disk = len([f for f in DIDEMO_VIDEOS.glob("*.mp4") if f.stat().st_size > 10_000])
    print(f"\n  📁 Total videos on disk: {total_on_disk:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
