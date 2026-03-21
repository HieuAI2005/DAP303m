#!/usr/bin/env python3
"""
download_activitynet_videos.py
==============================
Download ActivityNet videos using yt-dlp, covering all annotated segments.

Strategy:
  1. Parse activity_net.v1-3.json to get per-video annotation time ranges
  2. For each video, download from min(start) to max(end) across all its segments
  3. Skip videos already downloaded
  4. Parallel workers with progress checkpoint every 100 videos

Key fix: old script used --download-sections "*0-30" which missed 41.7% of
annotated segments that start after 30s. New approach downloads exactly the
range needed to cover all annotations per video.

Usage:
    python scripts/download_activitynet_videos.py              # Check status
    python scripts/download_activitynet_videos.py --execute   # Download all (parallel)
    python scripts/download_activitynet_videos.py --execute --limit 500  # Test 500
    python scripts/download_activitynet_videos.py --execute --full       # Full videos
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ActivityNetDownloader")


# ── Paths ────────────────────────────────────────────────────────────────────

ANNOTATIONS_JSON = PROJECT_ROOT / "data" / "ActivityNet_Captions" / "activity_net.v1-3.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ActivityNet_Videos"
PROGRESS_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "activitynet_downloaded.txt"


# ── Extract YouTube IDs with annotation ranges ────────────────────────────────

def get_videos_with_ranges() -> list[dict]:
    """
    Parse annotation JSON and return per-video download info.

    Returns list of dicts:
      {
        "yt_id": str,        # 11-char YouTube ID
        "t_start": float,    # earliest annotation start (seconds)
        "t_end":   float,    # latest annotation end (seconds)
        "has_captions": bool
      }

    Download range covers all annotated segments: *t_start-t_end
    """
    with open(ANNOTATIONS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("database", {})
    result = []

    for vid, info in videos.items():
        if not (vid and len(vid) == 11 and all(c.isalnum() or c in "-_" for c in vid)):
            continue

        annotations = info.get("annotations", [])
        has_captions = bool(annotations)

        if annotations:
            # Compute span that covers all annotated segments
            all_starts = [seg["segment"][0] for seg in annotations if "segment" in seg]
            all_ends   = [seg["segment"][1] for seg in annotations if "segment" in seg]
            t_start = max(0.0, min(all_starts) - 2.0)  # 2s padding before first segment
            t_end   = max(all_ends) + 2.0               # 2s padding after last segment
        else:
            # No annotations: download first 60s as fallback
            t_start = 0.0
            t_end   = 60.0

        result.append({
            "yt_id": vid,
            "t_start": round(t_start, 1),
            "t_end":   round(t_end, 1),
            "has_captions": has_captions,
        })

    # Sort: annotated videos first
    result.sort(key=lambda x: (not x["has_captions"], x["yt_id"]))
    return result


# ── Download ─────────────────────────────────────────────────────────────────

def download_video(entry: dict, output_dir: Path, full_video: bool = False) -> tuple[str, bool]:
    """
    Download one video using annotation time range.
    If full_video=True, download complete video.
    Returns (yt_id, success).
    """
    yt_id = entry["yt_id"]
    out_path = output_dir / f"{yt_id}.mp4"

    if out_path.exists() and out_path.stat().st_size > 5000:
        return yt_id, True

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "best[height<=720]/best",
        "-o", str(out_path.resolve()),
        "--no-check-certificates",
        "--user-agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ]

    if not full_video:
        t_start = entry["t_start"]
        t_end   = entry["t_end"]
        cmd += ["--download-sections", f"*{t_start}-{t_end}"]

    cmd.append(f"https://www.youtube.com/watch?v={yt_id}")

    # Timeout: annotated segments can be up to ~3 min; full videos need more time
    timeout = 300 if full_video else 120

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        for _ in range(10):
            if out_path.exists() and out_path.stat().st_size > 5000:
                return yt_id, True
            part = output_dir / f"{out_path.stem}.part"
            if not part.exists():
                break
            time.sleep(0.5)
        return yt_id, out_path.exists() and out_path.stat().st_size > 5000
    except subprocess.TimeoutExpired:
        if out_path.exists():
            out_path.unlink()
        return yt_id, False
    except Exception:
        return yt_id, False


def download_all(workers: int = 8, limit: int = 0, full_video: bool = False):
    """Download all ActivityNet videos using parallel workers."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = get_videos_with_ranges()
    logger.info(f"Found {len(entries)} ActivityNet YouTube IDs")

    annotated = sum(1 for e in entries if e["has_captions"])
    avg_span  = sum(e["t_end"] - e["t_start"] for e in entries if e["has_captions"]) / max(annotated, 1)
    logger.info(f"  With annotations: {annotated} | Avg span to download: {avg_span:.1f}s")

    already = {f.stem for f in OUTPUT_DIR.glob("*.mp4") if f.stat().st_size > 5000}
    remaining = [e for e in entries if e["yt_id"] not in already]
    logger.info(f"Already downloaded: {len(already)}")
    logger.info(f"Remaining to download: {len(remaining)}")

    if limit:
        remaining = remaining[:limit]

    if not remaining:
        logger.info("All videos already downloaded!")
        return

    mode = "full video" if full_video else "annotation range"
    logger.info(f"Starting download ({workers} workers, {mode} mode)...")

    start_time = time.time()
    done = set(already)
    success_count = 0
    fail_count = 0
    total = len(remaining)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_video, entry, OUTPUT_DIR, full_video): entry["yt_id"]
            for entry in remaining
        }

        for i, future in enumerate(as_completed(futures)):
            yt_id, ok = future.result()
            if ok:
                success_count += 1
                done.add(yt_id)
            else:
                fail_count += 1

            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                eta = (total - i - 1) / max(rate, 1) / 60
                logger.info(
                    f"  Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%) "
                    f"| OK:{success_count} FAIL:{fail_count} "
                    f"| {rate:.1f}/min | ETA: {eta:.0f}min"
                )
                with open(PROGRESS_FILE, "w") as f:
                    f.write("\n".join(sorted(done)))

    with open(PROGRESS_FILE, "w") as f:
        f.write("\n".join(sorted(done)))

    total_done = sum(1 for e in remaining if (OUTPUT_DIR / f"{e['yt_id']}.mp4").exists())
    logger.info(f"Done! {total_done}/{total} downloaded this run | Total: {len(done)}")


def check_status():
    """Show download status and annotation coverage analysis."""
    entries = get_videos_with_ranges()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = {f.stem for f in OUTPUT_DIR.glob("*.mp4") if f.stat().st_size > 5000}

    annotated = [e for e in entries if e["has_captions"]]
    total_span_old = len(annotated) * 30  # old script: 30s per video
    total_span_new = sum(e["t_end"] - e["t_start"] for e in annotated)

    logger.info("ActivityNet videos:")
    logger.info(f"  Total YouTube IDs:           {len(entries)}")
    logger.info(f"  Videos with annotations:     {len(annotated)}")
    logger.info(f"  Downloaded:                  {len(downloaded)}")
    logger.info(f"  Remaining:                   {len(entries) - len(downloaded)}")
    logger.info(f"  Storage used:                {sum(f.stat().st_size for f in OUTPUT_DIR.glob('*.mp4')) / 1e9:.2f} GB")
    logger.info("")
    logger.info("Coverage analysis (annotated videos only):")
    logger.info(f"  Old script (30s clips):      {total_span_old/3600:.1f}h total footage")
    logger.info(f"  New script (annotation span): {total_span_new/3600:.1f}h total footage")
    # Count segments that would have been missed by 30s clip
    missed = sum(
        1 for e in annotated
        if e["t_start"] > 30
    )
    logger.info(f"  Segments missed by old 30s:  {missed}/{len(annotated)} ({100*missed/max(len(annotated),1):.1f}%)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download ActivityNet videos")
    parser.add_argument("--execute", action="store_true", help="Actually download (dry run by default)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of downloads (0=all)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--full", action="store_true", help="Download full videos instead of annotation range")
    args = parser.parse_args()

    if args.execute:
        download_all(workers=args.workers, limit=args.limit, full_video=args.full)
    else:
        check_status()


if __name__ == "__main__":
    main()
