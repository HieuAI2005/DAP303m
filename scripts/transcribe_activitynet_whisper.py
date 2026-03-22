#!/usr/bin/env python3
"""
transcribe_activitynet_whisper.py
==================================
Transcribe ActivityNet videos using Groq Whisper API.

Workflow:
  1. Find downloaded .mp4 files in data/ActivityNet_Videos/
  2. Extract audio (ffmpeg 16kHz mono WAV)
  3. Send to Groq Whisper API
  4. Save transcript JSON per video
  5. Update ActivityNet chunks with real transcript text

Usage:
    python scripts/transcribe_activitynet_whisper.py           # Dry run (test 3)
    python scripts/transcribe_activitynet_whisper.py --execute  # Process all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("WhisperTranscriber")


# ── Paths ────────────────────────────────────────────────────────────────────

VIDEO_DIR = PROJECT_ROOT / "data" / "ActivityNet_Videos"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "ActivityNet_Videos" / "transcripts"
TEMP_AUDIO_DIR = PROJECT_ROOT / "data" / "ActivityNet_Videos" / "temp_audio"
PROGRESS_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "activitynet_transcribed.txt"


# ── Audio extraction ─────────────────────────────────────────────────────────

def extract_audio(video_path: Path, audio_out: Path) -> bool:
    """Extract 16kHz mono WAV from video using ffmpeg."""
    if audio_out.exists():
        return True
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                str(audio_out),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return audio_out.exists()
    except Exception:
        return False


# ── Groq Whisper API ─────────────────────────────────────────────────────────

class DailyLimitExceeded(Exception):
    """Raised when Groq daily ASPD quota is exhausted for this key."""


def transcribe_audio(audio_path: Path, client=None) -> dict | None:
    """Transcribe audio using Groq Whisper API."""
    if client is None:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        import groq
        client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

    try:
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = []
        for seg in getattr(response, "segments", []):
            segments.append({
                "start": round(float(getattr(seg, "start", 0)), 2),
                "end": round(float(getattr(seg, "end", 0)), 2),
                "text": getattr(seg, "text", "").strip(),
            })

        return {
            "text": getattr(response, "text", "").strip(),
            "language": getattr(response, "language", "en"),
            "segments": segments,
        }
    except Exception as e:
        err_str = str(e)
        logger.warning(f"  Whisper API error: {e}")
        # Detect daily ASPD quota exhaustion — raise so caller can rotate key
        if "ASPD" in err_str or ("seconds of audio per day" in err_str and "Limit" in err_str):
            raise DailyLimitExceeded(err_str) from e
        return None


# ── Process videos ────────────────────────────────────────────────────────────

def get_rate_limited_client():
    """Create Groq client with rotation across available keys."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    import groq
    keys = [
        os.environ.get("GROQ_API_KEY", ""),
        os.environ.get("GROQ_API_KEY_1", ""),
        os.environ.get("GROQ_API_KEY_2", ""),
    ]
    keys = [k for k in keys if k]
    return groq.Groq(api_key=keys[0] if keys else ""), keys


def process_video(video_path: Path, rate_limiter, client) -> dict | None:
    """Extract audio → transcribe → return transcript.
    Raises DailyLimitExceeded if the API key's daily quota is exhausted."""
    video_id = video_path.stem
    audio_out = TEMP_AUDIO_DIR / f"{video_id}.wav"

    # Extract audio
    if not extract_audio(video_path, audio_out):
        return None

    # Rate limit
    rate_limiter.wait()

    # Transcribe — pass client so key rotation works
    # DailyLimitExceeded propagates to caller for key rotation
    return transcribe_audio(audio_out, client=client)


class RateLimiter:
    def __init__(self, max_per_min: int = 30):
        self.max_per_min = max_per_min
        self.interval = 60.0 / max_per_min
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


def transcribe_all(dry_run: bool = True, limit: int = 0):
    """Transcribe all downloaded ActivityNet videos."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Find downloaded videos
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    logger.info(f"Found {len(videos)} downloaded videos")

    # Load progress
    done_ids = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            done_ids = set(line.strip() for line in f if line.strip())

    remaining = [v for v in videos if v.stem not in done_ids]
    logger.info(f"Already transcribed: {len(done_ids)}")
    logger.info(f"Remaining: {len(remaining)}")

    if dry_run:
        remaining = remaining[:limit or 3]
        logger.info(f"DRY RUN: testing {len(remaining)} videos")

    if not remaining:
        logger.info("All videos already transcribed!")
        return

    rate_limiter = RateLimiter(max_per_min=30)
    import groq
    keys = [
        os.environ.get("GROQ_API_KEY", ""),
        os.environ.get("GROQ_API_KEY_1", ""),
        os.environ.get("GROQ_API_KEY_2", ""),
    ]
    keys = [k for k in keys if k]
    key_idx = 0
    exhausted_keys = set()
    client = groq.Groq(api_key=keys[key_idx])
    logger.info(f"Using key {key_idx+1}/{len(keys)}")

    success = 0
    failed = 0
    checkpoint = 20

    for i, video_path in enumerate(remaining):
        video_id = video_path.stem
        transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"

        try:
            result = process_video(video_path, rate_limiter, client)
        except DailyLimitExceeded:
            # This key's daily quota is exhausted — rotate to next key
            exhausted_keys.add(key_idx)
            next_keys = [j for j in range(len(keys)) if j not in exhausted_keys]
            if next_keys:
                key_idx = next_keys[0]
                client = groq.Groq(api_key=keys[key_idx])
                logger.info(f"  🔄 Daily limit hit — rotating to key {key_idx+1}/{len(keys)}")
                # Retry the same video with new key
                try:
                    result = process_video(video_path, rate_limiter, client)
                except DailyLimitExceeded:
                    exhausted_keys.add(key_idx)
                    next_keys2 = [j for j in range(len(keys)) if j not in exhausted_keys]
                    if next_keys2:
                        key_idx = next_keys2[0]
                        client = groq.Groq(api_key=keys[key_idx])
                        logger.info(f"  🔄 Daily limit hit — rotating to key {key_idx+1}/{len(keys)}")
                        try:
                            result = process_video(video_path, rate_limiter, client)
                        except DailyLimitExceeded:
                            logger.warning("  ⚠️ All keys exhausted for today. Stopping.")
                            break
                    else:
                        logger.warning("  ⚠️ All keys exhausted for today. Stopping.")
                        break
            else:
                logger.warning("  ⚠️ All keys exhausted for today. Stopping.")
                break

        if result:
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump({"video_id": video_id, **result}, f, indent=2)
            success += 1
            done_ids.add(video_id)
        else:
            failed += 1

        if (i + 1) % checkpoint == 0:
            pct = 100 * (i + 1) / len(remaining)
            remaining_est = (len(remaining) - i - 1) / max((i + 1) / ((time.time() - start_time) / 60), 0.01)
            logger.info(f"  Progress: {i+1}/{len(remaining)} ({pct:.1f}%) "
                        f"~{int(remaining_est)}min left | ✅{success} ❌{failed}")
            with open(PROGRESS_FILE, "w") as f:
                f.write("\n".join(sorted(done_ids)))

    # Final save
    with open(PROGRESS_FILE, "w") as f:
        f.write("\n".join(sorted(done_ids)))

    # Cleanup temp audio
    for f in TEMP_AUDIO_DIR.glob("*.wav"):
        f.unlink()

    logger.info(f"✅ Done! {success} transcribed, {failed} failed")


def update_chunks_with_transcripts():
    """Merge Whisper transcripts into ActivityNet chunks."""
    chunks_path = PROJECT_ROOT / "data" / "pipeline_output" / "activitynet_chunks" / "all_chunks.json"
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks", data if isinstance(data, list) else [])

    updated = 0
    for chunk in chunks:
        vid = chunk.get("video_id", "")
        transcript_path = TRANSCRIPT_DIR / f"{vid}.json"
        if transcript_path.exists():
            with open(transcript_path, encoding="utf-8") as f:
                t = json.load(f)

            start = chunk.get("start_seconds", 0)
            end = chunk.get("end_seconds", 0)

            # Extract text within this time range
            segs = t.get("segments", [])
            relevant = [s["text"] for s in segs
                        if s["start"] < end and s["end"] > start]
            if relevant:
                chunk["text"] = " ".join(relevant)
                chunk["dialogue_text"] = " ".join(relevant)
                chunk["audio_language"] = t.get("language", "en")
                updated += 1

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Updated {updated}/{len(chunks)} chunks with transcripts")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transcribe ActivityNet videos with Groq Whisper")
    parser.add_argument("--execute", action="store_true", help="Actually transcribe (dry run by default)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number (0=all)")
    parser.add_argument("--update-chunks", action="store_true", help="Update chunks after transcription")
    args = parser.parse_args()

    global start_time
    start_time = time.time()

    if args.execute:
        transcribe_all(dry_run=False, limit=args.limit)
        if args.update_chunks:
            update_chunks_with_transcripts()
    else:
        transcribe_all(dry_run=True, limit=args.limit)


if __name__ == "__main__":
    main()
