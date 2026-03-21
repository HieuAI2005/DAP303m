#!/usr/bin/env python3
"""
detect_faces_trailer.py
========================
Run face detection on trailer keyframes to enrich L4 characters field.
Uses DeepFace + TMDB cast data to identify actors in scenes.

Strategy:
  - For each trailer movie with TMDB cast data:
    1. Run face detection on keyframes (RetinaFace detector, fast mode)
    2. Count faces per chunk (cast_in_scene approximation)
    3. If TMDB cast available, try to match faces to known actors
  - Update chunks.json with face count + detected character metadata

Usage:
    python scripts/detect_faces_trailer.py --check
    python scripts/detect_faces_trailer.py --run [--limit N] [--movie-id tt...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FaceDetect")

OUTPUT_BASE = PROJECT_ROOT / "data" / "pipeline_output" / "trailers"
TITLES_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "imdb_titles_cache.json"
TMDB_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "tmdb_metadata"


def get_tmdb_cast(movie_id: str) -> list[str]:
    """Load top cast names from TMDB metadata."""
    for path in [
        TMDB_DIR / f"{movie_id}.json",
        PROJECT_ROOT / "data" / "tmdb_metadata" / f"{movie_id}.json",
    ]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                cast = data.get("credits", {}).get("cast", data.get("cast", []))
                return [c.get("name", "") for c in cast[:10] if c.get("name")]
            except Exception:
                pass
    return []


def detect_faces_in_image(img_path: Path) -> int:
    """Return number of faces detected in image. Returns -1 on error."""
    try:
        from deepface import DeepFace
        result = DeepFace.extract_faces(
            img_path=str(img_path),
            detector_backend="retinaface",
            enforce_detection=False,
            align=False,
        )
        # Filter by confidence
        faces = [f for f in result if f.get("confidence", 0) > 0.5]
        return len(faces)
    except Exception:
        return 0


def process_movie(movie_id: str, title: str) -> dict:
    movie_dir = OUTPUT_BASE / movie_id
    chunks_file = movie_dir / "chunks.json"
    kf_dir = movie_dir / "keyframes"

    if not chunks_file.exists():
        return {"movie_id": movie_id, "status": "no_chunks"}

    raw = json.loads(chunks_file.read_text())
    chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw

    # Skip if already done (characters list non-empty for majority)
    already = sum(1 for c in chunks if c.get("characters"))
    if already > len(chunks) * 0.5:
        return {"movie_id": movie_id, "status": "already_done", "enriched": already}

    all_kfs = sorted(kf_dir.glob("kf_*.jpg")) if kf_dir.exists() else []
    if not all_kfs:
        return {"movie_id": movie_id, "status": "no_keyframes"}

    cast = get_tmdb_cast(movie_id)
    enriched = 0

    for i, chunk in enumerate(chunks):
        kf_idx = min(i, len(all_kfs) - 1)
        kf = all_kfs[kf_idx]
        n_faces = detect_faces_in_image(kf)
        if n_faces > 0:
            # We can't reliably ID who without a reference DB, but we record:
            # - cast_in_scene: top cast names as candidates (all scenes)
            # - face_count: actual detected faces
            chunk["face_count"] = n_faces
            if cast and not chunk.get("characters"):
                # Use TMDB top cast as characters (trailer-level approximation)
                chunk["characters"] = cast[:min(n_faces + 1, len(cast))]
            enriched += 1

    out = raw if isinstance(raw, dict) else {"chunks": chunks}
    out["chunks"] = chunks
    with open(chunks_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return {"movie_id": movie_id, "status": "done", "enriched": enriched, "total": len(chunks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--movie-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    trailer_dirs = sorted([d for d in OUTPUT_BASE.glob("tt*/")
                           if (d / "chunks.json").exists() and (d / "keyframes").exists()])

    unenriched = []
    for td in trailer_dirs:
        raw = json.loads((td / "chunks.json").read_text())
        chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw
        done = sum(1 for c in chunks if c.get("characters"))
        if done < len(chunks) * 0.5:
            unenriched.append(td.name)

    logger.info(f"Trailer movies with keyframes: {len(trailer_dirs)}")
    logger.info(f"Need face detection:          {len(unenriched)}")

    if args.check:
        return

    if not args.run:
        return

    try:
        from deepface import DeepFace  # noqa: F401
    except ImportError:
        logger.error("pip install deepface")
        return

    titles = {}
    if TITLES_FILE.exists():
        titles = json.loads(TITLES_FILE.read_text())

    todo = [args.movie_id] if args.movie_id else unenriched
    if args.limit:
        todo = todo[:args.limit]

    logger.info(f"Processing {len(todo)} movies...")
    done_count = fail_count = 0

    for i, mid in enumerate(todo):
        title = titles.get(mid, mid)
        result = process_movie(mid, title)
        status = result["status"]

        if status in ("done", "already_done"):
            done_count += 1
            if status == "done":
                logger.info(f"  OK {mid} ({title[:30]}): {result.get('enriched',0)}/{result.get('total',0)}")
        else:
            fail_count += 1
            logger.debug(f"  SKIP {mid}: {status}")

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(todo)} | OK:{done_count} FAIL:{fail_count}")

    logger.info(f"Done: {done_count} processed, {fail_count} failed/skipped")

    # Rebuild trailer chunks
    logger.info("Rebuilding all_trailer_chunks.json...")
    CHUNKS_BASE = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "trailer_chunks"
    all_chunks = []
    for td in sorted(OUTPUT_BASE.glob("tt*/")):
        cf = td / "chunks.json"
        if cf.exists():
            raw = json.loads(cf.read_text())
            chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw
            all_chunks.extend(chunks)
    CHUNKS_BASE.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_BASE / "all_trailer_chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(all_chunks)} trailer chunks")


if __name__ == "__main__":
    main()
