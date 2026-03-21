#!/usr/bin/env python3
"""
process_trailers_clip.py
========================
Run CLIP pipeline on downloaded trailer videos:
  1. Extract keyframes (1 per 3 seconds via ffmpeg)
  2. Encode with CLIP ViT-B/32
  3. Build per-movie FAISS visual index
  4. Merge into global visual_index.faiss
  5. Create 5-layer chunks for knowledge index

Designed to run incrementally — skips already-processed trailers.

Usage:
    python scripts/process_trailers_clip.py --check     # Show status
    python scripts/process_trailers_clip.py --run       # Process all new trailers
    python scripts/process_trailers_clip.py --run --workers 4
    python scripts/process_trailers_clip.py --rebuild-index  # Merge all into global index
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
logger = logging.getLogger("TrailerCLIP")

TRAILERS_DIR = PROJECT_ROOT / "data" / "trailers"
OUTPUT_BASE = PROJECT_ROOT / "data" / "pipeline_output" / "trailers"
INDEX_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "indexes"
CHUNKS_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "trailer_chunks"
TITLES_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "imdb_titles_cache.json"
KF_INTERVAL_SECS = 3  # Extract 1 keyframe every 3 seconds


def get_titles() -> dict:
    if TITLES_FILE.exists():
        return json.loads(TITLES_FILE.read_text())
    return {}


def extract_keyframes(video_path: Path, out_dir: Path, interval: int = KF_INTERVAL_SECS) -> list[Path]:
    """Extract keyframes using ffmpeg at fixed interval."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "kf_%05d.jpg"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{interval},scale=224:224:force_original_aspect_ratio=decrease",
        "-q:v", "2",
        str(pattern),
        "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:
        logger.warning(f"ffmpeg error for {video_path.name}: {e}")
    return sorted(out_dir.glob("kf_*.jpg"))


def encode_keyframes_clip(kf_paths: list[Path]) -> "np.ndarray":
    """Encode keyframes with CLIP, return L2-normalized embeddings."""
    import numpy as np
    import torch

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
    except Exception:
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device="cpu")
        except Exception as e:
            logger.error(f"CLIP not available: {e}")
            return np.zeros((len(kf_paths), 512), dtype=np.float32)

    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embeddings = []
    batch_size = 32
    for i in range(0, len(kf_paths), batch_size):
        batch = kf_paths[i:i + batch_size]
        imgs = []
        for p in batch:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                imgs.append(img)
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))

        imgs_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            try:
                feats = model.encode_image(imgs_tensor)
            except Exception:
                feats = torch.zeros(len(imgs), 512)
        feats = feats.cpu().float().numpy()
        # L2-normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        feats = feats / norms
        embeddings.append(feats)

    return np.vstack(embeddings) if embeddings else np.zeros((len(kf_paths), 512), dtype=np.float32)


def build_chunks(imdb_id: str, title: str, kf_paths: list[Path], interval: int) -> list[dict]:
    """Build 5-layer chunks from keyframes (one chunk per shot ~30s)."""
    chunks = []
    shots_per_chunk = max(1, 30 // interval)  # Group frames into ~30s chunks

    for i in range(0, len(kf_paths), shots_per_chunk):
        group = kf_paths[i:i + shots_per_chunk]
        start_s = i * interval
        end_s = min((i + shots_per_chunk) * interval, len(kf_paths) * interval)
        chunk_id = f"{imdb_id}_trailer_chunk_{i // shots_per_chunk:04d}"

        chunks.append({
            "chunk_id": chunk_id,
            "movie_id": imdb_id,
            "title": title,
            "source": "trailer",
            "start_seconds": float(start_s),
            "end_seconds": float(end_s),
            "duration": float(end_s - start_s),
            "shot_id": f"shot_{i // shots_per_chunk:04d}",
            "timestamp_source": "trailer_ffmpeg",
            "description": f"Trailer scene {i // shots_per_chunk + 1}",
            "situation": "",
            "vision_setting": "",
            "vision_actions": [],
            "emotional_tone": "",
            "dialogue_text": "[NO_DIALOGUE]",
            "speaker": "",
            "audio_events": [],
            "background_music": "",
            "characters": [],
            "character_emotions": {},
            "cast_in_scene": [],
            "narrative_arc": "trailer",
            "causal_relations": [],
            "screenplay_context": "",
            "script_primary_heading": "",
            "keyframe_paths": [str(p) for p in group],
        })

    return chunks


def process_one_trailer(imdb_id: str, title: str) -> dict:
    """Process one trailer: keyframes → CLIP → chunks. Returns stats."""
    video_path = TRAILERS_DIR / f"{imdb_id}.mp4"
    if not video_path.exists() or video_path.stat().st_size < 100_000:
        return {"imdb_id": imdb_id, "status": "no_video"}

    out_dir = OUTPUT_BASE / imdb_id
    clip_file = out_dir / "clip_embeddings.npy"
    chunks_file = out_dir / "chunks.json"

    if clip_file.exists() and chunks_file.exists():
        return {"imdb_id": imdb_id, "status": "already_done"}

    import numpy as np
    import faiss

    # Step 1: Extract keyframes
    kf_dir = out_dir / "keyframes"
    existing_kfs = sorted(kf_dir.glob("kf_*.jpg")) if kf_dir.exists() else []
    if len(existing_kfs) < 3:
        kf_paths = extract_keyframes(video_path, kf_dir)
    else:
        kf_paths = existing_kfs

    if not kf_paths:
        return {"imdb_id": imdb_id, "status": "no_keyframes"}

    # Step 2: CLIP encode
    embeddings = encode_keyframes_clip(kf_paths)
    np.save(str(out_dir / "clip_embeddings.npy"), embeddings)

    # Step 3: Build per-movie CLIP FAISS
    index = faiss.IndexFlatIP(512)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(out_dir / "clip_chunks.faiss"))

    # Step 4: Build chunks
    chunks = build_chunks(imdb_id, title, kf_paths, KF_INTERVAL_SECS)
    with open(chunks_file, "w") as f:
        json.dump({"chunks": chunks}, f, indent=2, ensure_ascii=False)

    return {
        "imdb_id": imdb_id,
        "status": "done",
        "keyframes": len(kf_paths),
        "clips": len(embeddings),
        "chunks": len(chunks),
    }


def rebuild_global_index():
    """Merge all trailer CLIP embeddings into global visual_index.faiss."""
    import numpy as np
    import faiss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing visual index (BBB + ED001)
    existing_index_path = INDEX_DIR / "visual_index.faiss"
    existing_meta_path = INDEX_DIR / "visual_index_metadata.json"

    all_vecs = []
    all_meta = []

    if existing_index_path.exists():
        idx = faiss.read_index(str(existing_index_path))
        existing_vecs = np.zeros((idx.ntotal, idx.d), dtype=np.float32)
        idx.reconstruct_n(0, idx.ntotal, existing_vecs)
        all_vecs.append(existing_vecs)
        if existing_meta_path.exists():
            all_meta.extend(json.loads(existing_meta_path.read_text()))
        logger.info(f"  Loaded existing visual index: {idx.ntotal} vectors")

    # Add trailer embeddings
    trailer_dirs = sorted(OUTPUT_BASE.glob("tt*/"))
    added = 0
    for td in trailer_dirs:
        clip_file = td / "clip_embeddings.npy"
        if not clip_file.exists():
            continue
        imdb_id = td.name
        vecs = np.load(str(clip_file)).astype(np.float32)
        all_vecs.append(vecs)
        for i in range(len(vecs)):
            all_meta.append({
                "id": f"{imdb_id}_kf_{i:05d}",
                "movie_id": imdb_id,
                "frame_idx": i,
                "path": str(td / "keyframes" / f"kf_{i+1:05d}.jpg"),
                "source": "trailer",
            })
        added += len(vecs)

    if not all_vecs:
        logger.warning("No embeddings found")
        return

    all_vecs_arr = np.vstack(all_vecs).astype(np.float32)
    logger.info(f"  Total vectors: {len(all_vecs_arr)} ({added} from trailers)")

    # Build merged index
    merged_idx = faiss.IndexFlatIP(512)
    merged_idx.add(all_vecs_arr)
    faiss.write_index(merged_idx, str(INDEX_DIR / "visual_index.faiss"))
    with open(INDEX_DIR / "visual_index_metadata.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    logger.info(f"  Saved visual_index.faiss: {merged_idx.ntotal} vectors")


def main():
    parser = argparse.ArgumentParser(description="Process trailer videos through CLIP pipeline")
    parser.add_argument("--check", action="store_true", help="Show status")
    parser.add_argument("--run", action="store_true", help="Process all new trailers")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild global visual index")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    args = parser.parse_args()

    titles = get_titles()
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all downloaded trailers
    trailers = sorted([f for f in TRAILERS_DIR.glob("tt*.mp4") if f.stat().st_size > 100_000])
    processed = [d for d in OUTPUT_BASE.glob("tt*/") if (d / "clip_embeddings.npy").exists()]

    logger.info(f"Downloaded trailers: {len(trailers)}")
    logger.info(f"CLIP-processed:      {len(processed)}")
    logger.info(f"Remaining:           {len(trailers) - len(processed)}")

    if args.check:
        return

    if args.rebuild_index:
        logger.info("Rebuilding global visual index...")
        rebuild_global_index()
        return

    if args.run:
        todo = [t for t in trailers
                if not (OUTPUT_BASE / t.stem / "clip_embeddings.npy").exists()]
        logger.info(f"Processing {len(todo)} trailers ({args.workers} workers)...")

        done_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_one_trailer,
                    t.stem,
                    titles.get(t.stem, t.stem)
                ): t.stem
                for t in todo
            }
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                mid = result["imdb_id"]
                status = result["status"]
                if status == "done":
                    done_count += 1
                    logger.info(f"  ✅ {mid}: {result['keyframes']} kf, {result['clips']} clips, {result['chunks']} chunks")
                elif status == "already_done":
                    done_count += 1
                else:
                    fail_count += 1
                    logger.warning(f"  ❌ {mid}: {status}")

                if (i + 1) % 20 == 0:
                    logger.info(f"  Progress: {i+1}/{len(todo)} | OK:{done_count} FAIL:{fail_count}")

        logger.info(f"Done: {done_count} processed, {fail_count} failed")

        # Rebuild knowledge chunks
        logger.info("Rebuilding trailer chunks file...")
        all_chunks = []
        for td in sorted(OUTPUT_BASE.glob("tt*/")):
            cf = td / "chunks.json"
            if cf.exists():
                raw = json.loads(cf.read_text())
                chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw
                all_chunks.extend(chunks)

        with open(CHUNKS_DIR / "all_trailer_chunks.json", "w") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(all_chunks)} trailer chunks")

        # Rebuild global visual index
        logger.info("Rebuilding global visual index...")
        rebuild_global_index()


if __name__ == "__main__":
    main()
