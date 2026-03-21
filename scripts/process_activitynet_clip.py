#!/usr/bin/env python3
"""
process_activitynet_clip.py
===========================
Run CLIP pipeline on ActivityNet videos:
  1. Extract keyframes (1 per 3 sec via ffmpeg)
  2. Encode with CLIP ViT-B/32
  3. Build per-video FAISS visual index
  4. Merge into activitynet_visual_index.faiss
  5. Save chunks.json per video

Usage:
    python scripts/process_activitynet_clip.py --check
    python scripts/process_activitynet_clip.py --run --workers 4
    python scripts/process_activitynet_clip.py --rebuild-index
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
logger = logging.getLogger("ActivityNetCLIP")

VIDEOS_DIR = Path("/home/hiwe/project/DAP303m/VideoRag/data/activitynet_videos")
OUTPUT_BASE = PROJECT_ROOT / "data" / "pipeline_output" / "activitynet"
INDEX_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "indexes"
CHUNKS_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "activitynet_chunks"
KF_INTERVAL_SECS = 3


def extract_keyframes(video_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "kf_%05d.jpg"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{KF_INTERVAL_SECS},scale=224:224:force_original_aspect_ratio=decrease",
        "-q:v", "2", str(pattern), "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:
        logger.warning(f"ffmpeg error for {video_path.name}: {e}")
    return sorted(out_dir.glob("kf_*.jpg"))


def encode_keyframes_clip(kf_paths: list[Path]):
    import numpy as np
    import torch
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model.eval()
    except Exception:
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cpu")

    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embeddings = []
    for i in range(0, len(kf_paths), 32):
        batch = kf_paths[i:i + 32]
        imgs = []
        for p in batch:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        imgs_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            try:
                feats = model.encode_image(imgs_tensor)
            except Exception:
                feats = torch.zeros(len(imgs), 512)
        feats = feats.cpu().float().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.where(norms == 0, 1, norms)
        embeddings.append(feats)
    import numpy as np
    return np.vstack(embeddings) if embeddings else np.zeros((len(kf_paths), 512), dtype="float32")


def build_chunks(video_id: str, kf_paths: list[Path]) -> list[dict]:
    chunks = []
    shots_per_chunk = max(1, 30 // KF_INTERVAL_SECS)
    for i in range(0, len(kf_paths), shots_per_chunk):
        group = kf_paths[i:i + shots_per_chunk]
        start_s = i * KF_INTERVAL_SECS
        end_s = min((i + shots_per_chunk) * KF_INTERVAL_SECS, len(kf_paths) * KF_INTERVAL_SECS)
        chunks.append({
            "chunk_id": f"{video_id}_chunk_{i // shots_per_chunk:04d}",
            "movie_id": video_id,
            "title": video_id,
            "source": "activitynet",
            "start_seconds": float(start_s),
            "end_seconds": float(end_s),
            "duration": float(end_s - start_s),
            "shot_id": f"shot_{i // shots_per_chunk:04d}",
            "description": f"ActivityNet clip {i // shots_per_chunk + 1}",
            "situation": "", "vision_setting": "", "vision_actions": [],
            "emotional_tone": "", "dialogue_text": "[NO_DIALOGUE]",
            "characters": [], "narrative_arc": "activitynet",
            "keyframe_paths": [str(p) for p in group],
        })
    return chunks


def process_one_video(video_path: Path) -> dict:
    import numpy as np
    import faiss

    video_id = video_path.stem
    out_dir = OUTPUT_BASE / video_id
    clip_file = out_dir / "clip_embeddings.npy"
    chunks_file = out_dir / "chunks.json"

    if clip_file.exists() and chunks_file.exists():
        return {"video_id": video_id, "status": "already_done"}

    if video_path.stat().st_size < 50_000:
        return {"video_id": video_id, "status": "too_small"}

    kf_dir = out_dir / "keyframes"
    existing = sorted(kf_dir.glob("kf_*.jpg")) if kf_dir.exists() else []
    kf_paths = existing if len(existing) >= 2 else extract_keyframes(video_path, kf_dir)

    if not kf_paths:
        return {"video_id": video_id, "status": "no_keyframes"}

    embeddings = encode_keyframes_clip(kf_paths)
    np.save(str(out_dir / "clip_embeddings.npy"), embeddings)

    index = faiss.IndexFlatIP(512)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(out_dir / "clip_chunks.faiss"))

    chunks = build_chunks(video_id, kf_paths)
    with open(chunks_file, "w") as f:
        json.dump({"chunks": chunks}, f, indent=2)

    return {"video_id": video_id, "status": "done",
            "keyframes": len(kf_paths), "chunks": len(chunks)}


def rebuild_global_index():
    import numpy as np
    import faiss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    all_vecs, all_meta = [], []

    for vd in sorted(OUTPUT_BASE.glob("v_*/")):
        clip_file = vd / "clip_embeddings.npy"
        if not clip_file.exists():
            continue
        vid = vd.name
        vecs = np.load(str(clip_file)).astype("float32")
        all_vecs.append(vecs)
        for i in range(len(vecs)):
            all_meta.append({"id": f"{vid}_kf_{i:05d}", "video_id": vid,
                             "frame_idx": i, "source": "activitynet"})

    if not all_vecs:
        logger.warning("No embeddings found")
        return

    arr = np.vstack(all_vecs).astype("float32")
    idx = faiss.IndexFlatIP(512)
    idx.add(arr)
    faiss.write_index(idx, str(INDEX_DIR / "activitynet_visual_index.faiss"))
    with open(INDEX_DIR / "activitynet_visual_meta.json", "w") as f:
        json.dump(all_meta, f)
    logger.info(f"Saved activitynet_visual_index.faiss: {idx.ntotal} vectors")

    # Save all chunks
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for vd in sorted(OUTPUT_BASE.glob("v_*/")):
        cf = vd / "chunks.json"
        if cf.exists():
            raw = json.loads(cf.read_text())
            all_chunks.extend(raw.get("chunks", raw) if isinstance(raw, dict) else raw)
    with open(CHUNKS_DIR / "all_activitynet_chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)
    logger.info(f"Saved {len(all_chunks)} ActivityNet chunks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    videos = sorted([f for f in VIDEOS_DIR.glob("v_*.mp4") if f.stat().st_size > 50_000])
    processed = [d for d in OUTPUT_BASE.glob("v_*/") if (d / "clip_embeddings.npy").exists()]

    logger.info(f"ActivityNet videos: {len(videos)}")
    logger.info(f"CLIP-processed:     {len(processed)}")
    logger.info(f"Remaining:          {len(videos) - len(processed)}")

    if args.check:
        return

    if args.rebuild_index:
        rebuild_global_index()
        return

    if args.run:
        todo = [v for v in videos if not (OUTPUT_BASE / v.stem / "clip_embeddings.npy").exists()]
        logger.info(f"Processing {len(todo)} videos ({args.workers} workers)...")
        done_count = fail_count = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_video, v): v.stem for v in todo}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                vid = result["video_id"]
                if result["status"] == "done":
                    done_count += 1
                    logger.info(f"  OK {vid}: {result['keyframes']} kf, {result['chunks']} chunks")
                elif result["status"] == "already_done":
                    done_count += 1
                else:
                    fail_count += 1
                    logger.warning(f"  ERR {vid}: {result['status']}")
                if (i + 1) % 50 == 0:
                    logger.info(f"  Progress: {i+1}/{len(todo)} | OK:{done_count} FAIL:{fail_count}")

        logger.info(f"Done: {done_count} processed, {fail_count} failed")
        logger.info("Rebuilding ActivityNet index...")
        rebuild_global_index()


if __name__ == "__main__":
    main()
