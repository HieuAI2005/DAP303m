#!/usr/bin/env python3
"""
build_activitynet_chunks.py
==========================
Build 5-Layer chunks from ActivityNet Captions annotations + 820 real videos.

Sources:
  - Annotations: ActivityNet GitHub (activity_net.v1-3.min.json)
  - Videos: VideoRag data/activitynet_videos/ (820 videos)
  - Keyframes: VideoRag data/activitynet_keyframes/ (99,452 keyframes)
  - Visual Index: VideoRag benchmark visual_index_benchmark.faiss (linked)

Output:
  - data/pipeline_output/activitynet_chunks/all_chunks.json
  - FAISS knowledge index (sentence embeddings of captions)

Usage:
    python scripts/build_activitynet_chunks.py --execute
    python scripts/build_activitynet_chunks.py --execute --build-index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ActivityNetBuilder")


# ── Paths ────────────────────────────────────────────────────────────────────

AN_CAPTIONS = PROJECT_ROOT / "data" / "ActivityNet_Captions" / "activity_net.v1-3.json"
AN_VIDEOS = Path("/home/hiwe/project/DAP303m/VideoRag/data/activitynet_videos")
AN_KEYFRAMES = Path("/home/hiwe/project/DAP303m/VideoRag/data/activitynet_keyframes")
OUTPUT_CHUNKS = PROJECT_ROOT / "data" / "pipeline_output" / "activitynet_chunks"
OUTPUT_INDEXES = PROJECT_ROOT / "data" / "pipeline_output" / "indexes"


# ── Schema ──────────────────────────────────────────────────────────────────

def build_chunk(video_id: str, segment: dict, subset: str, duration: float) -> dict:
    """Build a 5-Layer chunk from ActivityNet segment."""
    start_sec = float(segment["segment"][0])
    end_sec = float(segment["segment"][1])
    caption = segment["label"]

    chunk_id = f"an_{video_id}_{int(start_sec*100):08d}"

    return {
        # L1: Temporal Anchor
        "chunk_id": chunk_id,
        "video_id": video_id,
        "movie_id": "activitynet",
        "start_seconds": start_sec,
        "end_seconds": end_sec,
        "duration": end_sec - start_sec,
        "frame_start": None,
        "frame_end": None,

        # L2: Semantic Description
        "description": caption,
        "situation": _classify_situation(caption),
        "vision_setting": _infer_setting(caption),
        "vision_actions": _infer_actions(caption),
        "emotional_tone": _infer_emotion(caption),

        # L3: Dialogue & Audio
        "text": caption,
        "dialogue_text": caption,
        "speaker": "",
        "audio_events": ["speech"],
        "background_music": "",

        # L4: Cast & Characters
        "characters": [],
        "cast_in_scene": [],
        "character_emotions": {},
        "face_tracking_ids": [],
        "action_labels": [caption],  # ActivityNet caption IS the action label

        # L5: Narrative
        "script_heading": caption,
        "screenplay_context": caption,
        "narrative_arc": _infer_narrative(caption),
        "causal_relations": [],
        "scene_graph": {
            "action_class": caption,
        },

        # Metadata
        "source": "activitynet",
        "split": subset,
        "language": "en",
        "type": "activity_video",
        "keyframe_paths": [],  # Will be filled if keyframes exist
        "num_keyframes": 0,
        "video_path": str(AN_VIDEOS / f"{video_id}.mp4") if (AN_VIDEOS / f"{video_id}.mp4").exists() else "",
        "subset": subset,
        "video_duration": duration,
    }


def _classify_situation(caption: str) -> str:
    text = caption.lower()
    if any(w in text for w in ["walking", "running", "jogging", "exercising"]):
        return "physical_activity"
    if any(w in text for w in ["cooking", "baking", "frying", "grilling"]):
        return "cooking"
    if any(w in text for w in ["playing", "guitar", "piano", "drums", "music"]):
        return "music_performance"
    if any(w in text for w in ["talking", "speaking", "conversation"]):
        return "conversation"
    if any(w in text for w in ["dancing", "dance"]):
        return "dancing"
    if any(w in text for w in ["grooming", "shaving", "brushing", "washing"]):
        return "grooming"
    if any(w in text for w in ["assembling", "repairing", "fixing"]):
        return "repairing"
    if any(w in text for w in ["children", "playing", "kids"]):
        return "child_play"
    if any(w in text for w in ["eating", "drinking", "having meal"]):
        return "eating"
    if any(w in text for w in ["dog", "cat", "pet", "animal"]):
        return "pet_activity"
    return "activity"


def _infer_setting(caption: str) -> str:
    text = caption.lower()
    if any(w in text for w in ["gym", "workout", "exercise"]):
        return "indoor_gym"
    if any(w in text for w in ["kitchen", "cooking", "baking"]):
        return "indoor_kitchen"
    if any(w in text for w in ["outdoor", "park", "street", "garden", "yard"]):
        return "outdoor"
    if any(w in text for w in ["beach", "ocean", "water"]):
        return "outdoor_beach"
    if any(w in text for w in ["bathroom", "shower", "toilet"]):
        return "indoor_bathroom"
    if any(w in text for w in ["bedroom", "sleeping", "lying"]):
        return "indoor_bedroom"
    if any(w in text for w in ["car", "driving", "vehicle"]):
        return "outdoor_vehicle"
    return "indoor"


def _infer_actions(caption: str) -> list:
    text = caption.lower()
    actions = []
    verbs = ["walking", "running", "jumping", "dancing", "cooking", "eating", "drinking",
             "talking", "laughing", "crying", "singing", "playing", "working", "reading",
             "writing", "driving", "swimming", "fighting", "grooming", "exercising"]
    for v in verbs:
        if v in text:
            actions.append(v)
    return actions if actions else ["interacting"]


def _infer_emotion(caption: str) -> str:
    text = caption.lower()
    if any(w in text for w in ["happy", "laughing", "joy", "celebrating"]):
        return "positive_happy"
    if any(w in text for w in ["sad", "crying", "grief"]):
        return "negative_sad"
    if any(w in text for w in ["angry", "fighting", "arguing"]):
        return "negative_angry"
    if any(w in text for w in ["scary", "fear", "running away"]):
        return "negative_fearful"
    if any(w in text for w in ["romantic", "kissing", "couple"]):
        return "positive_romantic"
    if any(w in text for w in ["funny", "comedy", "joking"]):
        return "positive_funny"
    return "neutral"


def _infer_narrative(caption: str) -> str:
    text = caption.lower()
    if any(w in text for w in ["intro", "beginning", "starting"]):
        return "introduction"
    if any(w in text for w in ["finish", "end", "completing"]):
        return "resolution"
    return "scene"


# ── Main ─────────────────────────────────────────────────────────────────────

def build_activitynet_chunks(dry_run: bool = False, build_index: bool = False) -> None:
    print("\n" + "=" * 70)
    print("  🎬 ActivityNet Chunk Builder")
    print("=" * 70)

    # Load annotations
    if not AN_CAPTIONS.exists():
        logger.error(f"ActivityNet captions not found: {AN_CAPTIONS}")
        logger.info("Run: curl -s 'https://raw.githubusercontent.com/...activity_net.v1-3.min.json' -o data/ActivityNet_Captions/activity_net.v1-3.json")
        return

    with open(AN_CAPTIONS, encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("database", {})
    taxonomy = {t["nodeId"]: t["nodeName"] for t in data.get("taxonomy", [])}

    # Available videos in VideoRag
    # Caption IDs: sJFgo9H6zNo, VideoRag IDs: v_sJFgo9H6zNo
    vr_videos = {f.stem: f.stem for f in AN_VIDEOS.glob("*.mp4")}
    vr_videos_norm = {f.stem.replace("v_", ""): f.stem for f in AN_VIDEOS.glob("*.mp4")}
    logger.info(f"VideoRag ActivityNet videos: {len(vr_videos)}")

    # Available keyframes
    vr_keyframes = {d.name: d.name for d in AN_KEYFRAMES.iterdir() if d.is_dir()}
    vr_keyframes_norm = {d.name.replace("v_", ""): d.name for d in AN_KEYFRAMES.iterdir() if d.is_dir()}
    logger.info(f"VideoRag ActivityNet keyframe dirs: {len(vr_keyframes)}")

    # Build chunks
    chunks = []
    stats = {"total": 0, "with_video": 0, "with_keyframes": 0, "by_subset": {}}

    for video_id, info in videos.items():
        subset = info.get("subset", "")
        duration = float(info.get("duration", 0))
        annotations = info.get("annotations", [])

        if not annotations:
            continue

        # Try both with and without v_ prefix
        has_video = video_id in vr_videos or video_id in vr_videos_norm
        has_kf = video_id in vr_keyframes or video_id in vr_keyframes_norm

        # Get the correct VideoRag keyframe directory name
        vr_video_id = vr_videos.get(video_id) or vr_videos_norm.get(video_id, video_id)
        vr_kf_id = vr_keyframes.get(video_id) or vr_keyframes_norm.get(video_id, video_id)

        for seg in annotations:
            chunk = build_chunk(video_id, seg, subset, duration)
            chunk["has_video"] = has_video
            chunk["has_keyframes"] = has_kf

            # Add keyframe paths if available
            if has_kf:
                kf_dir = AN_KEYFRAMES / vr_kf_id
                frames = sorted(kf_dir.glob("*.jpg"))
                # Filter to frames within this segment's time range
                start_s = chunk["start_seconds"]
                end_s = chunk["end_seconds"]
                fps_est = len(frames) / duration if duration > 0 else 1
                relevant = []
                for f in frames:
                    # Frame filename: frame_XXXXX.jpg → extract frame number
                    fname = f.stem
                    if fname.startswith("frame_"):
                        frame_num = int(fname.replace("frame_", ""))
                        frame_time = frame_num / fps_est
                        if start_s <= frame_time <= end_s:
                            relevant.append(str(f))
                chunk["keyframe_paths"] = relevant
                chunk["num_keyframes"] = len(relevant)

            chunks.append(chunk)
            stats["total"] += 1
            if has_video:
                stats["with_video"] += 1
            if has_kf:
                stats["with_keyframes"] += 1
            stats["by_subset"][subset] = stats["by_subset"].get(subset, 0) + 1

    logger.info(f"Built {stats['total']:,} chunks")
    logger.info(f"  With VideoRag video: {stats['with_video']:,}")
    logger.info(f"  With VideoRag keyframes: {stats['with_keyframes']:,}")
    logger.info(f"  By subset: {stats['by_subset']}")

    if dry_run:
        logger.info("Dry run — not saving. Re-run with --execute")
        return

    # Save chunks
    OUTPUT_CHUNKS.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_CHUNKS / "all_chunks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "source": "activitynet",
                "version": "v1-3",
                "num_chunks": len(chunks),
                "stats": stats,
            },
            "chunks": chunks,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved: {out_file} ({len(chunks):,} chunks)")

    # Build FAISS knowledge index
    if build_index:
        _build_knowledge_index(chunks)


def _build_knowledge_index(chunks: list) -> None:
    """Build FAISS knowledge index from captions."""
    from sentence_transformers import SentenceTransformer

    logger.info("\n🔢 Building ActivityNet knowledge FAISS index...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["description"] for c in chunks if c.get("description")]
    ids = [c["chunk_id"] for c in chunks if c.get("description")]

    logger.info(f"  Embedding {len(texts):,} captions...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = embeddings.astype("float32")
    np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    OUTPUT_INDEXES.mkdir(parents=True, exist_ok=True)
    idx_path = OUTPUT_INDEXES / "knowledge_activitynet.faiss"
    meta_path = OUTPUT_INDEXES / "knowledge_activitynet_map.json"

    faiss.write_index(index, str(idx_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([{"chunk_id": cid, "description": desc} for cid, desc in zip(ids, texts)], f)

    size_mb = idx_path.stat().st_size // (1024 * 1024)
    logger.info(f"  ✅ ActivityNet knowledge index: {index.ntotal:,} vectors, {size_mb}MB")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build ActivityNet chunks from captions")
    parser.add_argument("--execute", action="store_true", help="Actually build (dry run by default)")
    parser.add_argument("--build-index", action="store_true", help="Also build FAISS knowledge index")
    args = parser.parse_args()

    build_activitynet_chunks(dry_run=not args.execute, build_index=args.build_index)


if __name__ == "__main__":
    main()
