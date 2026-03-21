#!/usr/bin/env python3
"""
build_youcook2_pipeline.py
==========================
Build VideoSceneRAG pipeline for YouCook2 dataset.

YouCook2: 4,650 segments from 609 YouTube cooking videos.
  - val:   3,179 segments (414 videos) → benchmark test set
  - test:  1,471 segments (195 videos) → evaluation

5-Layer Schema mapping:
  L1 Temporal:   chunk_id, video_id, youtube_id, start_seconds, end_seconds
  L2 Semantic:   description (sentence), situation, vision_setting,
                 vision_actions, emotional_tone
  L3 Dialogue:    dialogue_text (from Whisper), speaker, audio_events
  L4 Cast:       characters (empty — cooking videos)
  L5 Narrative:   narrative_arc, causal_relations, screenplay_context

Usage:
    python scripts/build_youcook2_pipeline.py --build-chunks
    python scripts/build_youcook2_pipeline.py --build-index
    python scripts/build_youcook2_pipeline.py --check
    python scripts/build_youcook2_pipeline.py --download-videos --limit 100
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
logger = logging.getLogger("YouCook2Pipeline")


# ── Paths ────────────────────────────────────────────────────────────────────

ANNOTATIONS_JSON = PROJECT_ROOT / "data" / "YouCook2" / "youcook2_annotations.json"
VIDEO_DIR = PROJECT_ROOT / "data" / "YouCook2" / "videos"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "YouCook2" / "transcripts"
OUTPUT_CHUNKS = PROJECT_ROOT / "data" / "pipeline_output" / "youcook2_chunks"
PROGRESS_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "youcook2_downloaded.txt"


# ── Load annotations ─────────────────────────────────────────────────────────

def load_annotations() -> dict:
    """Load YouCook2 annotations from HuggingFace JSON."""
    if not ANNOTATIONS_JSON.exists():
        logger.error(f"Annotations not found: {ANNOTATIONS_JSON}")
        logger.info("Run: python scripts/build_youcook2_pipeline.py --download-annotations")
        return {}

    with open(ANNOTATIONS_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    # Build lookup: youtube_id → list of segments
    all_entries = []
    for entry in raw:
        youtube_id = entry.get("youtube_id", "")
        if not youtube_id:
            continue
        all_entries.append({
            "id": entry.get("id", ""),
            "youtube_id": youtube_id,
            "video_url": entry.get("video_url", ""),
            "recipe_type": entry.get("recipe_type", ""),
            "segment": entry.get("segment", [0, 0]),
            "sentence": entry.get("sentence") or "",
            "video_path": entry.get("video_path", ""),
        })

    logger.info(f"Loaded {len(all_entries)} annotations")
    return all_entries


def download_annotations():
    """Download YouCook2 annotations from HuggingFace."""
    from datasets import load_dataset

    ANNOTATIONS_JSON.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading YouCook2 from HuggingFace (lmms-lab/YouCook2)...")
    ds = load_dataset("lmms-lab/YouCook2")

    all_entries = []
    for split_name, split_ds in ds.items():
        for entry in split_ds:
            if entry.get("sentence") is None:
                continue  # Skip test entries with no sentence
            all_entries.append({
                "id": entry["id"],
                "youtube_id": entry["youtube_id"],
                "video_url": entry["video_url"],
                "recipe_type": str(entry.get("recipe_type", "")),
                "segment": list(entry["segment"]),
                "sentence": entry["sentence"] or "",
                "split": split_name,
            })

    logger.info(f"Total entries with sentences: {len(all_entries)}")

    # Save combined
    with open(ANNOTATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved: {ANNOTATIONS_JSON}")

    # Show split distribution
    splits = {}
    for e in all_entries:
        splits[e["split"]] = splits.get(e["split"], 0) + 1
    for s, n in splits.items():
        logger.info(f"  {s}: {n} entries")

    return all_entries


# ── Build chunks ─────────────────────────────────────────────────────────────

# YouCook2 uses 3-digit recipe type codes (101-425)
# Mapping: 1xx=baking, 2xx=pasta/grains, 3xx=vegetables, 4xx=meat
RECIPE_CATEGORIES = {
    "1": "baking",  # 101-130
    "2": "main_dish",  # 201-230
    "3": "vegetable",  # 301-330
    "4": "meat",  # 401-430
}


def get_recipe_category(recipe_type: str) -> str:
    """Map YouCook2 recipe type to category."""
    code = str(recipe_type).zfill(3)
    prefix = code[0]  # first digit
    return RECIPE_CATEGORIES.get(prefix, "cooking")


def infer_situation(sentence: str, recipe_type: str) -> str:
    """Infer cooking situation from sentence."""
    s = sentence.lower()
    verbs = []
    for v in ["add", "pour", "stir", "mix", "heat", "cook", "bake",
              "fry", "cut", "slice", "boil", "simmer", "roast",
              "grill", "steam", "season", "flip", "plate", "garnish"]:
        if v in s:
            verbs.append(v)
    return verbs[0] if verbs else "cooking"


def infer_setting(sentence: str, recipe_type: str) -> str:
    """Infer kitchen setting."""
    s = sentence.lower()
    if "oven" in s or "bake" in s or "roast" in s:
        return "kitchen_oven"
    if "pan" in s or "fry" in s or "saute" in s:
        return "kitchen_stovetop"
    if "pot" in s or "boil" in s or "steam" in s:
        return "kitchen_stovetop"
    if "cutting board" in s or "knife" in s:
        return "kitchen_counter"
    return "kitchen"


def infer_actions(sentence: str) -> list:
    """Extract cooking actions from sentence."""
    s = sentence.lower()
    actions = []
    action_map = {
        "add": "adding", "pour": "pouring", "stir": "stirring",
        "mix": "mixing", "heat": "heating", "cook": "cooking",
        "bake": "baking", "fry": "frying", "cut": "cutting",
        "slice": "slicing", "boil": "boiling", "simmer": "simmering",
        "roast": "roasting", "grill": "grilling", "steam": "steaming",
        "season": "seasoning", "flip": "flipping", "plate": "plating",
        "garnish": "garnishing", "chop": "chopping", "whisk": "whisking",
        "preheat": "preheating", "blend": "blending", "knead": "kneading",
        "spread": "spreading", "drizzle": "drizzling", "sprinkle": "sprinkling",
    }
    for k, v in action_map.items():
        if k in s:
            actions.append(v)
    return actions if actions else ["preparing"]


def build_youcook2_chunks() -> list:
    """Build 5-Layer chunks from YouCook2 annotations."""
    entries = load_annotations()
    if not entries:
        return []

    chunks = []
    for entry in entries:
        yt_id = entry["youtube_id"]
        seg = entry["segment"]
        start_sec = float(seg[0])
        end_sec = float(seg[1])
        sentence = entry["sentence"].strip()
        recipe_type = str(entry.get("recipe_type", ""))

        chunk_id = f"yc2_{yt_id}_{start_sec:.0f}_{end_sec:.0f}"

        chunks.append({
            # L1: Temporal Anchor
            "chunk_id": chunk_id,
            "video_id": yt_id,
            "youtube_id": yt_id,
            "title": f"YouCook2 {get_recipe_category(recipe_type)}",
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration": end_sec - start_sec,
            "frame_start": None,
            "frame_end": None,
            "scene_id": chunk_id,
            "recipe_type": recipe_type,
            "recipe_category": get_recipe_category(recipe_type),

            # L2: Semantic Description
            "description": sentence,
            "text": sentence,
            "situation": infer_situation(sentence, recipe_type),
            "vision_setting": infer_setting(sentence, recipe_type),
            "vision_actions": infer_actions(sentence),
            "emotional_tone": "neutral",  # cooking videos, neutral

            # L3: Dialogue & Audio
            "dialogue_text": "",  # Will fill from Whisper
            "speaker": "",
            "audio_events": [],
            "background_music": "",

            # L4: Cast & Characters
            "characters": [],  # No characters in cooking videos
            "cast_in_scene": [],
            "character_emotions": {},
            "face_tracking_ids": [],
            "action_labels": [get_recipe_category(recipe_type)],

            # L5: Narrative
            "script_heading": get_recipe_category(recipe_type),
            "screenplay_context": sentence,
            "narrative_arc": "instructional_step",
            "causal_relations": [],  # No graph in YouCook2
            "scene_graph": {
                "entities": [],
                "triplets": [],
                "situation": sentence,
            },

            # Metadata
            "source": "youcook2",
            "split": entry.get("split", "val"),
            "language": "en",
            "type": "cooking_instruction",
            "keyframe_paths": [],
            "num_keyframes": 0,
        })

    return chunks


# ── Video Download ────────────────────────────────────────────────────────────

def download_video(yt_id: str, output_dir: Path) -> tuple[str, bool]:
    """Download one YouCook2 video via yt-dlp (30-sec clip)."""
    out_path = output_dir / f"{yt_id}.mp4"
    if out_path.exists() and out_path.stat().st_size > 5000:
        return yt_id, True

    # YouCook2 videos can be long (avg 5-10 min), download full for Whisper
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "best[height<=720]/best",
        "-o", str(out_path.resolve()),
        "--no-check-certificates",
        "--user-agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "--max-filesize", "200M",
        f"https://www.youtube.com/watch?v={yt_id}",
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=180)
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


def download_videos(workers: int = 2, limit: int = 0):
    """Download YouCook2 videos from YouTube."""
    entries = load_annotations()
    if not entries:
        return

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Get unique YouTube IDs
    all_yt_ids = list({e["youtube_id"] for e in entries if e["youtube_id"]})

    # Skip already downloaded
    downloaded = {f.stem for f in VIDEO_DIR.glob("*.mp4")}
    remaining = [yid for yid in all_yt_ids if yid not in downloaded]
    logger.info(f"YouTube IDs: {len(all_yt_ids)}")
    logger.info(f"Already downloaded: {len(downloaded)}")
    logger.info(f"Remaining: {len(remaining)}")

    if limit:
        remaining = remaining[:limit]

    if not remaining:
        logger.info("All videos already downloaded!")
        return

    logger.info(f"Starting download ({workers} workers)...")

    done = set(downloaded)
    success = 0
    fail = 0
    total = len(remaining)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_video, yid, VIDEO_DIR): yid for yid in remaining}

        for i, future in enumerate(as_completed(futures)):
            yt_id, ok = future.result()
            if ok:
                success += 1
                done.add(yt_id)
            else:
                fail += 1

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                logger.info(f"  Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%) "
                           f"| ✅{success} ❌{fail} | {rate:.1f}/min")
                with open(PROGRESS_FILE, "w") as f:
                    f.write("\n".join(sorted(done)))

    with open(PROGRESS_FILE, "w") as f:
        f.write("\n".join(sorted(done)))

    logger.info(f"✅ Done! {success}/{total} downloaded this run | Total: {len(done)}")


def check_status():
    """Show YouCook2 pipeline status."""
    entries = load_annotations()
    if not entries:
        logger.warning("No annotations found. Run: python scripts/build_youcook2_pipeline.py --download-annotations")
        return

    all_yt_ids = {e["youtube_id"] for e in entries}
    downloaded = {f.stem for f in VIDEO_DIR.glob("*.mp4")}
    on_disk = [yid for yid in all_yt_ids if yid in downloaded]

    chunks_path = OUTPUT_CHUNKS / "all_chunks.json"
    if chunks_path.exists():
        with open(chunks_path) as f:
            raw = json.load(f)
        chunks = raw if isinstance(raw, list) else raw.get("chunks", [])
    else:
        chunks = []

    logger.info("=== YouCook2 Pipeline Status ===")
    logger.info(f"  Annotations:     {len(entries)} entries ({len(all_yt_ids)} videos)")
    logger.info(f"  Videos on disk:  {len(on_disk)}/{len(all_yt_ids)}")
    logger.info(f"  Chunks built:    {len(chunks)}")

    # Check FAISS index
    idx = PROJECT_ROOT / "data" / "pipeline_output" / "indexes" / "knowledge_youcook2.faiss"
    if idx.exists():
        import faiss
        idx_obj = faiss.read_index(str(idx))
        logger.info(f"  FAISS index:     {idx_obj.ntotal:,} vectors")
    else:
        logger.info("  FAISS index:     ❌ Not built")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YouCook2 VideoSceneRAG Pipeline")
    parser.add_argument("--download-annotations", action="store_true",
                        help="Download annotations from HuggingFace")
    parser.add_argument("--build-chunks", action="store_true",
                        help="Build 5-Layer chunks")
    parser.add_argument("--build-index", action="store_true",
                        help="Build FAISS index for YouCook2")
    parser.add_argument("--download-videos", action="store_true",
                        help="Download YouTube videos")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit downloads")
    parser.add_argument("--check", action="store_true", help="Check status")
    args = parser.parse_args()

    if args.download_annotations:
        download_annotations()

    if args.build_chunks:
        chunks = build_youcook2_chunks()
        if chunks:
            OUTPUT_CHUNKS.mkdir(parents=True, exist_ok=True)
            out_file = OUTPUT_CHUNKS / "all_chunks.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": {
                        "source": "youcook2",
                        "version": "v1",
                        "num_chunks": len(chunks),
                        "num_videos": len({c["youtube_id"] for c in chunks}),
                    },
                    "chunks": chunks,
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(chunks)} chunks: {out_file}")

            # Field coverage
            n = len(chunks)
            for f in ["situation", "vision_setting", "description",
                      "narrative_arc", "screenplay_context"]:
                cnt = sum(1 for c in chunks if c.get(f))
                logger.info(f"  {f}: {cnt}/{n} ({100*cnt/n:.1f}%)")

    if args.build_index:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        chunks_path = OUTPUT_CHUNKS / "all_chunks.json"
        if not chunks_path.exists():
            logger.error("Chunks not found. Run --build-chunks first.")
            return

        with open(chunks_path, encoding="utf-8") as f:
            raw = json.load(f)
        chunks = raw if isinstance(raw, list) else raw.get("chunks", [])
        logger.info(f"Building index for {len(chunks)} chunks...")

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Build text for embedding
        texts = []
        for c in chunks:
            parts = [
                c.get("description", ""),
                c.get("situation", ""),
                c.get("vision_setting", ""),
                " ".join(c.get("vision_actions", [])),
                c.get("screenplay_context", ""),
            ]
            texts.append(" | ".join(p for p in parts if p))

        embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
        embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= (norms + 1e-8)

        # Build index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        idx_dir = PROJECT_ROOT / "data" / "pipeline_output" / "indexes"
        idx_dir.mkdir(parents=True, exist_ok=True)
        idx_path = idx_dir / "knowledge_youcook2.faiss"
        meta_path = idx_dir / "knowledge_youcook2_map.json"

        faiss.write_index(index, str(idx_path))

        # Save metadata
        meta = []
        for c in chunks:
            meta.append({
                "chunk_id": c["chunk_id"],
                "video_id": c.get("youtube_id", ""),
                "title": c.get("title", ""),
                "description": c.get("description", ""),
                "start_seconds": c.get("start_seconds"),
                "end_seconds": c.get("end_seconds"),
            })

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        logger.info(f"✅ Index: {index.ntotal:,} vectors, {idx_path.stat().st_size/1e6:.1f}MB")
        logger.info(f"    Metadata: {len(meta)} entries")

    if args.download_videos:
        download_videos(workers=args.workers, limit=args.limit)

    if args.check or (not any([args.download_annotations, args.build_chunks,
                                args.build_index, args.download_videos])):
        check_status()


if __name__ == "__main__":
    main()
