#!/usr/bin/env python3
"""
enrich_vlm_scenes.py
====================
Run VLM (Groq Llama vision) on existing keyframes to enrich chunks.json
with L2 semantic fields: description, vision_setting, vision_actions, emotional_tone.

Usage:
    python scripts/enrich_vlm_scenes.py --movie-id big_buck_bunny
    python scripts/enrich_vlm_scenes.py --movie-id ed001
    python scripts/enrich_vlm_scenes.py --all
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
logger = logging.getLogger("VLMEnrich")

OUTPUT_BASE = PROJECT_ROOT / "data" / "pipeline_output"

MOVIES = {
    "big_buck_bunny": "Big Buck Bunny (2008)",
    "ed001": "Elephants Dream (2006)",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_chunks(movie_dir: Path) -> tuple[dict, list]:
    p = movie_dir / "chunks.json"
    raw = json.loads(p.read_text())
    if isinstance(raw, dict):
        return raw, raw.get("chunks", [])
    return {"chunks": raw}, raw


def save_chunks(movie_dir: Path, root: dict):
    p = movie_dir / "chunks.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(root, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {p}")


def load_shots(movie_dir: Path) -> list:
    p = movie_dir / "shots.json"
    raw = json.loads(p.read_text())
    if isinstance(raw, dict):
        return raw.get("shots", [])
    return raw


def get_keyframe_for_shot(kf_dir: Path, shot_idx: int, all_kfs: list) -> Path | None:
    """Pick the keyframe that best matches shot_idx (middle of each shot's range)."""
    # Keyframes are ~1 per shot but may vary; pick the corresponding one
    if shot_idx < len(all_kfs):
        return all_kfs[shot_idx]
    return None


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def vlm_analyze_batch(
    client,
    shots: list,
    kf_dir: Path,
    all_kfs: list,
    batch_indices: list[int],
    movie_title: str,
) -> list[dict]:
    """
    Send up to 4 keyframes to Groq VLM and return per-shot analysis.
    Returns list of dicts with: shot_id, description, setting, actions, emotional_tone
    """
    # Build image content blocks
    content_blocks = []
    valid_indices = []

    for idx in batch_indices:
        kf = get_keyframe_for_shot(kf_dir, idx, all_kfs)
        if kf and kf.exists():
            b64 = encode_image(kf)
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            valid_indices.append(idx)

    if not valid_indices:
        return []

    # Build shot timestamp context
    shot_info = []
    for vi, idx in enumerate(valid_indices):
        if idx < len(shots):
            s = shots[idx]
            shot_info.append(f"Frame {vi + 1}: {s.get('start_seconds', 0):.0f}s-{s.get('end_seconds', 0):.0f}s")

    shot_context = "\n".join(shot_info)

    prompt_text = {
        "type": "text",
        "text": (
            f"You are analyzing frames from the video '{movie_title}'.\n\n"
            f"Frame timestamps:\n{shot_context}\n\n"
            "For each frame shown (in order), provide a JSON analysis:\n"
            "{\n"
            '  "scenes": [\n'
            '    {\n'
            '      "description": "What is happening in 1-2 sentences",\n'
            '      "setting": "Location/environment (e.g. outdoor forest, indoor room)",\n'
            '      "actions": ["action1", "action2"],\n'
            '      "emotional_tone": "e.g. calm, tense, joyful, melancholic"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Return ONLY valid JSON. Number of scenes must match number of frames shown."
        )
    }

    content_blocks.insert(0, prompt_text)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": content_blocks}],
            temperature=0.1,
            max_tokens=1024,
        )
        text = response.choices[0].message.content.strip()
        # Extract JSON — handle both plain JSON and markdown-wrapped JSON
        # Strip markdown code fences if present
        text_clean = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text_clean = re.sub(r'\s*```\s*$', '', text_clean, flags=re.MULTILINE).strip()
        json_match = re.search(r'\{.*\}', text_clean, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            scenes = data.get("scenes", [])
            result = []
            for vi, idx in enumerate(valid_indices):
                shot_id = shots[idx]["shot_id"] if idx < len(shots) else f"shot_{idx:04d}"
                if vi < len(scenes):
                    s = scenes[vi]
                    result.append({
                        "shot_id": shot_id,
                        "description": s.get("description", ""),
                        "vision_setting": s.get("setting", ""),
                        "vision_actions": s.get("actions", []),
                        "emotional_tone": s.get("emotional_tone", ""),
                    })
                else:
                    result.append({"shot_id": shot_id, "description": "", "vision_setting": "", "vision_actions": [], "emotional_tone": ""})
            return result
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e} | text: {text[:200]}")
    except Exception as e:
        logger.warning(f"VLM API error: {e}")

    # Fallback: empty entries for all valid shots
    return [{"shot_id": shots[idx]["shot_id"] if idx < len(shots) else f"shot_{idx:04d}",
             "description": "", "vision_setting": "", "vision_actions": [], "emotional_tone": ""}
            for idx in valid_indices]


# ── Main enrichment function ──────────────────────────────────────────────────

def enrich_movie(movie_id: str, api_key: str, force: bool = False):
    movie_dir = OUTPUT_BASE / movie_id
    if not movie_dir.exists():
        logger.error(f"Movie dir not found: {movie_dir}")
        return

    title = MOVIES.get(movie_id, movie_id)
    logger.info(f"=== VLM Enrichment: {title} ({movie_id}) ===")

    root, chunks = load_chunks(movie_dir)
    shots = load_shots(movie_dir)
    kf_dir = movie_dir / "keyframes"
    all_kfs = sorted(kf_dir.glob("*.jpg")) if kf_dir.exists() else []

    logger.info(f"  Chunks: {len(chunks)}, Shots: {len(shots)}, Keyframes: {len(all_kfs)}")

    # Check if already enriched
    already_enriched = sum(
        1 for c in chunks
        if c.get("description") and c["description"] not in ("", "Scene at 0s", f"Scene at {c.get('start_seconds',0):.0f}s")
        and c.get("vision_setting") and c["vision_setting"] != "unknown"
    )
    if already_enriched > len(chunks) * 0.5 and not force:
        logger.info(f"  Already {already_enriched}/{len(chunks)} chunks enriched. Use --force to re-run.")
        return

    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed: pip install groq")
        return

    client = Groq(api_key=api_key)

    # Build shot_id → chunk index mapping
    shot_to_chunk = {c.get("shot_id"): i for i, c in enumerate(chunks)}

    # Process in batches of 4
    batch_size = 4
    total_shots = min(len(shots), len(all_kfs))
    enriched_count = 0
    error_count = 0

    for batch_start in range(0, total_shots, batch_size):
        batch_end = min(batch_start + batch_size, total_shots)
        batch_indices = list(range(batch_start, batch_end))

        analyses = vlm_analyze_batch(client, shots, kf_dir, all_kfs, batch_indices, title)

        for analysis in analyses:
            shot_id = analysis["shot_id"]
            chunk_idx = shot_to_chunk.get(shot_id)
            if chunk_idx is None:
                continue

            chunk = chunks[chunk_idx]
            if analysis.get("description"):
                chunk["description"] = analysis["description"]
                chunk["vision_setting"] = analysis.get("vision_setting", "")
                chunk["vision_actions"] = analysis.get("vision_actions", [])
                chunk["emotional_tone"] = analysis.get("emotional_tone", "")
                enriched_count += 1
            else:
                error_count += 1

        if (batch_start // batch_size + 1) % 5 == 0:
            pct = 100 * batch_end / total_shots
            logger.info(f"  Progress: {batch_end}/{total_shots} shots ({pct:.0f}%) | OK:{enriched_count} ERR:{error_count}")
            # Save checkpoint
            save_chunks(movie_dir, root)

        # Rate limit: Groq allows ~30 req/min on vision
        time.sleep(2.0)

    # Final save
    save_chunks(movie_dir, root)
    logger.info(f"  Done: {enriched_count}/{total_shots} shots enriched | {error_count} errors")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enrich chunks with VLM scene analysis")
    parser.add_argument("--movie-id", type=str, default=None, help="Movie ID (e.g., big_buck_bunny)")
    parser.add_argument("--all", action="store_true", help="Process all movies in MOVIES dict")
    parser.add_argument("--force", action="store_true", help="Re-run even if already enriched")
    parser.add_argument("--api-key", type=str, default=None, help="Groq API key")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.error("GROQ_API_KEY not set. Use --api-key or export GROQ_API_KEY=gsk_...")
        return

    if args.all:
        for mid in MOVIES:
            enrich_movie(mid, api_key, force=args.force)
    elif args.movie_id:
        enrich_movie(args.movie_id, api_key, force=args.force)
    else:
        logger.error("Specify --movie-id or --all")


if __name__ == "__main__":
    main()
