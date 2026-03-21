#!/usr/bin/env python3
"""
vlm_enrich_videorag.py
======================
VLM enrichment for VideoRag chunks using Groq Llama 4 Scout API.

Enriches L2-L5 fields:
  L2: situation, vision_setting, vision_actions, emotional_tone
  L4: characters, character_emotions
  L5: narrative_arc, causal_relations, screenplay_context

Usage:
    python scripts/vlm_enrich_videorag.py              # Dry run (1 chunk)
    python scripts/vlm_enrich_videorag.py --execute     # Enrich all chunks
    python scripts/vlm_enrich_videorag.py --batch 5   # Test with 5 chunks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VLMEnricher")

# ── Paths ────────────────────────────────────────────────────────────────────

CHUNKS_IN = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "all_chunks.json"
CHUNKS_OUT = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "all_chunks_enriched.json"
PROGRESS_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "enrich_progress.json"


# ── VLM Client ───────────────────────────────────────────────────────────────

class GroqVLMClient:
    """Groq Llama 4 Scout VLM for scene analysis."""

    def __init__(self, api_key: str = None):
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.rate_limit_per_min = 30  # conservative

    def analyze_scene(self, chunk: dict) -> dict:
        """Analyze a single scene and return enriched fields."""
        prompt = self._build_prompt(chunk)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a movie scene analyst. Return ONLY a valid JSON object "
                            "with these keys: situation, vision_setting, vision_actions, "
                            "emotional_tone, narrative_arc, character_emotions (dict of "
                            "{name: emotion}), screenplay_context (1-2 sentence summary). "
                            "All values must be strings or objects. No extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                # Validate keys
                for k in ["situation", "vision_setting", "vision_actions",
                           "emotional_tone", "narrative_arc",
                           "character_emotions", "screenplay_context"]:
                    if k not in result:
                        result[k] = ""
                if isinstance(result.get("vision_actions"), str):
                    result["vision_actions"] = [result["vision_actions"]]
                return result
            else:
                logger.warning(f"  No JSON in response: {content[:100]}")
                return {}
        except Exception as e:
            logger.warning(f"  API error: {e}")
            return {}

    def _build_prompt(self, chunk: dict) -> str:
        title = chunk.get("title", "Unknown")
        movie_id = chunk.get("movie_id", "")
        description = chunk.get("description", "")
        situation = chunk.get("situation", "")
        scene_label = chunk.get("scene_label", "")
        characters = chunk.get("characters", [])
        dialogue = chunk.get("dialogue_text", "")[:300]
        attributes = chunk.get("attributes", [])

        chars_str = ", ".join(str(c) for c in characters[:5]) if characters else "Unknown"

        return f"""Analyze this movie scene and provide enriched metadata.

Movie: {title} ({movie_id})
Scene label: {scene_label}
Situation: {situation}

Description: {description}
Dialogue: {dialogue}

Characters in scene: {chars_str}
Attributes: {", ".join(str(a) for a in attributes[:10]) if attributes else "None"}

Return JSON:
{{
  "situation": "what is happening in 2-3 words",
  "vision_setting": "location type (e.g. indoor_kitchen, outdoor_beach, underwater)",
  "vision_actions": ["action1", "action2", "action3"],
  "emotional_tone": "positive_happy|negative_sad|neutral|tense|positive_funny|positive_romantic|negative_angry|negative_fearful",
  "narrative_arc": "introduction|rising_action|climax|resolution|transition|exposition",
  "character_emotions": {{"CharacterName": "emotion", ...}},
  "screenplay_context": "1-2 sentence summary of this scene's dramatic purpose"
}}"""


# ── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Simple token bucket rate limiter."""
    def __init__(self, max_per_min: int):
        self.max_per_min = max_per_min
        self.interval = 60.0 / max_per_min
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


# ── Enrichment ───────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(enriched_ids: set):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"enriched": list(enriched_ids)}, f)


def enrich_chunk(client: GroqVLMClient, chunk: dict, rate_limiter: RateLimiter) -> tuple[dict, bool]:
    """Enrich a single chunk. Returns (enriched_chunk, success)."""
    chunk_id = chunk.get("chunk_id", "")
    rate_limiter.wait()

    enriched = client.analyze_scene(chunk)

    if enriched:
        for k, v in enriched.items():
            if v and k in chunk:
                # Only fill empty or placeholder fields
                current = chunk.get(k, "")
                if not current or current in ("moment_description", "unspecified", "scene", "interacting"):
                    chunk[k] = v
        # Also update known empty fields
        for k in ["character_emotions", "narrative_arc", "vision_setting",
                   "emotional_tone", "screenplay_context"]:
            if k in enriched and enriched[k]:
                chunk[k] = enriched[k]
        return chunk, True
    return chunk, False


def enrich_all(dry_run: bool = True, batch_limit: int = 0) -> None:
    """Main enrichment pipeline."""
    print("\n" + "=" * 70)
    print("  🔮 VideoRag VLM Enrichment (Groq Llama 4 Scout)")
    print("=" * 70)

    # Load chunks
    with open(CHUNKS_IN, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", data if isinstance(data, list) else [])
    logger.info(f"Loaded {len(chunks)} chunks")

    if dry_run:
        # Test on first batch_limit chunks
        test_chunks = chunks[: batch_limit or 3]
        logger.info(f"DRY RUN: testing on {len(test_chunks)} chunks")

        try:
            client = GroqVLMClient()
            rate_limiter = RateLimiter(client.rate_limit_per_min)

            for chunk in test_chunks:
                result = client.analyze_scene(chunk)
                print(f"\n  Chunk: {chunk.get('chunk_id')}")
                print(f"  Movie: {chunk.get('title')}")
                print(f"  Original: {chunk.get('situation', 'N/A')}")
                print(f"  Enriched: {json.dumps(result, indent=4)}")
        except Exception as e:
            logger.error(f"Groq client error: {e}")
        return

    # Load progress
    progress = load_progress()
    done_ids = set(progress.get("enriched", []))
    logger.info(f"Already enriched: {len(done_ids)} chunks")

    # Filter
    remaining = [c for c in chunks if c.get("chunk_id") not in done_ids]
    logger.info(f"Remaining to enrich: {len(remaining)}")

    if not remaining:
        logger.info("All chunks already enriched!")
        return

    # Enrich
    client = GroqVLMClient()
    rate_limiter = RateLimiter(client.rate_limit_per_min)

    print(f"\nEnriching {len(remaining)} chunks...")
    success = 0
    failed = 0
    checkpoint = 50

    for i, chunk in enumerate(remaining):
        _, ok = enrich_chunk(client, chunk, rate_limiter)
        if ok:
            success += 1
        else:
            failed += 1
        done_ids.add(chunk.get("chunk_id", ""))

        if (i + 1) % checkpoint == 0:
            save_progress(done_ids)
            elapsed = i + 1
            rate = elapsed / 60  # rough
            remaining_est = (len(remaining) - elapsed) / max(rate, 1)
            print(f"  Progress: {i+1}/{len(remaining)} ({int(100*(i+1)/len(remaining))}%) "
                  f"~{int(remaining_est)}s remaining | ✅{success} ❌{failed}")
            sys.stdout.flush()

    # Final save
    save_progress(done_ids)

    # Write enriched chunks
    if isinstance(data, dict):
        data["chunks"] = chunks
        data["metadata"]["enriched"] = True
    else:
        data = chunks

    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Also overwrite original
    with open(CHUNKS_IN, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Done: {success} enriched, {failed} failed")
    logger.info(f"Saved: {CHUNKS_OUT}")
    logger.info(f"Progress saved: {PROGRESS_FILE}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLM enrich VideoRag chunks with Groq")
    parser.add_argument("--execute", action="store_true", help="Actually enrich (dry run by default)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Dry run with N chunks (default: 3)")
    args = parser.parse_args()

    dry_run = not args.execute
    batch_limit = args.batch if not args.execute else 0

    enrich_all(dry_run=dry_run, batch_limit=batch_limit)


if __name__ == "__main__":
    main()
