#!/usr/bin/env python3
"""
enrich_vlm_trailers.py
======================
Run VLM (Groq Llama-4-Scout) on trailer keyframes to enrich L2 fields:
  description, vision_setting, vision_actions, emotional_tone

Uses 3 GROQ API keys in rotation to stay within rate limits.

Usage:
    python scripts/enrich_vlm_trailers.py --check
    python scripts/enrich_vlm_trailers.py --run
    python scripts/enrich_vlm_trailers.py --run --limit 100
    python scripts/enrich_vlm_trailers.py --movie-id tt0111161
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
from itertools import cycle
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
logger = logging.getLogger("VLMTrailers")

OUTPUT_BASE = PROJECT_ROOT / "data" / "pipeline_output" / "trailers"
CHUNKS_BASE = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "trailer_chunks"
TITLES_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "imdb_titles_cache.json"


def get_api_keys() -> list[str]:
    keys = []
    for k in ["GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2"]:
        v = os.environ.get(k, "")
        if v and v not in keys:
            keys.append(v)
    return keys


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def vlm_analyze_batch(client, kf_paths: list[Path], title: str) -> list[dict]:
    content_blocks: list[dict] = []
    valid = []
    for kf in kf_paths:
        if kf.exists():
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(kf)}"}
            })
            valid.append(kf)

    if not valid:
        return []

    prompt = {
        "type": "text",
        "text": (
            f"Analyze these {len(valid)} keyframes from the movie trailer '{title}'.\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "scenes": [\n'
            "    {\n"
            '      "description": "1-2 sentence description of what is happening",\n'
            '      "setting": "location/environment (e.g. indoor office, outdoor forest)",\n'
            '      "actions": ["action1", "action2"],\n'
            '      "emotional_tone": "e.g. tense, joyful, melancholic, dramatic"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"Provide exactly {len(valid)} scene objects, one per frame shown."
        )
    }
    content_blocks.insert(0, prompt)

    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": content_blocks}],
            temperature=0.1,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE).strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            scenes = data.get("scenes", [])
            return [
                {
                    "description": s.get("description", ""),
                    "vision_setting": s.get("setting", ""),
                    "vision_actions": s.get("actions", []),
                    "emotional_tone": s.get("emotional_tone", ""),
                }
                for s in scenes[:len(valid)]
            ]
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e}")
    except Exception as e:
        logger.debug(f"VLM error: {e}")
    return [{"description": "", "vision_setting": "", "vision_actions": [], "emotional_tone": ""}
            for _ in valid]


def enrich_movie(imdb_id: str, title: str, client) -> dict:
    movie_dir = OUTPUT_BASE / imdb_id
    chunks_file = movie_dir / "chunks.json"
    kf_dir = movie_dir / "keyframes"

    if not chunks_file.exists():
        return {"imdb_id": imdb_id, "status": "no_chunks"}

    raw = json.loads(chunks_file.read_text())
    chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw

    # Skip if already enriched
    already = sum(1 for c in chunks if c.get("description") and c["description"] != f"Trailer scene {c.get('shot_id','')}")
    if already > len(chunks) * 0.5:
        return {"imdb_id": imdb_id, "status": "already_done", "enriched": already}

    all_kfs = sorted(kf_dir.glob("kf_*.jpg")) if kf_dir.exists() else []
    if not all_kfs:
        return {"imdb_id": imdb_id, "status": "no_keyframes"}

    enriched = 0
    batch_size = 4
    for i, chunk in enumerate(chunks):
        kf_idx = min(i * batch_size, len(all_kfs) - 1)
        batch_kfs = all_kfs[kf_idx:kf_idx + batch_size]
        if not batch_kfs:
            continue
        results = vlm_analyze_batch(client, batch_kfs[:1], title)
        if results and results[0].get("description"):
            chunk.update(results[0])
            enriched += 1
        time.sleep(0.1)  # micro-delay

    # Save
    out = raw if isinstance(raw, dict) else {"chunks": chunks}
    out["chunks"] = chunks
    with open(chunks_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return {"imdb_id": imdb_id, "status": "done", "enriched": enriched, "total": len(chunks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--movie-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_keys = get_api_keys()
    if not api_keys:
        logger.error("No GROQ API keys found")
        return

    logger.info(f"Using {len(api_keys)} GROQ API key(s)")

    # Find all processed trailers with chunks
    trailer_dirs = sorted([d for d in OUTPUT_BASE.glob("tt*/")
                          if (d / "chunks.json").exists() and (d / "keyframes").exists()])

    # Load titles
    titles = {}
    if TITLES_FILE.exists():
        titles = json.loads(TITLES_FILE.read_text())

    # Check enrichment status
    unenriched = []
    for td in trailer_dirs:
        raw = json.loads((td / "chunks.json").read_text())
        chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw
        done = sum(1 for c in chunks
                   if c.get("description") and
                   not c["description"].startswith("Trailer scene"))
        if done < len(chunks) * 0.5:
            unenriched.append(td.name)

    logger.info(f"Trailers with chunks: {len(trailer_dirs)}")
    logger.info(f"Need enrichment:      {len(unenriched)}")

    if args.check:
        return

    todo = [args.movie_id] if args.movie_id else unenriched
    if args.limit:
        todo = todo[:args.limit]

    logger.info(f"Enriching {len(todo)} trailers...")

    try:
        from groq import Groq
    except ImportError:
        logger.error("pip install groq")
        return

    # Rotate through API keys
    key_cycle = cycle(api_keys)
    clients = {k: Groq(api_key=k) for k in api_keys}

    done_count = fail_count = 0
    req_count = 0

    for i, mid in enumerate(todo):
        title = titles.get(mid, mid)
        key = next(key_cycle)
        client = clients[key]

        result = enrich_movie(mid, title, client)
        status = result["status"]

        if status in ("done", "already_done"):
            done_count += 1
            if status == "done":
                logger.info(f"  OK {mid} ({title[:30]}): {result.get('enriched',0)}/{result.get('total',0)} chunks")
        else:
            fail_count += 1
            logger.debug(f"  SKIP {mid}: {status}")

        req_count += 1
        # Rate limit: ~30 req/min per key, 3 keys = 90 req/min → sleep 0.7s
        if req_count % len(api_keys) == 0:
            time.sleep(2.0)

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(todo)} | OK:{done_count} FAIL:{fail_count}")

    logger.info(f"Done: {done_count} enriched, {fail_count} failed/skipped")

    # Rebuild master trailer chunks
    logger.info("Rebuilding all_trailer_chunks.json...")
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
