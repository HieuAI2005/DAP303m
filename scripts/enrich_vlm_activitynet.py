#!/usr/bin/env python3
"""
enrich_vlm_activitynet.py
==========================
Run VLM (Groq Llama-4-Scout) on ActivityNet keyframes to enrich L2 fields:
  description, vision_setting, vision_actions, emotional_tone

Usage:
    python scripts/enrich_vlm_activitynet.py --check
    python scripts/enrich_vlm_activitynet.py --run
    python scripts/enrich_vlm_activitynet.py --run --limit 100
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
logger = logging.getLogger("VLMActivityNet")

OUTPUT_BASE = PROJECT_ROOT / "data" / "pipeline_output" / "activitynet"
CHUNKS_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "activitynet_chunks"


def get_api_keys() -> list[str]:
    keys = []
    for k in ["GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2"]:
        v = os.environ.get(k, "")
        if v and v not in keys:
            keys.append(v)
    return keys


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def vlm_analyze_frame(client, kf_path: Path) -> dict:
    if not kf_path.exists():
        return {}
    content_blocks = [
        {
            "type": "text",
            "text": (
                "Analyze this video frame from an ActivityNet activity video.\n"
                "Return ONLY valid JSON:\n"
                "{\n"
                '  "description": "1-2 sentence description of what is happening",\n'
                '  "setting": "location/environment (e.g. indoor gym, outdoor park)",\n'
                '  "actions": ["action1", "action2"],\n'
                '  "emotional_tone": "e.g. energetic, calm, competitive"\n'
                "}\n"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(kf_path)}"},
        },
    ]
    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": content_blocks}],
            temperature=0.1,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE).strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                "description": data.get("description", ""),
                "vision_setting": data.get("setting", ""),
                "vision_actions": data.get("actions", []),
                "emotional_tone": data.get("emotional_tone", ""),
            }
    except json.JSONDecodeError:
        pass
    except Exception as e:
        logger.debug(f"VLM error: {e}")
    return {}


def enrich_video(video_id: str, client) -> dict:
    video_dir = OUTPUT_BASE / video_id
    chunks_file = video_dir / "chunks.json"
    kf_dir = video_dir / "keyframes"

    if not chunks_file.exists():
        return {"video_id": video_id, "status": "no_chunks"}

    raw = json.loads(chunks_file.read_text())
    chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw

    # Skip if already enriched
    already = sum(1 for c in chunks
                  if c.get("description") and not c["description"].startswith("ActivityNet clip"))
    if already > len(chunks) * 0.5:
        return {"video_id": video_id, "status": "already_done", "enriched": already}

    all_kfs = sorted(kf_dir.glob("kf_*.jpg")) if kf_dir.exists() else []
    if not all_kfs:
        return {"video_id": video_id, "status": "no_keyframes"}

    enriched = 0
    for i, chunk in enumerate(chunks):
        # Pick representative keyframe for this chunk
        kf_idx = min(i, len(all_kfs) - 1)
        result = vlm_analyze_frame(client, all_kfs[kf_idx])
        if result.get("description"):
            chunk.update(result)
            enriched += 1
        time.sleep(0.05)

    out = raw if isinstance(raw, dict) else {"chunks": chunks}
    out["chunks"] = chunks
    with open(chunks_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return {"video_id": video_id, "status": "done", "enriched": enriched, "total": len(chunks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_keys = get_api_keys()
    if not api_keys:
        logger.error("No GROQ API keys found")
        return

    logger.info(f"Using {len(api_keys)} GROQ API key(s)")

    video_dirs = sorted([d for d in OUTPUT_BASE.glob("v_*/")
                         if (d / "chunks.json").exists() and (d / "keyframes").exists()])

    unenriched = []
    for vd in video_dirs:
        raw = json.loads((vd / "chunks.json").read_text())
        chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw
        done = sum(1 for c in chunks
                   if c.get("description") and not c["description"].startswith("ActivityNet clip"))
        if done < len(chunks) * 0.5:
            unenriched.append(vd.name)

    logger.info(f"ActivityNet videos with chunks: {len(video_dirs)}")
    logger.info(f"Need enrichment:               {len(unenriched)}")

    if args.check:
        return

    if not args.run:
        return

    todo = unenriched[:args.limit] if args.limit else unenriched
    logger.info(f"Enriching {len(todo)} videos...")

    try:
        from groq import Groq
    except ImportError:
        logger.error("pip install groq")
        return

    key_cycle = cycle(api_keys)
    clients = {k: Groq(api_key=k) for k in api_keys}

    done_count = fail_count = req_count = 0

    for i, vid in enumerate(todo):
        key = next(key_cycle)
        client = clients[key]

        result = enrich_video(vid, client)
        status = result["status"]

        if status in ("done", "already_done"):
            done_count += 1
            if status == "done":
                logger.info(f"  OK {vid}: {result.get('enriched',0)}/{result.get('total',0)} chunks")
        else:
            fail_count += 1
            logger.debug(f"  SKIP {vid}: {status}")

        req_count += 1
        if req_count % len(api_keys) == 0:
            time.sleep(2.0)

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(todo)} | OK:{done_count} FAIL:{fail_count}")

    logger.info(f"Done: {done_count} enriched, {fail_count} failed/skipped")

    # Rebuild master activitynet chunks
    logger.info("Rebuilding all_activitynet_chunks.json...")
    all_chunks = []
    for vd in sorted(OUTPUT_BASE.glob("v_*/")):
        cf = vd / "chunks.json"
        if cf.exists():
            raw = json.loads(cf.read_text())
            chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw
            all_chunks.extend(chunks)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_DIR / "all_activitynet_chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(all_chunks)} ActivityNet chunks")


if __name__ == "__main__":
    main()
