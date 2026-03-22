#!/usr/bin/env python3
"""
Fix: Merge missing temporal_chunks into all_chunks.json, rebuild knowledge index,
     then re-run benchmark.

20 Tier-1 movies có full annotation temporal chunks (2,998 chunks) bị bỏ qua
khi merge — chỉ giữ lại 3-9 trailer keyframe chunks per movie.
"""

import json
import sys
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

TEMPORAL_DIR  = PROJECT / "data/pipeline_output/temporal_chunks"
ALL_CHUNKS    = PROJECT / "data/pipeline_output/videorag_chunks/all_chunks.json"
INDEX_DIR     = PROJECT / "data/pipeline_output/indexes"
KV_FAISS      = INDEX_DIR / "knowledge_videorag.faiss"
KV_META       = INDEX_DIR / "knowledge_videorag_metadata.json"
BENCHMARK     = PROJECT / "data/pipeline_output/benchmark/qa_benchmark.json"
BENCH_RESULTS = PROJECT / "data/pipeline_output/benchmark/benchmark_results.json"


# ─── Canonical make_text (same as merge_all_videorag_chunks.py) ─────────────
def make_text(c: dict) -> str:
    parts = []
    for f in ["description", "situation", "dialogue_text", "characters",
              "narrative_arc", "script_heading", "screenplay_context",
              "vision_setting", "vision_actions"]:
        v = c.get(f)
        if isinstance(v, list):
            v = " ".join(str(x) for x in v if x)
        if v and str(v).strip():
            parts.append(str(v).strip())
    return " | ".join(parts)


# ─── Normalize a temporal chunk to all_chunks schema ─────────────────────────
def normalize_temporal(c: dict) -> dict:
    chars = c.get("characters", [])
    if isinstance(chars, str):
        chars = [chars]

    return {
        "chunk_id":        c.get("chunk_id", ""),
        "movie_id":        c.get("movie_id", ""),
        "title":           c.get("title", ""),
        "start_seconds":   c.get("start_seconds", 0.0),
        "end_seconds":     c.get("end_seconds", 0.0),
        "duration":        c.get("duration_seconds", 0.0),
        "timestamp_source": c.get("timestamp_source", "annotation_frame"),
        "description":     c.get("description", ""),
        "situation":       c.get("situation", ""),
        "vision_setting":  c.get("scene_label", ""),   # best proxy
        "vision_actions":  "",
        "emotional_tone":  "",
        "dialogue_text":   c.get("dialogue_text", ""),
        "characters":      chars,
        "cast_in_scene":   c.get("cast_in_scene", []),
        "narrative_arc":   "",
        "causal_relations": "",
        "source":          "annotation_temporal",
        "keyframe_paths":  c.get("keyframe_paths", []),
    }


# ─── Step 1: Find movies with missing temporal chunks ─────────────────────────
def find_missing_movies(all_chunks: list) -> list:
    by_movie = {}
    for c in all_chunks:
        by_movie.setdefault(c["movie_id"], []).append(c)

    missing = []
    for tf in sorted(TEMPORAL_DIR.glob("*_chunks.json")):
        mid = tf.stem.replace("_chunks", "")
        if mid in ("all", "video", "my-videos "):
            continue
        with open(tf) as f:
            tc = json.load(f)
        n_temp = len(tc) if isinstance(tc, list) else 0
        n_all  = len(by_movie.get(mid, []))
        if n_temp - n_all > 10:
            missing.append((mid, tf, tc, n_temp, n_all))
            log.info(f"  {mid}: temporal={n_temp}, all_chunks={n_all}, missing={n_temp-n_all}")

    return missing


# ─── Step 2: Merge ───────────────────────────────────────────────────────────
def merge_chunks(all_chunks: list, missing_movies: list) -> list:
    # Collect existing chunk_ids
    existing_ids = {c["chunk_id"] for c in all_chunks}

    added = 0
    for mid, tf, tc, n_temp, n_all in missing_movies:
        tc_list = tc if isinstance(tc, list) else []
        for c in tc_list:
            cid = c.get("chunk_id", "")
            if cid in existing_ids:
                continue
            norm = normalize_temporal(c)
            all_chunks.append(norm)
            existing_ids.add(cid)
            added += 1

    log.info(f"Added {added} new chunks. Total: {len(all_chunks)}")
    return all_chunks


# ─── Step 3: Rebuild knowledge_videorag FAISS index ──────────────────────────
def rebuild_knowledge_index(all_chunks: list):
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    log.info(f"Encoding {len(all_chunks)} chunks with SentenceTransformer...")

    texts = [make_text(c) for c in all_chunks]
    meta  = []
    for c in all_chunks:
        chars = c.get("characters", [])
        if isinstance(chars, str):
            chars = [chars]
        meta.append({
            "chunk_id":        c.get("chunk_id", ""),
            "movie_id":        c.get("movie_id", ""),
            "title":           c.get("title", ""),
            "text":            make_text(c),
            # L2
            "description":     c.get("description", ""),
            "vision_setting":  c.get("vision_setting", "") or c.get("scene_label", ""),
            "vision_actions":  c.get("vision_actions", ""),
            "emotional_tone":  c.get("emotional_tone", ""),
            "situation":       c.get("situation", ""),
            # L3
            "dialogue_text":   c.get("dialogue_text", ""),
            "speaker":         c.get("speaker", ""),
            "audio_events":    c.get("audio_events", ""),
            # L4
            "characters":      chars,
            "cast_in_scene":   c.get("cast_in_scene", []),
            # L5
            "narrative_arc":   c.get("narrative_arc", ""),
            "causal_relations": c.get("causal_relations", ""),
            "screenplay_context": c.get("screenplay_context", ""),
            # Meta
            "source":          c.get("source", ""),
            "start_seconds":   c.get("start_seconds", 0),
            "end_seconds":     c.get("end_seconds", 0),
            "keyframe_paths":  c.get("keyframe_paths", []),
        })

    batch = 512
    embeddings = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        emb = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(emb)
        if (i // batch) % 10 == 0:
            log.info(f"  Encoded {min(i+batch, len(texts))}/{len(texts)}")

    embeddings = np.vstack(embeddings).astype(np.float32)
    dim = embeddings.shape[1]
    log.info(f"Building IndexFlatIP dim={dim}, n={len(embeddings)}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(KV_FAISS))
    with open(KV_META, "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    log.info(f"Saved {KV_FAISS} ({index.ntotal} vectors, dim={dim})")
    log.info(f"Saved {KV_META}")


# ─── Step 4: Run benchmark ────────────────────────────────────────────────────
def run_benchmark():
    import subprocess
    script = PROJECT / "scripts/build_qa_benchmark.py"
    if not script.exists():
        log.warning(f"Benchmark script not found: {script}")
        return
    log.info("Running benchmark --evaluate ...")
    result = subprocess.run(
        [sys.executable, str(script), "--evaluate"],
        capture_output=True, text=True, cwd=str(PROJECT)
    )
    if result.returncode == 0:
        log.info("Benchmark completed.")
        # Print summary from results file
        if BENCH_RESULTS.exists():
            with open(BENCH_RESULTS) as f:
                res = json.load(f)
            print("\n=== BENCHMARK RESULTS ===")
            for task_id, metrics in res.items():
                if isinstance(metrics, dict):
                    r1 = metrics.get("recall@1", metrics.get("r1", 0))
                    r5 = metrics.get("recall@5", metrics.get("r5", 0))
                    mrr = metrics.get("mrr", 0)
                    print(f"  {task_id}: R@1={r1:.1%}  R@5={r5:.1%}  MRR={mrr:.3f}")
    else:
        log.error(f"Benchmark failed:\n{result.stderr[-2000:]}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--merge",     action="store_true", help="Step 1+2: merge missing chunks")
    p.add_argument("--rebuild",   action="store_true", help="Step 3: rebuild knowledge FAISS")
    p.add_argument("--benchmark", action="store_true", help="Step 4: re-run benchmark")
    p.add_argument("--all",       action="store_true", help="Run all steps")
    args = p.parse_args()

    if args.all:
        args.merge = args.rebuild = args.benchmark = True

    if not any([args.merge, args.rebuild, args.benchmark]):
        p.print_help()
        return

    # Load current all_chunks.json
    log.info(f"Loading {ALL_CHUNKS} ...")
    with open(ALL_CHUNKS) as f:
        all_chunks = json.load(f)
    log.info(f"Current size: {len(all_chunks)} chunks")

    if args.merge:
        log.info("=== Step 1: Finding missing temporal chunks ===")
        missing = find_missing_movies(all_chunks)
        log.info(f"Found {len(missing)} movies with missing chunks")

        if missing:
            log.info("=== Step 2: Merging ===")
            all_chunks = merge_chunks(all_chunks, missing)

            # Backup + save
            backup = ALL_CHUNKS.with_suffix(".json.bak")
            import shutil
            shutil.copy(ALL_CHUNKS, backup)
            log.info(f"Backed up to {backup}")

            with open(ALL_CHUNKS, "w") as f:
                json.dump(all_chunks, f, ensure_ascii=False)
            log.info(f"Saved {ALL_CHUNKS} ({len(all_chunks)} chunks)")
        else:
            log.info("No missing chunks found — already complete.")

    if args.rebuild:
        log.info("=== Step 3: Rebuilding knowledge index ===")
        rebuild_knowledge_index(all_chunks)

    if args.benchmark:
        log.info("=== Step 4: Benchmark ===")
        run_benchmark()

    log.info("Done.")


if __name__ == "__main__":
    main()
