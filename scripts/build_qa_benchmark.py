#!/usr/bin/env python3
"""
build_qa_benchmark.py
=====================
Build end-to-end QA benchmark from existing annotated chunks.
Generates query-answer pairs across 4 task types:
  T1: Scene description retrieval (L2 → L1 temporal)
  T2: Character name retrieval   (L4 → movie_id)
  T3: Dialogue quote search      (L3 → scene timestamp)
  T4: Cross-movie visual concept (L0 → top-k movies)

Usage:
    python scripts/build_qa_benchmark.py --build
    python scripts/build_qa_benchmark.py --evaluate
    python scripts/build_qa_benchmark.py --build --evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("QABenchmark")

CHUNKS_FILE = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "all_chunks.json"
INDEX_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "indexes"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "benchmark"

random.seed(42)


# ── Query builders ────────────────────────────────────────────────────────────

def build_t1_description_queries(chunks: list[dict], n: int = 200) -> list[dict]:
    """T1: Query = description text → expected chunk_id + timestamp"""
    pool = [c for c in chunks
            if c.get("description") and
            c["description"] not in ("", "[NO_DIALOGUE]") and
            not c["description"].startswith("Trailer scene") and
            not c["description"].startswith("ActivityNet clip") and
            len(c["description"]) > 30]
    sample = random.sample(pool, min(n, len(pool)))
    queries = []
    for c in sample:
        queries.append({
            "query_id": f"T1_{c['chunk_id']}",
            "task": "scene_description",
            "query_text": c["description"],
            "expected_chunk_id": c["chunk_id"],
            "expected_movie_id": c.get("movie_id", ""),
            "expected_start_seconds": c.get("start_seconds", 0),
            "expected_end_seconds": c.get("end_seconds", 0),
            "metadata": {"title": c.get("title", ""), "source": c.get("source", "")}
        })
    return queries


def build_t2_character_queries(chunks: list[dict], n: int = 300) -> list[dict]:
    """T2: Query = character name → expected movie_id"""
    char_to_movies: dict[str, set] = defaultdict(set)
    char_to_chunks: dict[str, list] = defaultdict(list)

    for c in chunks:
        for char in c.get("characters", []):
            if char and len(char) > 2:
                char_to_movies[char].add(c["movie_id"])
                char_to_chunks[char].append(c["chunk_id"])

    # Only characters appearing in exactly 1 movie (unambiguous)
    unambiguous = {ch: list(movies)[0]
                   for ch, movies in char_to_movies.items()
                   if len(movies) == 1 and len(ch.split()) >= 2}

    sample_chars = random.sample(list(unambiguous.keys()), min(n, len(unambiguous)))
    queries = []
    for char in sample_chars:
        mid = unambiguous[char]
        chunk_ids = char_to_chunks[char]
        queries.append({
            "query_id": f"T2_{char.replace(' ', '_')}",
            "task": "character_retrieval",
            "query_text": char,
            "expected_movie_id": mid,
            "expected_chunk_ids": chunk_ids[:5],
            "metadata": {"character": char}
        })
    return queries


def build_t3_dialogue_queries(chunks: list[dict], n: int = 200) -> list[dict]:
    """T3: Query = dialogue quote → expected chunk_id + timestamp"""
    pool = [c for c in chunks
            if c.get("dialogue_text") and
            c["dialogue_text"] not in ("", "[NO_DIALOGUE]") and
            len(c["dialogue_text"]) > 20]
    sample = random.sample(pool, min(n, len(pool)))
    queries = []
    for c in sample:
        # Use first sentence as query
        dial = c["dialogue_text"]
        sentence = dial.split(".")[0].split("?")[0].split("!")[0].strip()
        if len(sentence) < 10:
            sentence = dial[:80]
        queries.append({
            "query_id": f"T3_{c['chunk_id']}",
            "task": "dialogue_grounding",
            "query_text": sentence,
            "expected_chunk_id": c["chunk_id"],
            "expected_movie_id": c.get("movie_id", ""),
            "expected_start_seconds": c.get("start_seconds", 0),
            "metadata": {"speaker": c.get("speaker", ""), "title": c.get("title", "")}
        })
    return queries


def build_t4_setting_queries(chunks: list[dict], n: int = 100) -> list[dict]:
    """T4: Query = visual setting → top movie IDs with that setting"""
    setting_to_movies: dict[str, list] = defaultdict(list)
    for c in chunks:
        s = c.get("vision_setting", "").strip().lower()
        if s and len(s) > 5 and s != "unknown":
            setting_to_movies[s].append(c["movie_id"])

    # Settings appearing in multiple movies (interesting cross-movie queries)
    multi_movie = {s: list(set(mids)) for s, mids in setting_to_movies.items()
                   if len(set(mids)) >= 2}

    sample_settings = random.sample(list(multi_movie.keys()), min(n, len(multi_movie)))
    queries = []
    for setting in sample_settings:
        movie_ids = multi_movie[setting]
        queries.append({
            "query_id": f"T4_{setting[:30].replace(' ', '_')}",
            "task": "visual_setting_search",
            "query_text": f"scene in {setting}",
            "expected_movie_ids": movie_ids[:10],
            "metadata": {"setting": setting, "n_movies": len(movie_ids)}
        })
    return queries


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_benchmark(queries: list[dict]) -> dict:
    import numpy as np
    import faiss

    # Load knowledge index
    idx_path = INDEX_DIR / "knowledge_videorag.faiss"
    meta_path = INDEX_DIR / "knowledge_videorag_meta.json"
    if not idx_path.exists():
        logger.error("knowledge_videorag.faiss not found")
        return {}

    index = faiss.read_index(str(idx_path))
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else []

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results_by_task: dict[str, dict] = {}

    for task_name in ["scene_description", "character_retrieval", "dialogue_grounding", "visual_setting_search"]:
        task_queries = [q for q in queries if q["task"] == task_name]
        if not task_queries:
            continue

        texts = [q["query_text"] for q in task_queries]
        embeddings = model.encode(texts, batch_size=256, normalize_embeddings=True).astype(np.float32)

        k = 20
        scores, indices = index.search(embeddings, k)

        r1, r5, r10, mrr_vals = [], [], [], []

        for qi, q in enumerate(task_queries):
            top_k_meta = [meta[idx] for idx in indices[qi] if idx < len(meta)]
            top_k_movies = [m["movie_id"] for m in top_k_meta]
            top_k_chunks = [m["chunk_id"] for m in top_k_meta]

            if task_name in ("scene_description", "dialogue_grounding"):
                expected_chunk = q.get("expected_chunk_id", "")
                hit_ranks = [j + 1 for j, cid in enumerate(top_k_chunks) if cid == expected_chunk]
                rank = hit_ranks[0] if hit_ranks else 999
            else:
                expected_movie = q.get("expected_movie_id", "")
                expected_movies = q.get("expected_movie_ids", [expected_movie])
                hit_ranks = [j + 1 for j, mid in enumerate(top_k_movies) if mid in expected_movies]
                rank = hit_ranks[0] if hit_ranks else 999

            r1.append(1 if rank <= 1 else 0)
            r5.append(1 if rank <= 5 else 0)
            r10.append(1 if rank <= 10 else 0)
            mrr_vals.append(1 / rank if rank <= k else 0)

        results_by_task[task_name] = {
            "n_queries": len(task_queries),
            "R@1": round(float(np.mean(r1)) * 100, 1),
            "R@5": round(float(np.mean(r5)) * 100, 1),
            "R@10": round(float(np.mean(r10)) * 100, 1),
            "MRR": round(float(np.mean(mrr_vals)), 4),
        }

    return results_by_task


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    chunks_path = BENCHMARK_DIR / "qa_benchmark.json"

    if args.build:
        logger.info("Loading chunks...")
        all_chunks = json.loads(CHUNKS_FILE.read_text())
        logger.info(f"  Total chunks: {len(all_chunks)}")

        logger.info("Building T1 (scene description) queries...")
        t1 = build_t1_description_queries(all_chunks, n=250)
        logger.info(f"  T1: {len(t1)} queries")

        logger.info("Building T2 (character retrieval) queries...")
        t2 = build_t2_character_queries(all_chunks, n=400)
        logger.info(f"  T2: {len(t2)} queries")

        logger.info("Building T3 (dialogue grounding) queries...")
        t3 = build_t3_dialogue_queries(all_chunks, n=250)
        logger.info(f"  T3: {len(t3)} queries")

        logger.info("Building T4 (visual setting) queries...")
        t4 = build_t4_setting_queries(all_chunks, n=150)
        logger.info(f"  T4: {len(t4)} queries")

        all_queries = t1 + t2 + t3 + t4
        with open(chunks_path, "w") as f:
            json.dump({"queries": all_queries, "total": len(all_queries)}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(all_queries)} queries → {chunks_path}")

    if args.evaluate:
        if not chunks_path.exists():
            logger.error("Run --build first")
            return

        queries = json.loads(chunks_path.read_text())["queries"]
        logger.info(f"Evaluating {len(queries)} queries on L3 Knowledge Index...")

        results = evaluate_benchmark(queries)

        # Print results table
        print("\n" + "="*60)
        print(f"{'Task':<28} {'N':>5} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>7}")
        print("-"*60)
        for task, r in results.items():
            print(f"{task:<28} {r['n_queries']:>5} {r['R@1']:>5.1f}% {r['R@5']:>5.1f}% {r['R@10']:>5.1f}% {r['MRR']:>7.4f}")
        print("="*60)

        # Save results
        results_path = BENCHMARK_DIR / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump({"results": results, "n_total_queries": len(queries)}, f, indent=2)
        logger.info(f"Saved results → {results_path}")


if __name__ == "__main__":
    main()
