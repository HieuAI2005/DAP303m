#!/usr/bin/env python3
"""
benchmark_visual_retrieval.py
==============================
Evaluate the L0 Visual Index and L1 Fusion Index on image-based queries.

Query construction:
  - Sample keyframes from known movies as queries
  - Expected: top-1 result should come from the same movie (within-movie visual consistency)
  - Also measure cross-movie visual disambiguation power

Metrics:
  - Visual R@1, R@5, R@10: fraction of queries where same-movie chunk is at rank k
  - MRR: mean reciprocal rank
  - Cross-movie hit rate: fraction where top-1 comes from a different movie

Usage:
    python scripts/benchmark_visual_retrieval.py --evaluate
    python scripts/benchmark_visual_retrieval.py --evaluate --n 500
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

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
logger = logging.getLogger("VisualBenchmark")

INDEX_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "indexes"
random.seed(42)


def load_visual_index():
    import faiss
    idx = faiss.read_index(str(INDEX_DIR / "visual_index.faiss"))
    meta = json.loads((INDEX_DIR / "visual_index_metadata.json").read_text())
    return idx, meta


def load_clip_model():
    import torch
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model.eval()
    except Exception:
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, preprocess, device


def encode_image(img_path: Path, model, preprocess, device) -> np.ndarray:
    import torch
    from PIL import Image
    try:
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
        feat = feat.cpu().float().numpy()[0]
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat
    except Exception:
        return np.zeros(512, dtype="float32")


def run_visual_benchmark(n_queries: int = 500):
    import faiss

    logger.info("Loading visual index...")
    idx, meta = load_visual_index()
    logger.info(f"  Index: {idx.ntotal} vectors")

    # Group meta by movie_id
    movie_to_frames: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        mid = m.get("movie_id") or m.get("video_id", "")
        if mid not in movie_to_frames:
            movie_to_frames[mid] = []
        movie_to_frames[mid].append(i)

    # Only movies with enough frames for query + index
    eligible = {mid: frames for mid, frames in movie_to_frames.items()
                if len(frames) >= 3 and not mid.startswith("v_")}
    logger.info(f"  Eligible movies: {len(eligible)}")

    # Sample query frames — use one frame per movie, leave rest in index
    query_indices = []
    query_movie_ids = []
    sample_movies = random.sample(list(eligible.keys()), min(n_queries, len(eligible)))
    for mid in sample_movies:
        frames = eligible[mid]
        query_indices.append(frames[0])  # first frame as query
        query_movie_ids.append(mid)

    logger.info(f"  Query set: {len(query_indices)} queries")
    logger.info("  Extracting query vectors from index...")

    # Extract query vectors directly from FAISS
    all_vecs = np.zeros((idx.ntotal, 512), dtype="float32")
    idx.reconstruct_n(0, idx.ntotal, all_vecs)
    query_vecs = all_vecs[query_indices].astype("float32")

    # Search
    k = 20
    logger.info(f"  Searching top-{k}...")
    scores, indices = idx.search(query_vecs, k + 1)  # +1 because query itself is in index

    r1, r5, r10, mrr_vals = [], [], [], []
    cross_movie_hits = []

    for qi, q_idx in enumerate(query_indices):
        q_movie = query_movie_ids[qi]
        # Exclude the query frame itself
        top_k = [indices[qi][j] for j in range(k + 1) if indices[qi][j] != q_idx][:k]
        top_k_movies = [meta[i].get("movie_id") or meta[i].get("video_id", "") for i in top_k]

        # Find first hit from same movie
        hit_ranks = [j + 1 for j, mid in enumerate(top_k_movies) if mid == q_movie]
        rank = hit_ranks[0] if hit_ranks else 999

        r1.append(1 if rank <= 1 else 0)
        r5.append(1 if rank <= 5 else 0)
        r10.append(1 if rank <= 10 else 0)
        mrr_vals.append(1 / rank if rank <= k else 0)
        cross_movie_hits.append(1 if top_k_movies[0] != q_movie else 0)

    results = {
        "n_queries": len(query_indices),
        "index_size": idx.ntotal,
        "R@1": round(float(np.mean(r1)) * 100, 1),
        "R@5": round(float(np.mean(r5)) * 100, 1),
        "R@10": round(float(np.mean(r10)) * 100, 1),
        "MRR": round(float(np.mean(mrr_vals)), 4),
        "cross_movie_hit_rate": round(float(np.mean(cross_movie_hits)) * 100, 1),
    }

    print("\n" + "=" * 60)
    print(f"  Visual Index Benchmark (L0 CLIP ViT-B/32)")
    print(f"  Index: {idx.ntotal:,} vectors | Queries: {len(query_indices)}")
    print("=" * 60)
    print(f"  R@1  (same-movie frame at rank 1): {results['R@1']:>6.1f}%")
    print(f"  R@5  (same-movie frame at rank ≤5): {results['R@5']:>6.1f}%")
    print(f"  R@10 (same-movie frame at rank ≤10): {results['R@10']:>6.1f}%")
    print(f"  MRR:                                {results['MRR']:>7.4f}")
    print(f"  Cross-movie confusion (top-1 wrong): {results['cross_movie_hit_rate']:>5.1f}%")
    print("=" * 60)

    # Save results
    out_path = PROJECT_ROOT / "data" / "pipeline_output" / "benchmark" / "visual_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved → {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    if args.evaluate:
        run_visual_benchmark(n_queries=args.n)


if __name__ == "__main__":
    main()
