#!/usr/bin/env python3
"""
run_benchmarks.py
=================
Benchmark the VideoSceneRAG pipeline on standard benchmarks.

Implemented:
  1. DiDeMo Temporal Grounding: R@IoU@0.5, R@1, MRR
  2. MSR-VTT Text-to-Video Retrieval: R@1, R@5, R@10, MRR

Usage:
    python scripts/run_benchmarks.py                    # All benchmarks
    python scripts/run_benchmarks.py --didemo          # DiDeMo only
    python scripts/run_benchmarks.py --msrvtt          # MSR-VTT only
    python scripts/run_benchmarks.py --limit 100       # Quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Benchmarks")


# ── Paths ────────────────────────────────────────────────────────────────────

KNOWLEDGE_INDEX = PROJECT_ROOT / "data" / "pipeline_output" / "indexes" / "knowledge_unified.faiss"
KNOWLEDGE_META = PROJECT_ROOT / "data" / "pipeline_output" / "indexes" / "knowledge_unified_map.json"
VISUAL_INDEX = PROJECT_ROOT / "data" / "pipeline_output" / "indexes" / "videorag_visual.faiss"
VISUAL_META = PROJECT_ROOT / "data" / "pipeline_output" / "indexes" / "videorag_visual_map.json"

DIDEMO_CHUNKS = PROJECT_ROOT / "data" / "pipeline_output" / "didemo_chunks" / "all_chunks.json"
DIDEMO_ANNOTATIONS = PROJECT_ROOT / "data" / "didemo_repo" / "data"
MSR_VTT_CHUNKS = PROJECT_ROOT / "data" / "pipeline_output" / "msr_vtt_chunks" / "all_chunks.json"


# ── FAISS Search ─────────────────────────────────────────────────────────────

def load_index_and_meta(idx_path: Path, meta_path: Path):
    import faiss
    if not idx_path.exists():
        # Try non-unified
        idx_path = idx_path.parent / "knowledge_index.faiss"
        meta_path = meta_path.parent / "knowledge_index_map.json"

    index = faiss.read_index(str(idx_path))
    meta = []
    if meta_path.exists():
        with open(str(meta_path), encoding="utf-8") as f:
            meta = json.load(f)
    return index, meta


def search_knn(index, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index. Returns (distances, indices)."""
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    query_vec = query_vec.astype("float32")
    import faiss
    faiss.normalize_L2(query_vec)
    return index.search(query_vec, k)


# ── DiDeMo Temporal Grounding ─────────────────────────────────────────────────

def load_didemo_data():
    """Load DiDeMo ground-truth annotations."""
    from collections import defaultdict

    # Load chunks
    with open(DIDEMO_CHUNKS, encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = chunks_data if isinstance(chunks_data, list) else chunks_data.get("chunks", [])

    # Build video_id → list of (start, end, description) for grounding
    video_segments = defaultdict(list)
    for c in chunks:
        vid = c.get("video_id", "")
        desc = c.get("description", "")
        start = c.get("start_seconds", 0)
        end = c.get("end_seconds", 0)
        if vid and desc:
            video_segments[vid].append({
                "start": float(start),
                "end": float(end),
                "description": desc,
                "chunk_id": c.get("chunk_id", ""),
            })

    # Load test annotations
    test_data = []
    for split in ["train", "val", "test"]:
        p = DIDEMO_ANNOTATIONS / f"{split}_data.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                # Ground truth time (in 5-second chunks)
                times = item.get("times", [[0, 0]])
                gt_start = times[0][0] * 5
                gt_end = (times[0][1] + 1) * 5
                test_data.append({
                    "video_id": item.get("video", "").split("_")[1],  # Flickr ID
                    "description": item.get("description", ""),
                    "gt_start": gt_start,
                    "gt_end": gt_end,
                    "num_segments": item.get("num_segments", 1),
                })

    return video_segments, test_data


def compute_iou(start1, end1, start2, end2):
    """Compute IoU of two time intervals."""
    inter = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return inter / union if union > 0 else 0


def benchmark_didemo(limit: int = 0, top_k: int = 10):
    """Run DiDeMo temporal grounding benchmark."""
    print("\n" + "=" * 70)
    print("  🎯 DiDeMo Temporal Grounding Benchmark")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load DiDeMo-specific index (not unified, to avoid cross-dataset pollution)
    idx_p = PROJECT_ROOT / "data" / "pipeline_output" / "indexes" / "knowledge_didemo.faiss"
    meta_p = PROJECT_ROOT / "data" / "pipeline_output" / "indexes" / "knowledge_didemo_map.json"
    if idx_p.exists() and meta_p.exists():
        index = faiss.read_index(str(idx_p))
        with open(str(meta_p)) as f:
            meta = json.load(f)
        logger.info(f"DiDeMo index: {index.ntotal:,} vectors")
    else:
        # Fallback to unified
        try:
            index, meta = load_index_and_meta(KNOWLEDGE_INDEX, KNOWLEDGE_META)
        except:
            logger.error("No DiDeMo or unified index found!")
            return {}

    # Build chunk_id → temporal info from metadata
    # Actually, we need the chunks for temporal data
    with open(DIDEMO_CHUNKS, encoding="utf-8") as f:
        chunks_raw = json.load(f)
    chunks = chunks_raw if isinstance(chunks_raw, list) else chunks_raw.get("chunks", [])

    chunk_map = {c["chunk_id"]: c for c in chunks}

    logger.info(f"Index: {index.ntotal:,} vectors")
    logger.info(f"Chunks: {len(chunks):,}")

    # Load test data
    _, test_data = load_didemo_data()
    if limit:
        test_data = test_data[:limit]

    logger.info(f"Test queries: {len(test_data):,}")

    # Encode queries
    queries = [t["description"] for t in test_data]
    logger.info("Encoding queries...")
    query_embs = model.encode(queries, batch_size=256, show_progress_bar=True)
    query_embs = query_embs.astype("float32")
    norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_embs /= (norms + 1e-8)

    # Search
    logger.info("Searching...")
    D, I = index.search(query_embs, top_k)

    # Evaluate
    ious_at_1 = []
    ious_at_5 = []
    mrr_scores = []

    rank_at_iou05 = []
    rank_at_iou07 = []

    for qi, (row_d, row_i) in enumerate(zip(D, I)):
        gt_start = test_data[qi]["gt_start"]
        gt_end = test_data[qi]["gt_end"]

        found_rank = None
        rr = 0.0

        for rk, (dist, idx) in enumerate(zip(row_d, row_i)):
            if idx < 0 or idx >= len(meta):
                continue

            chunk_meta = meta[idx]
            chunk_id = chunk_meta.get("chunk_id", "")
            if chunk_id not in chunk_map:
                continue

            chunk = chunk_map[chunk_id]
            start = chunk.get("start_seconds", 0)
            end = chunk.get("end_seconds", 0)

            iou = compute_iou(gt_start, gt_end, start, end)

            if found_rank is None:
                ious_at_1.append(iou)
                rr = 1.0 / (rk + 1)

            ious_at_5.append(iou)
            if found_rank is None and iou >= 0.5:
                found_rank = rk + 1
            if iou >= 0.5:
                rank_at_iou05.append(rk + 1)
            if iou >= 0.7:
                rank_at_iou07.append(rk + 1)

            if found_rank is not None:
                break

        mrr_scores.append(rr)

    # Compute metrics
    iou_at_1 = np.mean(ious_at_1) if ious_at_1 else 0
    iou_at_5 = np.mean(ious_at_5) if ious_at_5 else 0
    mrr = np.mean(mrr_scores) if mrr_scores else 0
    rank1_at_iou05 = sum(1 for c in rank_at_iou05 if c == 1) / len(rank_at_iou05) if rank_at_iou05 else 0
    recall_at_5 = sum(1 for c in rank_at_iou05) / len(test_data) if rank_at_iou05 else 0

    print(f"\n📊 Results (n={len(test_data):,}):")
    print(f"  IoU@1:    {iou_at_1:.4f}")
    print(f"  IoU@5:    {iou_at_5:.4f}")
    print(f"  MRR:      {mrr:.4f}")
    print(f"  R@IoU≥0.5@5: {recall_at_5:.4f}")
    print(f"  R@1@IoU≥0.5: {rank1_at_iou05:.4f}")

    return {
        "iou_at_1": float(iou_at_1),
        "iou_at_5": float(iou_at_5),
        "mrr": float(mrr),
        "recall_at_5": float(recall_at_5),
        "rank1_at_iou05": float(rank1_at_iou05),
        "num_queries": len(test_data),
    }


# ── MSR-VTT Text-to-Video Retrieval ─────────────────────────────────────────

def benchmark_msrvtt(limit: int = 0, top_k: int = 10):
    """Run MSR-VTT text-to-video retrieval benchmark."""
    print("\n" + "=" * 70)
    print("  🎯 MSR-VTT Text-to-Video Retrieval Benchmark")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load MSR-VTT chunks
    with open(MSR_VTT_CHUNKS, encoding="utf-8") as f:
        chunks_raw = json.load(f)
    chunks = chunks_raw if isinstance(chunks_raw, list) else chunks_raw.get("chunks", [])

    logger.info(f"MSR-VTT chunks: {len(chunks):,}")

    # Group by video_id
    from collections import defaultdict
    video_chunks = defaultdict(list)
    for c in chunks:
        vid = c.get("video_id", "")
        if vid:
            video_chunks[vid].append(c)

    video_ids = list(video_chunks.keys())
    logger.info(f"Unique videos: {len(video_ids):,}")

    # Build text representation for each video (concatenate captions)
    video_texts = {}
    for vid, vchunks in video_chunks.items():
        texts = [c.get("description", "") or c.get("text", "") for c in vchunks]
        video_texts[vid] = " ".join(texts[:10])  # limit context

    if limit:
        video_ids = video_ids[:limit]
        video_texts = {k: video_texts[k] for k in video_ids}

    # Encode video texts
    logger.info("Encoding video texts...")
    vid_list = list(video_texts.keys())
    texts_list = [video_texts[v] for v in vid_list]
    vid_embs = model.encode(texts_list, batch_size=256, show_progress_bar=True)
    vid_embs = vid_embs.astype("float32")
    norms = np.linalg.norm(vid_embs, axis=1, keepdims=True)
    vid_embs /= (norms + 1e-8)

    # Build a temporary FAISS index for video texts
    dim = vid_embs.shape[1]
    temp_index = faiss.IndexFlatIP(dim)
    temp_index.add(vid_embs)

    # For each query, search and evaluate
    # Use the MSR-VTT test set captions as queries
    # Build ground truth: video_id matches
    test_queries = []
    for vid in vid_list:
        for c in video_chunks[vid]:
            test_queries.append({
                "query": c.get("description", "") or c.get("text", ""),
                "video_id": vid,
            })

    if limit:
        test_queries = test_queries[:limit]

    logger.info(f"Test queries: {len(test_queries):,}")

    # Encode queries
    query_texts = [t["query"] for t in test_queries]
    logger.info("Encoding queries...")
    query_embs = model.encode(query_texts, batch_size=256, show_progress_bar=True)
    query_embs = query_embs.astype("float32")
    norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_embs /= (norms + 1e-8)

    # Search
    logger.info("Searching...")
    D, I = temp_index.search(query_embs, min(top_k, len(vid_list)))

    # Evaluate
    r_at_1 = 0
    r_at_5 = 0
    r_at_10 = 0
    mrr_sum = 0.0

    for qi, (row_d, row_i) in enumerate(zip(D, I)):
        gt_vid = test_queries[qi]["video_id"]

        for rk, idx in enumerate(row_i):
            if idx < 0:
                break
            retrieved_vid = vid_list[idx]
            if retrieved_vid == gt_vid:
                if rk < 1:
                    r_at_1 += 1
                if rk < 5:
                    r_at_5 += 1
                if rk < 10:
                    r_at_10 += 1
                mrr_sum += 1.0 / (rk + 1)
                break

    n = len(test_queries)
    r1 = r_at_1 / n
    r5 = r_at_5 / n
    r10 = r_at_10 / n
    mrr = mrr_sum / n

    print(f"\n📊 Results (n={n:,}):")
    print(f"  R@1:   {r1:.4f}")
    print(f"  R@5:   {r5:.4f}")
    print(f"  R@10:  {r10:.4f}")
    print(f"  MRR:   {mrr:.4f}")

    return {
        "r_at_1": float(r1),
        "r_at_5": float(r5),
        "r_at_10": float(r10),
        "mrr": float(mrr),
        "num_queries": n,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run VideoSceneRAG benchmarks")
    parser.add_argument("--didemo", action="store_true", help="DiDeMo only")
    parser.add_argument("--msrvtt", action="store_true", help="MSR-VTT only")
    parser.add_argument("--limit", type=int, default=0, help="Limit test queries (0=all)")
    args = parser.parse_args()

    results = {}

    if args.didemo and not args.msrvtt:
        results["didemo"] = benchmark_didemo(limit=args.limit)

    if args.msrvtt and not args.didemo:
        results["msrvtt"] = benchmark_msrvtt(limit=args.limit)

    if not args.didemo and not args.msrvtt:
        # Run both when no specific flag given
        results["didemo"] = benchmark_didemo(limit=args.limit)
        results["msrvtt"] = benchmark_msrvtt(limit=args.limit)

    # Save results
    out_file = PROJECT_ROOT / "data" / "pipeline_output" / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved: {out_file}")


if __name__ == "__main__":
    main()
