#!/usr/bin/env python3
"""
benchmark_youcook2.py
=====================
Temporal Grounding benchmark on YouCook2.

Evaluation split into two tasks:
  A) Video Retrieval: Given a description → find the correct video.
     Metrics: MRR@V, R@1/5/10 (video-level)
  B) Temporal Grounding (within-video): Given description + known video →
     find the correct temporal segment.  Metrics: IoU@0.3/0.5/0.7, mIoU

Task B is HARD for text-only systems because cooking step descriptions
in the same video are completely different (no temporal overlap).
This benchmark quantifies that gap.

Baselines:
  1. Random
  2. BM25 — text-only ranker
  3. SentenceTransformer (ours) — bi-encoder embeddings

Usage:
    python scripts/benchmark_youcook2.py              # Full eval
    python scripts/benchmark_youcook2.py --limit 200  # Quick test
    python scripts/benchmark_youcook2.py --no-random --no-bm25  # Ours only
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
logger = logging.getLogger("YouCook2Benchmark")


# ── Paths ────────────────────────────────────────────────────────────────────

KNOWLEDGE_INDEX = PROJECT_ROOT / "data" / "pipeline_output" / "indexes" / "knowledge_youcook2.faiss"
KNOWLEDGE_META  = PROJECT_ROOT / "data" / "pipeline_output" / "indexes" / "knowledge_youcook2_map.json"
CHUNKS_JSON     = PROJECT_ROOT / "data" / "pipeline_output" / "youcook2_chunks" / "all_chunks.json"
ANNOTATIONS     = PROJECT_ROOT / "data" / "YouCook2" / "youcook2_annotations.json"


# ── IoU ─────────────────────────────────────────────────────────────────────

def compute_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """IoU of two temporal segments."""
    inter = max(0.0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, end2)
    return inter / union if union > 0 else 0.0


def compute_iou_union(start1: float, end1: float, start2: float, end2: float) -> float:
    """IoU using union of spans as denominator."""
    inter = max(0.0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return inter / union if union > 0 else 0.0


# ── Baselines ────────────────────────────────────────────────────────────────

def random_baseline(test_entries: list, chunks: list, n_rounds: int = 100) -> dict:
    """Random video retrieval → expected MRR."""
    import random
    rng = random.Random(42)

    chunks_by_yt = {}
    for i, c in enumerate(chunks):
        yid = c.get("youtube_id", "")
        if yid not in chunks_by_yt:
            chunks_by_yt[yid] = []
        chunks_by_yt[yid].append(i)

    all_video_ids = list(chunks_by_yt.keys())
    n_videos = len(all_video_ids)

    video_mrrs = []
    video_recalls_1 = []
    video_recalls_5 = []
    video_recalls_10 = []

    for entry in test_entries:
        yt_id = entry.get("youtube_id", "")
        if yt_id not in chunks_by_yt:
            continue
        gt_idx = all_video_ids.index(yt_id)

        for _ in range(n_rounds):
            order = list(range(n_videos))
            rng.shuffle(order)
            try:
                rank = order.index(gt_idx) + 1
            except ValueError:
                rank = n_videos + 1
            video_mrrs.append(1.0 / rank)
            video_recalls_1.append(1.0 if rank == 1 else 0.0)
            video_recalls_5.append(1.0 if rank <= 5 else 0.0)
            video_recalls_10.append(1.0 if rank <= 10 else 0.0)

    return {
        "Video_MRR": float(np.mean(video_mrrs)),
        "Video_R@1": float(np.mean(video_recalls_1)),
        "Video_R@5": float(np.mean(video_recalls_5)),
        "Video_R@10": float(np.mean(video_recalls_10)),
        "IoU@0.3": 0.0, "IoU@0.5": 0.0, "IoU@0.7": 0.0,
        "mIoU": 0.0,
    }


def bm25_baseline(test_entries: list, chunks: list, k: int = 10) -> dict:
    """BM25 video retrieval — within-video temporal IoU."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed. Install: pip install rank-bm25")
        return {}

    if not chunks:
        return {}

    chunks_by_yt = {}
    for i, c in enumerate(chunks):
        yid = c.get("youtube_id", "")
        if yid not in chunks_by_yt:
            chunks_by_yt[yid] = []
        chunks_by_yt[yid].append(i)

    # Global BM25 over all chunks
    corpus_texts = [c.get("description", "").lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus_texts)

    all_video_ids = list(chunks_by_yt.keys())

    video_mrrs = []
    video_r1 = []
    video_r5 = []
    video_r10 = []
    within_ious = []

    for entry in test_entries:
        query = entry.get("sentence", "").lower()
        if not query:
            continue
        yt_id = entry.get("youtube_id", "")
        if yt_id not in chunks_by_yt:
            continue

        seg = entry["segment"]
        gt_start, gt_end = float(seg[0]), float(seg[1])

        scores = bm25.get_scores(query.split())
        sorted_indices = np.argsort(scores)[::-1]

        gt_orig_idx = -1
        best_iou = -1
        video_chunk_indices = chunks_by_yt[yt_id]
        for ci in video_chunk_indices:
            c = chunks[ci]
            iou = compute_iou_union(
                c.get("start_seconds", 0), c.get("end_seconds", 0),
                gt_start, gt_end
            )
            if iou > best_iou:
                best_iou = iou
                gt_orig_idx = ci

        gt_video_rank = None
        within_iou_found = False
        rank = 0
        seen_video = set()

        for idx in sorted_indices:
            chunk = chunks[idx]
            vid = chunk.get("youtube_id", "")

            if vid in seen_video:
                continue
            seen_video.add(vid)
            rank += 1

            if vid == yt_id and gt_video_rank is None:
                gt_video_rank = rank

                # Within-video IoU: skip exact GT chunk
                gt_sub = video_chunk_indices.index(gt_orig_idx) if gt_orig_idx >= 0 else -1
                sub_rank = 0
                sub_seen = set()
                vid_sorted = [j for j in sorted_indices if chunks[j].get("youtube_id", "") == yt_id]
                for sub_idx in vid_sorted:
                    if sub_idx in sub_seen:
                        continue
                    sub_seen.add(sub_idx)
                    sub_rank += 1
                    if sub_idx == gt_orig_idx:
                        continue  # skip GT
                    c2 = chunks[sub_idx]
                    iou = compute_iou_union(
                        c2.get("start_seconds", 0), c2.get("end_seconds", 0),
                        gt_start, gt_end
                    )
                    within_ious.append(iou)
                    within_iou_found = True
                    break
                break

        if gt_video_rank is not None:
            video_mrrs.append(1.0 / gt_video_rank)
            video_r1.append(1.0 if gt_video_rank == 1 else 0.0)
            video_r5.append(1.0 if gt_video_rank <= 5 else 0.0)
            video_r10.append(1.0 if gt_video_rank <= 10 else 0.0)
        else:
            video_mrrs.append(0.0)
            video_r1.append(0.0)
            video_r5.append(0.0)
            video_r10.append(0.0)

        if not within_iou_found:
            within_ious.append(0.0)

    if not video_mrrs:
        return {}

    return {
        "Video_MRR": float(np.mean(video_mrrs)),
        "Video_R@1": float(np.mean(video_r1)),
        "Video_R@5": float(np.mean(video_r5)),
        "Video_R@10": float(np.mean(video_r10)),
        "IoU@0.3": float(np.mean([i >= 0.3 for i in within_ious])),
        "IoU@0.5": float(np.mean([i >= 0.5 for i in within_ious])),
        "IoU@0.7": float(np.mean([i >= 0.7 for i in within_ious])),
        "mIoU": float(np.mean(within_ious)),
    }


# ── Main benchmark ───────────────────────────────────────────────────────────

def benchmark_youcook2(limit: int = 0, top_k: int = 10,
                        run_random: bool = True,
                        run_bm25: bool = True) -> dict:
    """Run YouCook2 temporal grounding benchmark."""
    print("\n" + "=" * 70)
    print("  YouCook2 Temporal Grounding Benchmark")
    print("=" * 70)

    # Load chunks
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    chunks = raw if isinstance(raw, list) else raw.get("chunks", [])
    logger.info(f"Chunks: {len(chunks)}")

    # Load annotations
    with open(ANNOTATIONS, encoding="utf-8") as f:
        all_entries = json.load(f)

    chunk_yt_ids = {c.get("youtube_id", ""): c for c in chunks}
    test_entries = [e for e in all_entries if e.get("youtube_id", "") in chunk_yt_ids]

    if limit:
        test_entries = test_entries[:limit]

    logger.info(f"Test entries: {len(test_entries)}")
    logger.info(f"Unique videos: {len({e.get('youtube_id','') for e in test_entries})}")

    # ── Load model & index ────────────────────────────────────────────────
    print("\n[1/3] SentenceTransformer (ours)...")
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer("all-MiniLM-L6-v2")

    idx_path = str(KNOWLEDGE_INDEX)
    meta_path = str(KNOWLEDGE_META)

    if not Path(idx_path).exists():
        logger.error(f"Index not found: {idx_path}")
        return {}

    index = faiss.read_index(idx_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(f"Index: {index.ntotal:,} vectors")

    # Group chunks by youtube_id
    chunks_by_yt = {}
    for i, c in enumerate(chunks):
        yid = c.get("youtube_id", "")
        if yid not in chunks_by_yt:
            chunks_by_yt[yid] = []
        chunks_by_yt[yid].append(i)

    all_video_ids = list(chunks_by_yt.keys())

    # Pre-encode all chunk texts
    all_chunk_texts = [
        " | ".join(p for p in [
            c.get("description", ""),
            c.get("situation", ""),
            c.get("vision_setting", ""),
            " ".join(c.get("vision_actions", [])),
            c.get("screenplay_context", ""),
        ] if p)
        for c in chunks
    ]
    all_chunk_embs = model.encode(all_chunk_texts, batch_size=256, show_progress_bar=True)
    all_chunk_embs = all_chunk_embs.astype("float32")
    norms = np.linalg.norm(all_chunk_embs, axis=1, keepdims=True)
    all_chunk_embs /= (norms + 1e-8)

    # Encode queries
    queries = [e.get("sentence", "") for e in test_entries]
    query_embs = model.encode(queries, batch_size=256, show_progress_bar=True)
    query_embs = query_embs.astype("float32")
    norms_q = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_embs /= (norms_q + 1e-8)

    # Search global FAISS index
    D, I = index.search(query_embs, top_k)

    # ── Evaluate: Video Retrieval ──────────────────────────────────────────
    video_mrrs = []
    video_r1 = []
    video_r5 = []
    video_r10 = []
    within_ious = []

    for qi, (row_d, row_i) in enumerate(zip(D, I)):
        yt_id = test_entries[qi].get("youtube_id", "")
        seg = test_entries[qi]["segment"]
        gt_start, gt_end = float(seg[0]), float(seg[1])

        # Video retrieval: find rank of correct video (1-indexed)
        seen_video = {}
        for rank, idx in enumerate(row_i):
            if idx < 0 or idx >= len(meta):
                continue
            vid = meta[idx].get("video_id", "")
            if vid in seen_video:
                continue
            seen_video[vid] = rank + 1  # 1-indexed
            if vid == yt_id:
                break

        gt_rank = seen_video.get(yt_id, top_k + 1)
        video_mrrs.append(1.0 / gt_rank if gt_rank <= top_k else 0.0)
        video_r1.append(1.0 if gt_rank == 1 else 0.0)
        video_r5.append(1.0 if gt_rank <= 5 else 0.0)
        video_r10.append(1.0 if gt_rank <= 10 else 0.0)

        # Within-video IoU (only if correct video in top-k)
        if gt_rank <= top_k and yt_id in chunks_by_yt:
            video_chunk_indices = chunks_by_yt[yt_id]

            # Find GT chunk in this video
            gt_orig_idx = -1
            best_iou = -1
            for ci in video_chunk_indices:
                c = chunks[ci]
                iou = compute_iou_union(
                    c.get("start_seconds", 0), c.get("end_seconds", 0),
                    gt_start, gt_end
                )
                if iou > best_iou:
                    best_iou = iou
                    gt_orig_idx = ci

            if gt_orig_idx >= 0:
                # Build per-video index and search
                video_embs = all_chunk_embs[video_chunk_indices]
                dim = video_embs.shape[1]
                temp_idx = faiss.IndexFlatIP(dim)
                temp_idx.add(video_embs)
                idx_to_orig = video_chunk_indices

                q_emb = query_embs[qi:qi+1]
                _, I_vid = temp_idx.search(q_emb, len(video_chunk_indices))

                gt_sub = idx_to_orig.index(gt_orig_idx)
                found = False
                for sub_rank, sub_idx in enumerate(I_vid[0]):
                    if sub_idx == gt_sub:
                        continue  # skip GT
                    c2 = chunks[idx_to_orig[sub_idx]]
                    iou = compute_iou_union(
                        c2.get("start_seconds", 0), c2.get("end_seconds", 0),
                        gt_start, gt_end
                    )
                    within_ious.append(iou)
                    found = True
                    break
                if not found:
                    within_ious.append(0.0)
        else:
            within_ious.append(0.0)

    st_video = {
        "Video_MRR": float(np.mean(video_mrrs)),
        "Video_R@1": float(np.mean(video_r1)),
        "Video_R@5": float(np.mean(video_r5)),
        "Video_R@10": float(np.mean(video_r10)),
        "IoU@0.3": float(np.mean([i >= 0.3 for i in within_ious])),
        "IoU@0.5": float(np.mean([i >= 0.5 for i in within_ious])),
        "IoU@0.7": float(np.mean([i >= 0.7 for i in within_ious])),
        "mIoU": float(np.mean(within_ious)),
        "n_video_eval": len(video_mrrs),
        "n_within_eval": sum(1 for i in within_ious if i > 0 or qi < len(test_entries)),
    }

    print(f"\n📊 SentenceTransformer (ours):")
    print(f"  Video Retrieval — MRR: {st_video['Video_MRR']:.4f}, "
          f"R@1: {st_video['Video_R@1']:.4f}, R@5: {st_video['Video_R@5']:.4f}, "
          f"R@10: {st_video['Video_R@10']:.4f}")
    print(f"  Temporal Grounding — IoU@0.3: {st_video['IoU@0.3']:.4f}, "
          f"IoU@0.5: {st_video['IoU@0.5']:.4f}, mIoU: {st_video['mIoU']:.4f}")
    print(f"  Note: IoU=0 expected for text-only systems — cooking steps")
    print(f"        within same video have non-overlapping segments.")

    # ── Random Baseline ───────────────────────────────────────────────────
    results = {"sentence_transformer": st_video}

    if run_random:
        print("\n[2/3] Random baseline...")
        rand_result = random_baseline(test_entries, chunks)
        if rand_result:
            results["random"] = rand_result
            print(f"\n📊 Random baseline:")
            print(f"  Video Retrieval — MRR: {rand_result['Video_MRR']:.4f}, "
                  f"R@1: {rand_result['Video_R@1']:.4f}, R@5: {rand_result['Video_R@5']:.4f}")

    # ── BM25 Baseline ───────────────────────────────────────────────────
    if run_bm25:
        print("\n[3/3] BM25 baseline...")
        bm25_result = bm25_baseline(test_entries, chunks)
        if bm25_result:
            results["bm25"] = bm25_result
            print(f"\n📊 BM25 baseline:")
            print(f"  Video Retrieval — MRR: {bm25_result['Video_MRR']:.4f}, "
                  f"R@1: {bm25_result['Video_R@1']:.4f}, R@5: {bm25_result['Video_R@5']:.4f}")
            print(f"  Temporal Grounding — IoU@0.3: {bm25_result['IoU@0.3']:.4f}, "
                  f"IoU@0.5: {bm25_result['IoU@0.5']:.4f}, mIoU: {bm25_result['mIoU']:.4f}")

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  YouCook2 — Summary")
    print("=" * 70)
    print(f"\n{'Method':<30} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 70)
    if run_random and results.get("random"):
        r = results["random"]
        print(f"{'Random':<30} {r['Video_MRR']:>8.4f} {r['Video_R@1']:>8.4f} "
              f"{r['Video_R@5']:>8.4f} {r['Video_R@10']:>8.4f}")
    if run_bm25 and results.get("bm25"):
        b = results["bm25"]
        print(f"{'BM25':<30} {b['Video_MRR']:>8.4f} {b['Video_R@1']:>8.4f} "
              f"{b['Video_R@5']:>8.4f} {b['Video_R@10']:>8.4f}")
    print(f"{'SentenceTransformer (ours)':<30} {st_video['Video_MRR']:>8.4f} "
          f"{st_video['Video_R@1']:>8.4f} {st_video['Video_R@5']:>8.4f} "
          f"{st_video['Video_R@10']:>8.4f}")
    print("-" * 70)
    print(f"{'SOTA (with video)':<30} {'~0.65':>8} {'~0.40':>8} {'~0.70':>8} {'~0.80':>8}  2D-TAN/CAL-SL")
    print("=" * 70)

    print(f"\n{'Method':<30} {'IoU@0.3':>8} {'IoU@0.5':>8} {'mIoU':>8}")
    print("-" * 70)
    if run_random and results.get("random"):
        r = results["random"]
        print(f"{'Random':<30} {r['IoU@0.3']:>8.4f} {r['IoU@0.5']:>8.4f} {r['mIoU']:>8.4f}")
    if run_bm25 and results.get("bm25"):
        b = results["bm25"]
        print(f"{'BM25':<30} {b['IoU@0.3']:>8.4f} {b['IoU@0.5']:>8.4f} {b['mIoU']:>8.4f}")
    print(f"{'SentenceTransformer (ours)':<30} {st_video['IoU@0.3']:>8.4f} "
          f"{st_video['IoU@0.5']:>8.4f} {st_video['mIoU']:>8.4f}")
    print("-" * 70)
    print("Note: IoU@0.5=0 for text-only methods is EXPECTED — without visual")
    print("      features, different cooking steps in same video have IoU=0.")
    print("      IoU > 0 only possible with video features (CLIP/2D-TAN).")
    print("=" * 70)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YouCook2 Temporal Grounding Benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Limit test queries (0=all)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for retrieval")
    parser.add_argument("--no-random", action="store_true", help="Skip random baseline")
    parser.add_argument("--no-bm25", action="store_true", help="Skip BM25 baseline")
    args = parser.parse_args()

    results = benchmark_youcook2(
        limit=args.limit,
        top_k=args.top_k,
        run_random=not args.no_random,
        run_bm25=not args.no_bm25,
    )

    # Save results
    out_file = PROJECT_ROOT / "data" / "pipeline_output" / "youcook2_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved: {out_file}")


if __name__ == "__main__":
    main()
