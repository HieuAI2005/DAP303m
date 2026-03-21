#!/usr/bin/env python3
"""
build_unified_indexes.py
========================
Merge all FAISS indexes into unified indexes and verify all data.

Datasets:
  - VideoRag:   3,229 chunks  | visual: 128,410 vectors | knowledge: see below
  - ActivityNet: 23,064 chunks | knowledge: 23,064 vectors
  - DiDeMo:    165,216 chunks | knowledge: 165,216 vectors
  - MSR-VTT:    10,000 chunks | knowledge: 10,000 vectors

Usage:
    python scripts/build_unified_indexes.py --check-only
    python scripts/build_unified_indexes.py --build
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("UnifiedIndexer")


# ── Paths ────────────────────────────────────────────────────────────────────

OUTPUT_INDEXES = PROJECT_ROOT / "data" / "pipeline_output" / "indexes"
CHUNK_DIRS = {
    "videorag":    PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks",
    "activitynet": PROJECT_ROOT / "data" / "pipeline_output" / "activitynet_chunks",
    "youcook2":    PROJECT_ROOT / "data" / "pipeline_output" / "youcook2_chunks",
}


def load_chunks(chunk_dir: Path) -> list:
    """Load chunks from all_chunks.json."""
    path = chunk_dir / "all_chunks.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        chunks = data.get("chunks", [])
    elif isinstance(data, list):
        chunks = data
    else:
        chunks = []
    return chunks


def load_index_meta(index_path: Path, meta_path: Path) -> tuple:
    """Load FAISS index and metadata."""
    if not index_path.exists():
        return None, []
    index = faiss.read_index(str(index_path))
    meta = []
    if meta_path.exists():
        with open(str(meta_path), encoding="utf-8") as f:
            meta = json.load(f)
    return index, meta


def merge_indexes(index_specs: list) -> tuple:
    """
    Merge multiple FAISS indexes into one.
    Each spec: {"index": faiss_index, "meta": [...], "label": str}
    Returns merged_index, merged_meta
    """
    total = sum(s["index"].ntotal for s in index_specs)
    dim = index_specs[0]["index"].d
    merged = faiss.IndexFlatIP(dim)
    merged_meta = []

    for spec in index_specs:
        idx = spec["index"]
        meta = spec["meta"]
        vecs = faiss.vector_to_array(idx.get_direct_vectors())
        # reshape: ntotal x dim
        vecs = vecs.reshape(idx.ntotal, dim)
        merged.add(vecs)
        merged_meta.extend(meta)

    return merged, merged_meta


def check_all() -> None:
    """Check all datasets and indexes."""
    print("\n" + "=" * 70)
    print("  📊 Unified Data & Index Check")
    print("=" * 70)

    total_chunks = 0
    total_index_vectors = 0

    # Chunk stats
    print("\n📋 Chunk Datasets:")
    for name, chunk_dir in CHUNK_DIRS.items():
        chunks = load_chunks(chunk_dir)
        n = len(chunks)
        total_chunks += n
        if n == 0:
            print(f"  {name:15s}: ❌ not found")
            continue

        # Count real data
        has_desc = sum(1 for c in chunks if c.get("description"))
        has_chars = sum(1 for c in chunks if c.get("characters"))
        has_kf = sum(1 for c in chunks if c.get("keyframe_paths"))
        has_video = sum(1 for c in chunks if c.get("video_path") and Path(c.get("video_path", "")).exists())

        print(f"  {name:15s}: ✅ {n:>8,} chunks")
        print(f"    {'Descriptions':20s}: {has_desc:>8,} ({int(100*has_desc/n)}%)")
        print(f"    {'Characters':20s}: {has_chars:>8,} ({int(100*has_chars/n)}%)")
        print(f"    {'Keyframes':20s}: {has_kf:>8,} ({int(100*has_kf/n)}%)")
        print(f"    {'Video files':20s}: {has_video:>8,}")

        # Source breakdown
        sources = {}
        for c in chunks:
            src = c.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        print(f"    Sources: {sources}")

    print(f"\n  Total chunks: {total_chunks:,}")

    # Index stats
    print("\n🔢 FAISS Indexes:")
    index_files = sorted(OUTPUT_INDEXES.glob("*.faiss"))
    for idx_file in index_files:
        meta_file = idx_file.with_suffix(".faiss_map.json")
        if not meta_file.exists():
            meta_file = idx_file.with_name(idx_file.stem + "_map.json")

        idx, meta = load_index_meta(idx_file, meta_file)
        if idx is None:
            print(f"  {idx_file.name:40s}: ❌ load failed")
            continue

        size_mb = idx_file.stat().st_size // (1024 * 1024)
        is_link = idx_file.is_symlink()
        src = ""
        if is_link:
            src = f" → {idx_file.resolve().name}"
        print(f"  {idx_file.name:40s}: {idx.ntotal:>10,} vecs | {size_mb:>5}MB{src}")
        total_index_vectors += idx.ntotal

    print(f"\n  Total index vectors: {total_index_vectors:,}")

    # Visual index
    visual = OUTPUT_INDEXES / "videorag_visual.faiss"
    if visual.exists():
        idx, meta = load_index_meta(visual, OUTPUT_INDEXES / "videorag_visual_map.json")
        if idx:
            print(f"\n  📷 Visual (CLIP keyframe) index:")
            print(f"     Vectors: {idx.ntotal:,}")
            print(f"     Metadata entries: {len(meta):,}")

    # Knowledge indexes
    know_files = [f for f in OUTPUT_INDEXES.glob("knowledge_*.faiss") if not f.is_symlink() or "videorag" not in f.name]
    if know_files:
        print(f"\n  📚 Knowledge indexes:")
        for f in know_files:
            idx, _ = load_index_meta(f, f.with_suffix(".faiss_map.json") if ".faiss" in f.suffix else f.with_name(f.stem + "_map.json"))
            if idx:
                print(f"     {f.name:40s}: {idx.ntotal:,} vectors")

    print("\n" + "=" * 70)


def build_unified() -> None:
    """Build unified knowledge FAISS index from all chunk datasets."""
    print("\n" + "=" * 70)
    print("  🔢 Building Unified Knowledge Index")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Collect all chunks
    all_chunks = []
    for name, chunk_dir in CHUNK_DIRS.items():
        chunks = load_chunks(chunk_dir)
        for c in chunks:
            c["_dataset"] = name
        all_chunks.extend(chunks)
        logger.info(f"  {name}: {len(chunks):,} chunks")

    logger.info(f"  Total: {len(all_chunks):,} chunks")

    # Get unique descriptions
    logger.info(f"  Embedding...")
    texts = []
    for c in all_chunks:
        desc = c.get("description", "") or c.get("text", "")
        if desc:
            texts.append(desc)
        else:
            texts.append(f"{c.get('movie_id', '')} {c.get('situation', '')}")

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=512)
    embeddings = embeddings.astype("float32")
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Build metadata
    meta = []
    for c, desc in zip(all_chunks, texts):
        meta.append({
            "chunk_id": c.get("chunk_id", ""),
            "movie_id": c.get("movie_id", "") or c.get("youtube_id", ""),
            "video_id": c.get("youtube_id", "") or c.get("movie_id", ""),
            "dataset": c.get("_dataset", ""),
            "description": desc,
            "start_seconds": c.get("start_seconds"),
            "end_seconds": c.get("end_seconds"),
            "title": c.get("title", ""),
        })

    # Save
    OUTPUT_INDEXES.mkdir(parents=True, exist_ok=True)
    idx_path = OUTPUT_INDEXES / "knowledge_unified.faiss"
    meta_path = OUTPUT_INDEXES / "knowledge_unified_map.json"

    faiss.write_index(index, str(idx_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    size_mb = idx_path.stat().st_size // (1024 * 1024)
    logger.info(f"  ✅ Unified knowledge index: {index.ntotal:,} vectors, {size_mb}MB")

    # Test search
    test_query = "a person diving into the ocean"
    q_emb = model.encode([test_query]).astype("float32")
    q_emb /= (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
    D, I = index.search(q_emb, 3)
    print(f"\n  Test query: '{test_query}'")
    for i, (d, idx_r) in enumerate(zip(D[0], I[0])):
        if idx_r < len(meta):
            m = meta[idx_r]
            print(f"    {i+1}. [{d:.3f}] {m['dataset']}/{m['chunk_id']}: {m['description'][:60]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build/check unified FAISS indexes")
    parser.add_argument("--check-only", action="store_true", help="Only check, don't build")
    parser.add_argument("--build", action="store_true", help="Build unified index")
    args = parser.parse_args()

    if args.check_only or not args.build:
        check_all()
    if args.build:
        build_unified()


if __name__ == "__main__":
    main()
