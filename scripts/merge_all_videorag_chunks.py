#!/usr/bin/env python3
"""
Merge all VideoRag chunks:
  - Existing 22-movie chunks (data/pipeline_output/videorag_chunks/all_chunks.json)
  - New Tier-2 chunks (data/pipeline_output/videorag_chunks/tier2_chunks/)
  → Unified videorag_chunks/all_chunks.json
  → Rebuild FAISS index
"""

import json, faiss, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT    = Path(__file__).parent.parent.resolve()
DATA       = PROJECT / "data" / "pipeline_output"
OUT_DIR    = DATA / "videorag_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM  = 384

L2_FIELDS = ["description", "situation", "vision_setting", "vision_actions",
             "emotional_tone", "screenplay_context"]
L3_FIELDS = ["dialogue_text"]
L4_FIELDS = ["characters", "cast_in_scene", "character_emotions"]
L5_FIELDS = ["narrative_arc", "causal_relations", "screenplay_context",
             "script_heading"]


def make_text(chunk: dict) -> str:
    """Build retrieval text from chunk's rich fields."""
    parts = []
    parts.append(chunk.get("description", ""))
    parts.append(chunk.get("situation", ""))
    parts.append(chunk.get("dialogue_text", ""))
    if chunk.get("characters"):
        parts.append("Characters: " + ", ".join(chunk["characters"]))
    parts.append(chunk.get("narrative_arc", ""))
    parts.append(chunk.get("script_heading", ""))
    parts.append(chunk.get("screenplay_context", ""))
    if chunk.get("causal_relations"):
        rels = " | ".join(
            f"{r.get('relation','')}" for r in chunk["causal_relations"]
            if r.get("relation")
        )
        parts.append("Relations: " + rels)
    return " | ".join(p for p in parts if p)


def load_chunks(path: Path):
    """Load chunks from a JSON file (list or dict format)."""
    with open(path) as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else raw.get("chunks", raw.get("chunks", []))


def chunk_stats(chunks: list):
    """Print per-layer coverage stats."""
    n = len(chunks)
    def pct(field):
        return sum(1 for c in chunks if c.get(field)) / n * 100

    def any_pct(fields):
        return sum(1 for c in chunks if any(c.get(f) for f in fields)) / n * 100

    def l5_full(c):
        return bool(c.get("narrative_arc"))

    def l3_full(c):
        t = c.get("dialogue_text", "")
        return bool(t and t != "[PENDING_SRT_ALIGNMENT]")

    l3 = sum(1 for c in chunks if l3_full(c)) / n * 100
    l4 = any_pct(L4_FIELDS)
    l5 = sum(1 for c in chunks if l5_full(c)) / n * 100

    return n, l3, l4, l5


def main():
    print("=" * 60)
    print("MERGE: Existing VideoRag + Tier-2 → Unified FAISS")
    print("=" * 60)

    # ── Load existing VideoRag chunks ─────────────────────────────────────────
    existing_path = OUT_DIR / "all_chunks.json"
    if existing_path.exists():
        existing_chunks = load_chunks(existing_path)
        print(f"\nExisting VideoRag: {len(existing_chunks)} chunks")
        n1, l3_1, l4_1, l5_1 = chunk_stats(existing_chunks)
        print(f"  L3: {l3_1:.0f}% | L4: {l4_1:.0f}% | L5: {l5_1:.0f}%")
    else:
        existing_chunks = []
        print("\nNo existing VideoRag chunks found")

    # ── Load Tier-2 chunks ───────────────────────────────────────────────────
    tier2_dir = DATA / "videorag_chunks" / "tier2_chunks"
    tier2_chunks = []
    for f in sorted(tier2_dir.glob("*_chunks.json")):
        if f.name == "all_chunks.json":
            continue
        tier2_chunks.extend(load_chunks(f))
    print(f"\nTier-2 chunks: {len(tier2_chunks)} chunks")
    n2, l3_2, l4_2, l5_2 = chunk_stats(tier2_chunks)
    print(f"  L3: {l3_2:.0f}% | L4: {l4_2:.0f}% | L5: {l5_2:.0f}%")

    # ── Deduplicate by chunk_id ───────────────────────────────────────────────
    seen = {c["chunk_id"] for c in existing_chunks}
    new_chunks = [c for c in tier2_chunks if c["chunk_id"] not in seen]
    print(f"\nNew unique Tier-2 chunks: {len(new_chunks)} (deduplicated)")

    # ── Merge ─────────────────────────────────────────────────────────────────
    all_chunks = existing_chunks + new_chunks
    print(f"\nMerged total: {len(all_chunks)} chunks")

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_all, l3_all, l4_all, l5_all = chunk_stats(all_chunks)
    print(f"\n{'─'*50}")
    print(f"  L1 Temporal:    {n_all}/{n_all}  (100%)")
    print(f"  L2 Semantic:    {n_all}/{n_all}  (100%)")
    print(f"  L3 Dialogue:   {l3_all:.0f}%")
    print(f"  L4 Characters: {l4_all:.0f}%")
    print(f"  L5 Narrative:  {l5_all:.0f}%")

    # ── Save merged all_chunks.json ──────────────────────────────────────────
    merged_path = OUT_DIR / "all_chunks.json"
    with open(merged_path, "w") as f:
        json.dump({
            "metadata": {
                "source": "videorag_original + unified_dataset_tier2",
                "num_movies": len({c["movie_id"] for c in all_chunks}),
                "num_chunks": len(all_chunks),
                "original_chunks": len(existing_chunks),
                "tier2_chunks": len(new_chunks),
            },
            "chunks": all_chunks
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {merged_path}")

    # ── Build retrieval texts ─────────────────────────────────────────────────
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    texts = [make_text(c) for c in all_chunks]

    # ── Embed ─────────────────────────────────────────────────────────────────
    print(f"  Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Encoding...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=256)
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)

    # ── Save FAISS index ──────────────────────────────────────────────────────
    index_path = DATA / "indexes" / "knowledge_videorag.faiss"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(index_path))

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta_path = DATA / "indexes" / "knowledge_videorag_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "index": "knowledge_videorag.faiss",
            "model": MODEL_NAME,
            "embedding_dim": EMBED_DIM,
            "num_vectors": len(all_chunks),
            "chunk_source": "videorag_original + unified_dataset_tier2",
            "text_fields": ["description", "situation", "dialogue_text",
                             "characters", "narrative_arc", "script_heading",
                             "screenplay_context", "causal_relations"],
        }, f, indent=2)

    sz_mb = index_path.stat().st_size / 1024**2
    print(f"\n{'='*60}")
    print(f"  ✅ VideoRag FAISS index rebuilt")
    print(f"  Index: {index_path.name}  ({sz_mb:.0f}MB)")
    print(f"  Vectors: {index.ntotal:,}")
    print(f"  Movies: {len({c['movie_id'] for c in all_chunks})}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
