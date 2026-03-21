#!/usr/bin/env python3
"""
Step 9 – Whisper Alignment & FAISS Rebuild
==========================================
Aligns Whisper/SRT transcript segments → 5-Layer chunks (L3 Dialogue)
Cleans Whisper artifacts (repeated text, empty segments)
Rebuilds FAISS index with enriched chunks.

Usage:
  python scripts/align_whisper_and_rebuild.py --movie-id big_buck_bunny
"""
import argparse, json, sys, faiss, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT / "src"))

def load_chunks(movie_id: str):
    path = PROJECT / "data" / "pipeline_output" / movie_id / "chunks.json"
    with open(path) as f:
        raw = json.load(f)
    return raw.get("chunks", raw) if isinstance(raw, dict) else raw

def load_transcript(movie_id: str):
    path = PROJECT / "data" / "pipeline_output" / movie_id / "whisper_transcript.json"
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    segments = raw.get("segments", [])
    # Deduplicate: collapse segments with same text repeated consecutively
    cleaned = []
    prev_text = None
    for s in segments:
        t = s["text"].strip()
        if not t:
            continue
        if t != prev_text:
            cleaned.append({"start": s["start"], "end": s["end"], "text": t})
            prev_text = t
    return cleaned

def load_srt(movie_id: str):
    """Try to find an SRT file for this movie."""
    possible = [
        PROJECT / "data" / "movies" / f"{movie_id}.srt",
        PROJECT / "data" / "movies" / f"{movie_id}.en.srt",
        PROJECT / "subtitles" / f"{movie_id}.srt",
    ]
    for p in possible:
        if p.exists():
            return parse_srt(p)
    return []

def parse_srt(path: Path):
    """Parse SRT file into list of {start, end, text}."""
    import re
    with open(path) as f:
        content = f.read()
    blocks = re.split(r'\n\n+', content.strip())
    segments = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        time_match = re.match(r'(\d{2}:\d{2}:\d{2}),(\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}),(\d{3})', lines[1])
        if not time_match:
            continue
        def to_sec(h, m, s, ms):
            return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
        start = to_sec(*time_match.groups()[0:4])
        end = to_sec(*time_match.groups()[4:8])
        text = ' '.join(lines[2:])
        segments.append({"start": start, "end": end, "text": text})
    return segments

def align_segments_to_chunks(transcript_segments: list, chunks: list) -> list:
    """Assign dialogue_text to each chunk based on overlapping transcript segments."""
    # Group transcript segments by which chunk they fall into
    chunk_dialogue = {i: [] for i in range(len(chunks))}
    
    for seg in transcript_segments:
        for i, chunk in enumerate(chunks):
            cs, ce = chunk["start_seconds"], chunk["end_seconds"]
            ss, se = seg["start"], seg["end"]
            # Overlap check
            if ss < ce and se > cs:
                overlap = min(se, ce) - max(ss, cs)
                if overlap > 0.1:  # At least 0.1s overlap
                    chunk_dialogue[i].append(seg["text"])
    
    # Update chunks with dialogue
    updated = 0
    for i, chunk in enumerate(chunks):
        texts = chunk_dialogue[i]
        if texts:
            combined = " ".join(texts)
            # Truncate to avoid huge strings
            if len(combined) > 500:
                combined = combined[:500] + "..."
            chunk["dialogue_text"] = combined
            chunk["speaker"] = ""
            updated += 1
        else:
            if chunk.get("dialogue_text") in ("[NO_DIALOGUE]", "", None):
                chunk["dialogue_text"] = "[NO_DIALOGUE]"
    
    return chunks, updated

def make_text(c: dict) -> str:
    """Build searchable text from chunk for embedding."""
    parts = []
    parts.append(c.get("description", ""))
    parts.append(c.get("situation", ""))
    if c.get("dialogue_text") and c.get("dialogue_text") not in ("[NO_DIALOGUE]", ""):
        parts.append(c.get("dialogue_text", ""))
    if c.get("characters"):
        parts.append("Characters: " + ", ".join(c["characters"][:5]))
    parts.append(c.get("narrative_arc", ""))
    return " | ".join(p for p in parts if p and p not in ("unknown",))

def build_faiss(chunks: list, model_name: str = "all-MiniLM-L6-v2") -> faiss.Index:
    """Build FAISS IndexFlatIP from chunks."""
    model = SentenceTransformer(model_name)
    texts = [make_text(c) for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    vectors = vectors.astype("float32")
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vectors)
    return idx

def save_outputs(movie_id: str, chunks: list, faiss_idx):
    out_dir = PROJECT / "data" / "pipeline_output" / movie_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save chunks
    with open(out_dir / "chunks_enriched.json", "w") as f:
        json.dump({"metadata": {"movie_id": movie_id, "num_chunks": len(chunks), "enriched": True},
                   "chunks": chunks}, f, indent=2)
    
    # Save FAISS index
    faiss.write_index(faiss_idx, str(out_dir / "text_chunks.faiss"))
    
    print(f"  ✅ Saved {len(chunks)} chunks + FAISS index ({faiss_idx.ntotal} vectors)")

def print_stats(chunks: list):
    n = len(chunks)
    l1 = sum(1 for c in chunks if c.get("start_seconds", 0) >= 0) / max(n, 1) * 100
    l2 = sum(1 for c in chunks if c.get("description", "") not in ("", "unknown", "Scene at 0s")) / max(n, 1) * 100
    l3 = sum(1 for c in chunks if c.get("dialogue_text") not in ("[NO_DIALOGUE]", "", None, "unknown")) / max(n, 1) * 100
    l4 = sum(1 for c in chunks if c.get("characters")) / max(n, 1) * 100
    l5 = sum(1 for c in chunks if c.get("narrative_arc", "") not in ("", None, "unknown")) / max(n, 1) * 100
    
    print(f"\n📊 5-Layer Coverage After Enrichment:")
    print(f"  L1 Temporal:  {l1:.0f}%")
    print(f"  L2 Semantic:  {l2:.0f}%")
    print(f"  L3 Dialogue: {l3:.0f}%  ← {'✅ Whisper/SRT aligned!' if l3 > 0 else '❌ No transcript'}")
    print(f"  L4 Characters:{l4:.0f}%  ← {'✅' if l4 > 0 else '❌ Need VLM analysis'}")
    print(f"  L5 Narrative: {l5:.0f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie-id", default="big_buck_bunny")
    parser.add_argument("--srt", default=None, help="Path to SRT file (optional)")
    args = parser.parse_args()
    
    movie_id = args.movie_id
    print(f"\n🎬 Aligning Transcript → Chunks for: {movie_id}")
    print("=" * 60)
    
    # Load chunks
    chunks = load_chunks(movie_id)
    print(f"  Loaded {len(chunks)} chunks")
    
    # Try SRT first (higher quality), then Whisper
    srt_segs = []
    if args.srt:
        srt_segs = parse_srt(Path(args.srt))
        print(f"  Using SRT: {len(srt_segs)} segments from {args.srt}")
    else:
        srt_segs = load_srt(movie_id)
        if srt_segs:
            print(f"  Found SRT: {len(srt_segs)} segments")
    
    whisper_segs = []
    if not srt_segs:
        whisper_segs = load_transcript(movie_id)
        print(f"  Using Whisper: {len(whisper_segs)} cleaned segments")
    
    transcript_segs = srt_segs if srt_segs else whisper_segs
    
    if not transcript_segs:
        print("  ⚠️  No transcript found — L3 Dialogue will remain empty")
        print("  (Add SRT to data/movies/ or run Whisper first)")
    else:
        chunks, updated = align_segments_to_chunks(transcript_segs, chunks)
        print(f"  ✅ Aligned {updated}/{len(chunks)} chunks with dialogue text")
    
    # Print stats
    print_stats(chunks)
    
    # Build FAISS
    print(f"\n🔍 Building FAISS index...")
    idx = build_faiss(chunks)
    print(f"  ✅ FAISS index: {idx.ntotal} vectors")
    
    # Save
    save_outputs(movie_id, chunks, idx)
    print(f"\n✨ Done! Output: data/pipeline_output/{movie_id}/chunks_enriched.json")

if __name__ == "__main__":
    main()
