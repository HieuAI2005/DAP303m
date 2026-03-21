#!/usr/bin/env python3
"""
Phase 2: Align SRT subtitles → Tier-2 VideoRag chunks (L3 Dialogue).

For each of the 19 Tier-2 movies:
  1. Parse the .srt file from data/pipeline_output/subtitle/
  2. For each chunk, collect all SRT entries whose time range overlaps
  3. Concatenate dialogue text → fill dialogue_text field
  4. Handle speaker extraction from SRT markup where possible

Output: Updated tier-2_chunks.json files with dialogue_text filled.

Sample SRT entry:
  1
  00:01:16,820 --> 00:01:19,660
  I believe in America.
"""

import json, re, sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent.resolve()
DATA    = PROJECT / "data" / "pipeline_output"
CHUNK_DIR = DATA / "videorag_chunks" / "tier2_chunks"


# ── SRT parsing ────────────────────────────────────────────────────────────────

def parse_srt(path: Path) -> list:
    """Parse SRT file → list of {start, end, text} entries (seconds)."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            raw = path.read_text(encoding="latin-1")
        except Exception:
            return []
    entries = []
    blocks = re.split(r"\n\n+", raw.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        # Parse time line: "00:01:16,820 --> 00:01:19,660"
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1]
        )
        if not time_match:
            continue
        start = _ts2sec(time_match.group(1))
        end   = _ts2sec(time_match.group(2))
        text  = " ".join(_clean_srt_text(l) for l in lines[2:] if l.strip())
        if text:
            entries.append({"start": start, "end": end, "text": text})
    return entries


def _ts2sec(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' → float seconds."""
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _clean_srt_text(line: str) -> str:
    """Remove SRT markup (HTML tags, speaker labels, etc.)."""
    # Remove HTML tags like <i>, </i>, <u>, <font>
    line = re.sub(r"<[^>]+>", "", line)
    # Remove speaker labels like [NARRATOR:], (Man:), (Woman:)
    line = re.sub(r"\[[^\]]*:\]", "", line)
    line = re.sub(r"\([A-Z][a-z]+:", "", line)
    line = re.sub(r"\([A-Z][a-z]+\s+[A-Z][a-z]+:", "", line)
    # Collapse whitespace
    line = re.sub(r"\s+", " ", line).strip()
    return line


# ── SRT → Chunk alignment ──────────────────────────────────────────────────────

def align_srt_to_chunks(
    srt_entries: list,
    chunks: list,
    overlap_pct: float = 0.3,
) -> list:
    """
    For each chunk, collect SRT entries that overlap by at least overlap_pct.
    Returns list of aligned_dialogue strings.
    """
    results = []
    for chunk in chunks:
        start = chunk["start_seconds"]
        end   = chunk["end_seconds"]
        chunk_duration = end - start

        overlapping = []
        for entry in srt_entries:
            # Skip entries outside the chunk window
            if entry["end"] < start or entry["start"] > end:
                continue
            # Compute overlap ratio
            ov_start = max(start, entry["start"])
            ov_end   = min(end,   entry["end"])
            ov_dur   = max(0.0, ov_end - ov_start)
            ov_ratio = ov_dur / max(chunk_duration, 1.0)

            if ov_ratio >= overlap_pct or ov_dur >= 1.0:
                overlapping.append(entry["text"])

        dialogue = " ".join(overlapping).strip()
        if not dialogue:
            # No overlapping SRT — try a wider window
            for entry in srt_entries:
                if abs(entry["start"] - start) < 5.0:  # within 5s of chunk start
                    overlapping.append(entry["text"])
            dialogue = " ".join(overlapping).strip()

        results.append(dialogue)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    chunk_files = sorted([f for f in CHUNK_DIR.glob("*_chunks.json")
                          if f.name != "all_chunks.json"
                          and "_chunks_chunks" not in f.name])
    if not chunk_files:
        print("ERROR: No Tier-2 chunk files found.")
        sys.exit(1)

    total_chunks     = 0
    total_aligned    = 0
    total_empty      = 0

    for chunk_path in chunk_files:
        movie_id = chunk_path.stem.replace("_chunks", "")  # "tt0109830_chunks" → "tt0109830"
        srt_path = DATA / "subtitle" / f"{movie_id}.srt"

        # Load chunks (handle both list and dict formats)
        with open(chunk_path) as f:
            raw = json.load(f)
        chunks = raw if isinstance(raw, list) else raw.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            continue
        if not chunks:
            continue

        # Parse SRT
        if not srt_path.exists():
            print(f"  ⚠ {movie_id}: SRT not found")
            continue

        srt_entries = parse_srt(srt_path)
        if not srt_entries:
            print(f"  ⚠ {movie_id}: SRT empty, skipping")
            continue

        # Align
        aligned = align_srt_to_chunks(srt_entries, chunks)

        # Update chunks
        filled_count = 0
        for chunk, dialogue in zip(chunks, aligned):
            chunk["dialogue_text"] = dialogue
            if dialogue and dialogue != "[PENDING_SRT_ALIGNMENT]":
                filled_count += 1

        # Save updated chunks
        with open(chunk_path, "w") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        total_chunks  += len(chunks)
        total_aligned += filled_count
        total_empty   += len(chunks) - filled_count

        cov = filled_count / max(len(chunks),1) * 100
        print(f"  ✅ {movie_id}: {filled_count}/{len(chunks)} chunks aligned ({cov:.0f}%)")

    # ── Re-write merged all_chunks.json ────────────────────────────────────
    all_chunks = []
    for chunk_path in sorted([f for f in CHUNK_DIR.glob("*_chunks.json")
                              if f.stem != "all"]):
        with open(chunk_path) as f:
            raw = json.load(f)
        items = raw if isinstance(raw, list) else raw.get("chunks", [])
        all_chunks.extend(items)

    merged = {
        "metadata": {
            "source": "unified_dataset",
            "num_movies": len(chunk_files),
            "num_chunks": len(all_chunks),
            "l3_note": "dialogue_text aligned from SRT subtitles",
        },
        "chunks": all_chunks
    }
    with open(CHUNK_DIR / "all_chunks.json", "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ✅ L3 Dialogue alignment done")
    print(f"  Total:  {total_chunks:,} chunks")
    print(f"  Filled: {total_aligned:,} ({total_aligned/max(total_chunks,1)*100:.1f}%)")
    print(f"  Empty:  {total_empty:,} ({total_empty/max(total_chunks,1)*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
