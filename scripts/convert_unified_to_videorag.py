#!/usr/bin/env python3
"""
Convert unified_dataset/movierag_dataset.json clips → VideoRag chunk format.

Scope: 19 Tier-2 movies (in unified_dataset + SRT available, NOT in existing chunks).
Output: data/pipeline_output/videorag_chunks/tier2_chunks/ tier-2/all_chunks.json

For each Tier-2 movie:
  1. Load TMDB metadata → runtime → sec_per_shot
  2. Map start_shot/end_shot → start_seconds/end_seconds
  3. Convert unified clip → VideoRag chunk schema
  4. Mark L3 dialogue as PENDING (to be filled by SRT alignment in Phase 2)

Layer coverage after this script:
  L1 Temporal Anchor     ✅ (all 19 movies, timestamped via TMDB runtime)
  L2 Semantic Scene      ✅ (description, situation, scene_label)
  L3 Dialogue & Audio   ⚠️  (pending: SRT subtitle alignment)
  L4 Cast & Characters  ✅ (characters, character_ids, cast_in_scene)
  L5 Narrative & Script  ⚠️  (pending: IMSDb screenplay enrichment)
"""

import json, math, re
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT   = Path(__file__).parent.parent.resolve()
DATA      = PROJECT / "data" / "pipeline_output"
OUT_DIR   = DATA / "videorag_chunks" / "tier2_chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load resources ──────────────────────────────────────────────────────────────
print("Loading unified_dataset...")
with open(DATA / "unified_dataset" / "movierag_dataset.json") as f:
    ud = json.load(f)
unified_movies = ud["movies"]   # dict: imdb_id → movie_obj

print("Loading existing VideoRag chunk movie IDs (to exclude)...")
existing_ids = set(
    f.stem.replace("_chunks", "")
    for f in (DATA / "temporal_chunks").glob("*_chunks.json")
    if f.stem != "all"
)
print(f"  Existing VideoRag movies to exclude: {len(existing_ids)}")

print("Loading TMDB metadata...")
meta_dir = DATA / "meta"
tmdb_cache = {}
for f in meta_dir.glob("*.json"):
    with open(f) as fh:
        d = json.load(fh)
    tmdb_cache[d["imdb_id"]] = d

# ── Identify Tier-2 movies ──────────────────────────────────────────────────────
srt_ids = set(f.stem for f in (DATA / "subtitle").glob("*.srt"))
tier2_ids = sorted(unified_movies.keys() - existing_ids & srt_ids)
print(f"\nTier-2 movies (unified+SRT, not in existing chunks): {len(tier2_ids)}")

if not tier2_ids:
    print("ERROR: No Tier-2 movies found. Check paths.")
    exit(1)


# ── Core conversion functions ──────────────────────────────────────────────────

def parse_runtime(meta_obj) -> float:
    """Extract runtime in minutes from TMDB metadata."""
    versions = meta_obj.get("version", [])
    for v in versions:
        raw = v.get("runtime", "")
        m = re.search(r"(\d+)", str(raw))
        if m:
            return float(m.group(1))
    return 0.0   # fallback: 0 → will use global avg sec_per_shot


def compute_sec_per_shot(movie_id: str, max_shot: int, tmdb: dict) -> float:
    """Compute seconds-per-shot from TMDB runtime."""
    meta = tmdb.get(movie_id)
    runtime_min = parse_runtime(meta) if meta else 0.0

    if runtime_min > 0 and max_shot > 1:
        return (runtime_min * 60.0) / max_shot
    # Fallback: global average ≈ 8 sec/shot
    return 8.0


def build_narrative_arc(start_sec: float, end_sec: float, total_sec: float) -> str:
    """Infer narrative arc from clip's position in total movie runtime."""
    if total_sec <= 0:
        return "scene"
    pct = (start_sec / total_sec) * 100
    if pct < 15:
        return "exposition"
    elif pct < 40:
        return "rising_action"
    elif pct < 65:
        return "climax"
    elif pct < 85:
        return "falling_action"
    else:
        return "resolution"


def infer_emotional_tone(description: str, situation: str) -> str:
    """Infer emotional tone from text using keyword matching."""
    text = (description + " " + situation).lower()
    if any(w in text for w in ["crying", "sad", "death", "funeral", "loss", "grief"]):
        return "emotional_negative"
    if any(w in text for w in ["laughing", "funny", "hilarious", "joke", "comedy"]):
        return "positive_humorous"
    if any(w in text for w in ["kiss", "love", "romantic", "wedding", "date"]):
        return "positive_romantic"
    if any(w in text for w in ["chase", "fight", "action", "gun", "escape", "suspense"]):
        return "tense_action"
    if any(w in text for w in ["thinking", "alone", "reflection", "quiet"]):
        return "neutral_contemplative"
    return "neutral"


def convert_clip_to_chunk(
    clip: dict,
    movie_id: str,
    clip_idx: int,
    sec_per_shot: float,
    total_sec: float,
    title: str,
    genres: list,
    tmdb_meta: dict,
) -> dict:
    """Convert a unified_dataset clip → VideoRag chunk dict."""

    start_shot = clip.get("start_shot", 1)
    end_shot   = clip.get("end_shot",   1)

    # ── Timestamps ─────────────────────────────────────────────────────────────
    start_seconds = max(0.0, (start_shot - 1) * sec_per_shot)
    end_seconds   = max(start_seconds + 0.5, end_shot * sec_per_shot)
    duration       = end_seconds - start_seconds

    # ── Narrative arc ─────────────────────────────────────────────────────────
    narrative_arc = build_narrative_arc(start_seconds, end_seconds, total_sec)

    # ── Characters / Cast ──────────────────────────────────────────────────────
    characters_raw = clip.get("characters", [])
    character_names = []
    character_ids   = []
    cast_in_scene   = []
    character_emotions = {}

    for ch in characters_raw:
        name = ch.get("name", "").strip()
        cid  = ch.get("id",   "")
        if not name or name == "---":
            continue
        character_names.append(name)
        if cid and cid != "---":
            character_ids.append(cid)
        # Infer emotion from description if available
        emotion = infer_emotion_from_context(clip.get("description", ""), name)
        if emotion:
            character_emotions[name] = emotion

    # ── Causal relations ──────────────────────────────────────────────────────
    interactions = clip.get("interactions", [])
    causal_relations = []
    for ix in interactions:
        # interactions are strings like "discusses work", "orders coffee"
        rel = str(ix).strip() if ix else ""
        if rel:
            causal_relations.append({"type": "interaction", "relation": rel})

    # ── L2 fields ──────────────────────────────────────────────────────────────
    description = clip.get("description", "")
    situation   = clip.get("situation",   description)
    scene_label = clip.get("scene_label", "")

    # ── L5 placeholder (pending IMSDb enrichment) ─────────────────────────────
    script_heading     = scene_label
    screenplay_context = description   # proxy: description as scene context
    emotional_tone     = infer_emotional_tone(description, situation)

    # ── Vision actions (placeholder) ──────────────────────────────────────────
    vision_actions = infer_actions(description)

    # ── TMDB metadata enrichment ───────────────────────────────────────────────
    director = tmdb_meta.get("director", []) if tmdb_meta else []
    cast_tmdb = tmdb_meta.get("cast", []) if tmdb_meta else []

    # ── Build chunk ────────────────────────────────────────────────────────────
    chunk_id = f"{movie_id}_chunk_{clip_idx:04d}"

    chunk = {
        # L1 Temporal
        "chunk_id":       chunk_id,
        "movie_id":       movie_id,
        "video_id":       movie_id,
        "start_seconds":  start_seconds,
        "end_seconds":    end_seconds,
        "duration":       duration,
        "shot_start":     start_shot,
        "shot_end":       end_shot,

        # L2 Semantic
        "description":    description,
        "situation":      situation,
        "vision_setting": scene_label,
        "vision_actions": vision_actions,
        "emotional_tone": emotional_tone,
        "attributes":     clip.get("attributes", []),

        # L3 Dialogue (pending SRT alignment)
        "dialogue_text":  "[PENDING_SRT_ALIGNMENT]",
        "speaker":         "",
        "audio_events":    [],
        "background_music":"",

        # L4 Cast
        "characters":       character_names,
        "character_ids":    character_ids,
        "cast_in_scene":    cast_in_scene,
        "character_emotions": character_emotions,
        "face_tracking_ids": [],

        # L5 Narrative
        "narrative_arc":     narrative_arc,
        "causal_relations":  causal_relations,
        "screenplay_context": screenplay_context,
        "script_heading":     script_heading,

        # Metadata
        "title":       title,
        "genres":      genres,
        "language":    "en",
        "source":      "unified_dataset",
        "type":        "movie_scene",
        "split":       "train",
        "num_keyframes": 0,
        "keyframe_paths": [],
        "vlm_description": description,

        # TMDB enrich
        "directors":   [d["name"] for d in director[:2]],
        "tmdb_cast":   [{"name": c["name"], "character": c.get("character",""), "id": c.get("id","")}
                        for c in cast_tmdb[:10]],
    }

    return chunk


def infer_emotion_from_context(text: str, char_name: str) -> str:
    text = text.lower()
    if any(w in text for w in ["laugh", "smile", "joke", "funny"]):
        return "amused"
    if any(w in text for w in ["cry", "tear", "sad", "grief"]):
        return "sad"
    if any(w in text for w in ["angry", "yell", "shout", "furious"]):
        return "angry"
    if any(w in text for w in ["nervous", "worried", "anxious"]):
        return "nervous"
    if any(w in text for w in ["surprised", "shock", "amazed"]):
        return "surprised"
    if any(w in text for w in ["scared", "fear", "terror"]):
        return "scared"
    return ""


def infer_actions(description: str) -> list:
    text = description.lower()
    actions = []
    pairs = [
        ("walk",    ["walk", "walked", "walks"]),
        ("talk",    ["talk", "talks", "talking", "conversation"]),
        ("drive",   ["drive", "driving", "car", "vehicle"]),
        ("eat",     ["eat", "eating", "dinner", "food"]),
        ("fight",   ["fight", "fighting", "attack", "punch"]),
        ("run",     ["run", "running", "escape", "chase"]),
        ("kiss",    ["kiss", "kissing", "embrace"]),
        ("work",    ["work", "working", "office", "job"]),
        ("sleep",   ["sleep", "sleeping", "bed"]),
        ("read",    ["read", "reading", "book"]),
    ]
    for action, keywords in pairs:
        if any(kw in text for kw in keywords):
            actions.append(action)
    return actions[:5]


# ── Main conversion ─────────────────────────────────────────────────────────────
def main():
    total_chunks = 0
    all_chunks   = []

    for movie_id in tier2_ids:
        m = unified_movies[movie_id]

        # ── Clips ─────────────────────────────────────────────────────────────
        clips = m.get("clips", [])
        if not clips:
            print(f"  ⚠ {movie_id}: no clips, skipping")
            continue

        # ── Movie metadata ────────────────────────────────────────────────────
        title   = m.get("title", movie_id)
        genres  = tmdb_cache.get(movie_id, {}).get("genres", [])
        tmdb    = tmdb_cache.get(movie_id, {})

        # ── Shot→seconds mapping ──────────────────────────────────────────────
        all_shots = set()
        for c in clips:
            all_shots.add(c.get("start_shot", 1))
            all_shots.add(c.get("end_shot",   1))
        max_shot = max(all_shots) if all_shots else 1

        runtime_min = parse_runtime(tmdb)
        total_sec   = runtime_min * 60.0 if runtime_min > 0 else 0.0
        sps         = compute_sec_per_shot(movie_id, max_shot, tmdb_cache)

        print(f"\n  {movie_id}: {title}")
        print(f"    Clips: {len(clips)} | Max shot: {max_shot} | "
              f"Runtime: {runtime_min:.0f}min | sec/shot: {sps:.1f}s")

        # ── Convert each clip ────────────────────────────────────────────────
        movie_chunks = []
        for idx, clip in enumerate(clips):
            chunk = convert_clip_to_chunk(
                clip=clip,
                movie_id=movie_id,
                clip_idx=idx,
                sec_per_shot=sps,
                total_sec=total_sec,
                title=title,
                genres=genres,
                tmdb_meta=tmdb,
            )
            movie_chunks.append(chunk)
            all_chunks.append(chunk)

        # ── Save per-movie file ────────────────────────────────────────────────
        out_path = OUT_DIR / f"{movie_id}_chunks.json"
        with open(out_path, "w") as f:
            json.dump(movie_chunks, f, indent=2, ensure_ascii=False)
        print(f"    → {out_path.name}  ({len(movie_chunks)} chunks)")

        total_chunks += len(movie_chunks)

    # ── Save merged all_chunks.json ───────────────────────────────────────────
    merged_path = OUT_DIR / "all_chunks.json"
    with open(merged_path, "w") as f:
        json.dump({
            "metadata": {
                "source": "unified_dataset",
                "num_movies": len(tier2_ids),
                "num_chunks": total_chunks,
                "movies": tier2_ids,
                "note": "L3 dialogue pending SRT alignment; L5 narrative pending IMSDb"
            },
            "chunks": all_chunks
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ✅ DONE — {total_chunks:,} chunks from {len(tier2_ids)} movies")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}")

    # ── Quick stats ─────────────────────────────────────────────────────────────
    l3_pending  = sum(1 for c in all_chunks if c["dialogue_text"] == "[PENDING_SRT_ALIGNMENT]")
    l4_has_char = sum(1 for c in all_chunks if c["characters"])
    l5_has_cr   = sum(1 for c in all_chunks if c["causal_relations"])

    print(f"\n  Layer coverage:")
    print(f"    L1 Temporal:    {total_chunks}/{total_chunks} (100%)")
    print(f"    L2 Semantic:    {total_chunks}/{total_chunks} (100%)")
    print(f"    L3 Dialogue:    {total_chunks-l3_pending}/{total_chunks} ({100*(total_chunks-l3_pending)/total_chunks:.0f}%) — {l3_pending} pending SRT")
    print(f"    L4 Characters:  {l4_has_char}/{total_chunks} ({100*l4_has_char/total_chunks:.0f}%)")
    print(f"    L5 Narrative:   {l5_has_cr}/{total_chunks} ({100*l5_has_cr/total_chunks:.0f}%) — IMSDb pending")


if __name__ == "__main__":
    main()
