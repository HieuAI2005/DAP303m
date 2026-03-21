#!/usr/bin/env python3
"""
build_moviegraphs_chunks.py
==========================
Build 5-Layer chunks from MovieGraphs dataset (52 movies, 7761 clips).

MovieGraphs has real scene graph annotations:
  ✅ characters, relationships, attributes
  ✅ scene descriptions with situational context
  ✅ graph edges: APPEARS_IN, INTERACTS_WITH, etc.

Output:
  - data/pipeline_output/moviegraphs_chunks/all_chunks.json

Usage:
    python scripts/build_moviegraphs_chunks.py
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Add MovieGraphs loader path
MG_LOADER_PATH = Path("/home/hiwe/project/DAP303m/VideoRag/data/MovieGraphs_repo/py3loader_new")
sys.path.insert(0, str(MG_LOADER_PATH))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MovieGraphsBuilder")


# ── Paths ────────────────────────────────────────────────────────────────────

MG_PKL = PROJECT_ROOT / "data" / "MovieGraphs_repo" / "py3loader_new" / "all_movies.pkl"
if not MG_PKL.exists():
    MG_PKL = Path("/home/hiwe/project/DAP303m/VideoRag/data/MovieGraphs_repo/py3loader_new/all_movies.pkl")

OUTPUT_CHUNKS = PROJECT_ROOT / "data" / "pipeline_output" / "moviegraphs_chunks"


# ── Schema ─────────────────────────────────────────────────────────────────

def build_chunk(movie_id: str, movie, clip_graph, idx: int) -> dict:
    """Build 5-Layer chunk from MovieGraphs ClipGraph object."""

    # Extract from movie metadata
    title = getattr(movie, "title", movie_id)
    year = getattr(movie, "year", "")

    # ClipGraph fields
    description = getattr(clip_graph, "description", "")
    situation = getattr(clip_graph, "situation", "")
    scene_label = getattr(clip_graph, "scene_label", "")
    video_info = getattr(clip_graph, "video", {}) or {}
    castlist_raw = getattr(movie, "castlist", "[]")

    # Timestamps from video info
    start_sec = float(video_info.get("ss", 0)) if isinstance(video_info, dict) else 0.0
    end_sec = float(video_info.get("es", 0)) if isinstance(video_info, dict) else 0.0

    chunk_id = f"mg_{movie_id}_{idx:04d}"

    # Characters from NetworkX graph
    try:
        G = getattr(clip_graph, "G", None)
        if G is not None:
            entities = list(G.nodes()) if hasattr(G, "nodes") else []
            char_names = [n for n in entities if G.nodes[n].get("type", "") == "character"] if hasattr(G, "nodes") else entities[:5]
        else:
            char_names = []
    except:
        char_names = []

    # Parse castlist
    import json as _json
    try:
        cast_list = _json.loads(castlist_raw) if isinstance(castlist_raw, str) else []
    except:
        cast_list = []

    # Relationships from graph
    causal_rels = []
    try:
        G = getattr(clip_graph, "G", None)
        if G is not None and hasattr(G, "edges"):
            for u, v, data in G.edges(data=True):
                causal_rels.append({
                    "type": "relationship",
                    "from": str(u),
                    "relation": data.get("relation", data.get("predicate", "related_to")),
                    "to": str(v),
                })
    except:
        pass

    # Graph nodes as entities
    entities = []
    try:
        G = getattr(clip_graph, "G", None)
        if G is not None and hasattr(G, "nodes"):
            entities = [str(n) for n in G.nodes()]
    except:
        pass

    # Get triplets
    triplets = []
    try:
        triplets = getattr(clip_graph, "find_all_triplets", lambda: [])()
    except:
        pass

    return {
        # L1: Temporal Anchor
        "chunk_id": chunk_id,
        "movie_id": movie_id,
        "video_id": movie_id,
        "title": f"{title} ({year})" if year else str(title),
        "start_seconds": start_sec,
        "end_seconds": end_sec,
        "duration": end_sec - start_sec,
        "frame_start": None,
        "frame_end": None,
        "scene_id": idx,

        # L2: Semantic Description
        "description": str(description) if description else "",
        "situation": str(situation) if situation else (str(scene_label) if scene_label else "unspecified"),
        "vision_setting": _infer_setting(str(scene_label), str(situation)),
        "vision_actions": _infer_actions(str(description), str(situation)),
        "emotional_tone": _infer_emotion(cast_list),

        # L3: Dialogue & Audio
        "text": str(description) if description else "",
        "dialogue_text": "",  # MovieGraphs doesn't have transcripts
        "speaker": "",
        "audio_events": [],
        "background_music": "",

        # L4: Cast & Characters
        "characters": char_names,
        "cast_in_scene": char_names,
        "character_emotions": {},
        "face_tracking_ids": [],
        "action_labels": [str(scene_label)] if scene_label else [],

        # L5: Narrative
        "script_heading": str(scene_label) if scene_label else "",
        "screenplay_context": str(situation) if situation else "",
        "narrative_arc": _infer_narrative(str(situation), str(description)),
        "causal_relations": causal_rels,
        "scene_graph": {
            "entities": entities,
            "triplets": triplets if isinstance(triplets, list) else [],
            "situation": str(situation) if situation else "",
        },

        # Metadata
        "source": "moviegraphs",
        "split": "test",
        "language": "en",
        "type": "movie_scene_graph",
        "keyframe_paths": [],
        "num_keyframes": 0,
    }


def _infer_setting(scene_label: str, situation: str) -> str:
    s = f"{scene_label} {situation}".lower()
    if any(w in s for w in ["beach", "ocean", "sea"]): return "beach"
    if any(w in s for w in ["car", "drive", "street"]): return "outdoor_vehicle"
    if any(w in s for w in ["home", "bedroom", "kitchen", "house"]): return "indoor_home"
    if any(w in s for w in ["office", "work"]): return "indoor_office"
    if any(w in s for w in ["restaurant", "cafe", "bar"]): return "indoor_restaurant"
    if any(w in s for w in ["hospital", "doctor"]): return "indoor_medical"
    if any(w in s for w in ["school", "classroom"]): return "indoor_school"
    if any(w in s for w in ["outdoor", "park", "city"]): return "outdoor_urban"
    return "indoor" if any(w in s for w in ["indoor", "room"]) else "outdoor"


def _infer_actions(description: str, situation: str) -> list:
    text = f"{description} {situation}".lower()
    actions = []
    for verb in ["talking", "driving", "fighting", "kissing", "dancing", "eating",
                 "running", "laughing", "crying", "working", "playing"]:
        if verb in text:
            actions.append(verb)
    return actions if actions else ["interacting"]


def _infer_emotion(attributes: list) -> str:
    attr_lower = [str(a).lower() for a in attributes]
    if any(w in attr_lower for w in ["happy", "joy", "excited"]): return "positive_happy"
    if any(w in attr_lower for w in ["sad", "crying"]): return "negative_sad"
    if any(w in attr_lower for w in ["angry", "fighting"]): return "negative_angry"
    if any(w in attr_lower for w in ["scary", "fear"]): return "negative_fearful"
    if any(w in attr_lower for w in ["romantic", "love"]): return "positive_romantic"
    if any(w in attr_lower for w in ["funny", "comedy"]): return "positive_funny"
    return "neutral"


def _infer_narrative(situation: str, description: str) -> str:
    s = f"{situation} {description}".lower()
    if any(w in s for w in ["intro", "beginning", "opening"]): return "introduction"
    if any(w in s for w in ["conflict", "argument", "tension", "fight"]): return "rising_action"
    if any(w in s for w in ["kiss", "resolution", "win", "celebrate"]): return "climax"
    if any(w in s for w in ["ending", "final"]): return "resolution"
    return "scene"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  🎬 MovieGraphs Chunk Builder")
    print("=" * 70)

    if not MG_PKL.exists():
        logger.error(f"MovieGraphs pickle not found: {MG_PKL}")
        logger.info("Run: python scripts/download_process_datasets.sh --verify moviegraphs")
        return

    # Load MovieGraphs data
    logger.info(f"Loading: {MG_PKL}")
    with open(MG_PKL, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        movies = data
    elif isinstance(data, dict):
        movies = list(data.values())
    else:
        logger.error(f"Unknown MovieGraphs format: {type(data)}")
        return

    logger.info(f"Loaded {len(movies)} movies")

    # Handle both dict and list formats
    if isinstance(movies, list):
        movies_iter = {getattr(m, "imdb_key", f"movie_{i}"): m for i, m in enumerate(movies)}.items()
    elif isinstance(movies, dict):
        movies_iter = movies.items()
    else:
        logger.error(f"Unknown MovieGraphs format: {type(movies)}")
        return

    # Build chunks
    all_chunks = []
    movie_stats = {}

    for movie_id, movie in movies_iter:
        title = getattr(movie, "title", movie_id)

        # Get all clip_graphs (OrderedDict)
        clip_graphs = getattr(movie, "clip_graphs", {})
        if not clip_graphs:
            continue

        movie_stats[movie_id] = {"title": title, "clips": len(clip_graphs)}

        for idx, (cg_id, clip_graph) in enumerate(clip_graphs.items()):
            chunk = build_chunk(movie_id, movie, clip_graph, idx)
            all_chunks.append(chunk)

    logger.info(f"Built {len(all_chunks)} chunks from {len(movie_stats)} movies")
    for mid, stats in sorted(movie_stats.items(), key=lambda x: -x[1]["clips"])[:5]:
        logger.info(f"  {mid}: {stats['title']} ({stats['clips']} clips)")

    # Save
    OUTPUT_CHUNKS.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_CHUNKS / "all_chunks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "source": "moviegraphs",
                "version": "v1",
                "num_chunks": len(all_chunks),
                "num_movies": len(movie_stats),
                "movies": list(movie_stats.keys()),
            },
            "chunks": all_chunks,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved: {out_file}")

    # Stats
    has_chars = sum(1 for c in all_chunks if c.get("characters"))
    has_rels = sum(1 for c in all_chunks if c.get("causal_relations"))
    has_entities = sum(1 for c in all_chunks if c.get("scene_graph", {}).get("entities"))

    print(f"\n📊 Field Coverage:")
    print(f"  Characters:    {has_chars}/{len(all_chunks)} ({int(100*has_chars/len(all_chunks))}%)")
    print(f"  Relationships: {has_rels}/{len(all_chunks)} ({int(100*has_rels/len(all_chunks))}%)")
    print(f"  Entities:     {has_entities}/{len(all_chunks)} ({int(100*has_entities/len(all_chunks))}%)")


if __name__ == "__main__":
    main()
