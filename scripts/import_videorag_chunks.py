#!/usr/bin/env python3
"""
import_videorag_chunks.py
=========================
Import real processed chunks from VideoRag into project_ky4's 5-Layer schema.

VideoRag has 3229 real chunks from 22 movies with:
  ✅ Real descriptions, characters, dialogue_text, scene_label
  ✅ keyframe_paths, shot boundaries
  ✅ cast_in_scene, attributes, interactions

project_ky4 needs:
  ✅ Full 5-Layer schema (L1-L5)
  ✅ FAISS visual index (frame + scene)
  ✅ Neo4j-ready knowledge graph

Strategy:
  1. Load VideoRag chunks from /path/to/VideoRag/data/temporal_chunks/
  2. Transform to project_ky4 5-Layer schema
  3. Add VideoRag keyframes to project_ky4's keyframe dir
  4. Build CLIP visual index from real keyframes
  5. Enrich L2-L5 with Groq API (situation, emotional_tone, narrative_arc...)
  6. Output to data/pipeline_output/videorag_chunks/all_chunks.json
  7. Build FAISS visual index

Usage:
    python scripts/import_videorag_chunks.py                    # Dry run (shows stats)
    python scripts/import_videorag_chunks.py --execute          # Actual import
    python scripts/import_videorag_chunks.py --enrich           # Also call Groq for L2-L5
    python scripts/import_videorag_chunks.py --build-index       # Build CLIP FAISS index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VideoRagImporter")

# ── Paths ────────────────────────────────────────────────────────────────────

VIDEORAG_ROOT = Path("/home/hiwe/project/DAP303m/VideoRag")
VIDEORAG_CHUNKS_DIR = VIDEORAG_ROOT / "data" / "temporal_chunks"
VIDEORAG_KEYFRAMES = VIDEORAG_ROOT / "data" / "trailer_keyframes"
VIDEORAG_VIDEOS = VIDEORAG_ROOT / "data" / "trailer_videos"
VIDEORAG_ACTIVITY_KEYFRAMES = VIDEORAG_ROOT / "data" / "activitynet_keyframes"
VIDEORAG_ACTIVITY_VIDEOS = VIDEORAG_ROOT / "data" / "activitynet_videos"

OUTPUT_ROOT = PROJECT_ROOT / "data"
OUTPUT_CHUNKS = OUTPUT_ROOT / "pipeline_output" / "videorag_chunks"
OUTPUT_KEYFRAMES = OUTPUT_ROOT / "pipeline_output" / "keyframes"
OUTPUT_INDEXES = OUTPUT_ROOT / "pipeline_output" / "indexes"
OUTPUT_GRAPHS = OUTPUT_ROOT / "pipeline_output" / "graphs"

# ── 5-Layer Schema Transformer ───────────────────────────────────────────────

def transform_videorag_chunk(vr_chunk: dict, movie_title: str = "") -> dict:
    """Transform VideoRag chunk into project_ky4 5-Layer schema."""

    # Parse timestamps
    start_sec = float(vr_chunk.get("start_seconds", 0))
    end_sec = float(vr_chunk.get("end_seconds", 0))
    duration = end_sec - start_sec

    # Build chunk_id
    movie_id = vr_chunk.get("movie_id", "unknown")
    chunk_idx = vr_chunk.get("clip_id", vr_chunk.get("chunk_id", "0"))
    if "_chunk_" in str(vr_chunk.get("chunk_id", "")):
        chunk_id = vr_chunk.get("chunk_id")
    else:
        chunk_id = f"{movie_id}_chunk_{int(chunk_idx):04d}"

    # L1: Temporal Anchor
    transformed = {
        "chunk_id": chunk_id,
        "movie_id": movie_id,
        "video_id": movie_id,           # for compatibility
        "title": movie_title or vr_chunk.get("title", ""),
        "start_seconds": start_sec,
        "end_seconds": end_sec,
        "duration": duration,
        "frame_start": None,
        "frame_end": None,
    }

    # L2: Semantic Description (from VideoRag real data)
    description = vr_chunk.get("description", "")
    situation = vr_chunk.get("situation", "")
    scene_label = vr_chunk.get("scene_label", "")
    attributes = vr_chunk.get("attributes", [])

    transformed.update({
        "description": description,
        "situation": situation or scene_label or "unspecified",
        "vision_setting": _infer_setting(scene_label, situation),
        "vision_actions": _infer_actions(description, situation),
        "emotional_tone": _infer_emotion(attributes, description),
        # VLM description: will be filled by Groq enrich
        "vlm_description": description,  # use existing as VLM proxy
    })

    # L3: Dialogue & Audio (from VideoRag real dialogue)
    dialogue_text = vr_chunk.get("dialogue_text", "") or ""
    dialogue_raw = vr_chunk.get("dialogue", "") or ""
    if isinstance(dialogue_text, list):
        dialogue_text = " ".join(str(d) for d in dialogue_text)
    if isinstance(dialogue_raw, list):
        dialogue_raw = " ".join(str(d) for d in dialogue_raw)

    transformed.update({
        "text": description,  # primary text = description
        "dialogue_text": dialogue_text or description,
        "speaker": "",  # VideoRag doesn't have per-speaker
        "audio_events": _detect_audio_events(dialogue_raw),
        "background_music": _detect_music(dialogue_raw),
    })

    # L4: Cast & Characters (from VideoRag real data)
    characters = vr_chunk.get("characters", [])
    cast_in_scene = vr_chunk.get("cast_in_scene", [])

    # Normalize characters
    if isinstance(characters, list):
        if characters and isinstance(characters[0], dict):
            char_names = [c.get("name", "") for c in characters]
        else:
            char_names = [str(c) for c in characters]
    else:
        char_names = []

    transformed.update({
        "characters": char_names,
        "cast_in_scene": cast_in_scene if isinstance(cast_in_scene, list) else char_names,
        "character_emotions": {},  # will be filled by VLM
        "face_tracking_ids": [],   # requires face detection on keyframes
        "action_labels": [],       # requires VideoMAE on video
    })

    # L5: Script & Narrative
    interactions = vr_chunk.get("interactions", [])
    if isinstance(interactions, str):
        interactions = [{"type": "interaction", "relation": interactions}]
    scene_label_text = vr_chunk.get("scene_label", "")

    transformed.update({
        "script_heading": scene_label_text,
        "screenplay_context": situation,
        "narrative_arc": _infer_narrative(situation, description),
        "causal_relations": _parse_causal(interactions),
        "scene_graph": {
            "scene_label": scene_label,
            "attributes": attributes,
            "interactions": interactions,
        },
    })

    # Metadata
    transformed.update({
        "source": "videorag",
        "split": "trailer",      # trailer/real-movie data
        "language": "en",
        "type": "movie_trailer",
        # keyframe paths (will be copied)
        "keyframe_paths": vr_chunk.get("keyframe_paths", []),
        "num_keyframes": vr_chunk.get("num_keyframes", 0),
        "shot_start": vr_chunk.get("shot_start"),
        "shot_end": vr_chunk.get("shot_end"),
        "timestamp_source": vr_chunk.get("timestamp_source", "annotation_frame"),
        "genres": vr_chunk.get("genres", []),
        "attributes": attributes,
        "interactions": interactions,
        "character_ids": vr_chunk.get("character_ids", []),
        # Original VideoRag data preserved
        "_videorag_original": {
            "clip_id": vr_chunk.get("clip_id"),
            "end_time": vr_chunk.get("end_time"),
            "start_time": vr_chunk.get("start_time"),
        },
    })

    return transformed


def _infer_setting(scene_label: str, situation: str) -> str:
    """Infer visual setting from scene label."""
    s = f"{scene_label} {situation}".lower()
    if any(w in s for w in ["beach", "ocean", "sea", "shore"]): return "beach"
    if any(w in s for w in ["car", "drive", "road", "street", "highway"]): return "outdoor_vehicle"
    if any(w in s for w in ["bedroom", "home", "house", "kitchen", "bathroom"]): return "indoor_home"
    if any(w in s for w in ["office", "work", "building"]): return "indoor_office"
    if any(w in s for w in ["restaurant", "cafe", "bar", "diner"]): return "indoor_restaurant"
    if any(w in s for w in ["forest", "mountain", "jungle", "desert"]): return "outdoor_nature"
    if any(w in s for w in ["city", "urban", "downtown"]): return "outdoor_urban"
    if any(w in s for w in ["underwater", "seabed", "ocean_floor"]): return "underwater"
    return "indoor" if any(w in s for w in ["indoor", "room", "inside"]) else "outdoor"


def _infer_actions(description: str, situation: str) -> list:
    """Infer action verbs from description."""
    text = f"{description} {situation}".lower()
    actions = []
    if any(w in text for w in ["running", "run", "sprint"]): actions.append("running")
    if any(w in text for w in ["walking", "walk"]): actions.append("walking")
    if any(w in text for w in ["talking", "speaking", "conversing", "chatting"]): actions.append("talking")
    if any(w in text for w in ["driving", "drives", "car"]): actions.append("driving")
    if any(w in text for w in ["fighting", "punch", "attack", "kick"]): actions.append("fighting")
    if any(w in text for w in ["kissing", "kiss"]): actions.append("kissing")
    if any(w in text for w in ["dancing", "dance"]): actions.append("dancing")
    if any(w in text for w in ["eating", "food", "cooking"]): actions.append("eating")
    if any(w in text for w in ["singing", "sing", "music"]): actions.append("singing")
    if any(w in text for w in ["crying", "tears", "sad"]): actions.append("crying")
    if any(w in text for w in ["laughing", "laugh", "smile"]): actions.append("laughing")
    if any(w in text for w in ["diving", "swimming", "water"]): actions.append("swimming")
    if any(w in text for w in ["shooting", "gun", "weapon"]): actions.append("shooting")
    return actions if actions else ["interacting"]


def _infer_emotion(attributes: list, description: str) -> str:
    """Infer emotional tone from attributes + description."""
    attr_lower = [a.lower() for a in attributes]
    text = description.lower()
    if any(w in attr_lower for w in ["happy", "joy", "excited", "cheerful"]): return "positive_happy"
    if any(w in attr_lower for w in ["sad", "crying", "depressed", "grief"]): return "negative_sad"
    if any(w in attr_lower for w in ["angry", "rage", "furious", "hostile"]): return "negative_angry"
    if any(w in attr_lower for w in ["fear", "scary", "horror", "terror", "afraid"]): return "negative_fearful"
    if any(w in attr_lower for w in ["romantic", "love", "passionate"]): return "positive_romantic"
    if any(w in text for w in ["tense", "danger", "threat", "suspense"]): return "tense"
    if any(w in attr_lower for w in ["funny", "comedy", "humor", "hilarious"]): return "positive_funny"
    if any(w in attr_lower for w in ["neutral", "normal"]): return "neutral"
    return "neutral"


def _detect_audio_events(dialogue_raw) -> list:
    """Detect audio events from bracketed dialogue markers."""
    if not dialogue_raw:
        return []
    # Handle list format
    if isinstance(dialogue_raw, list):
        dialogue_raw = " ".join(str(d) for d in dialogue_raw)
    else:
        dialogue_raw = str(dialogue_raw)
    events = []
    text_lower = dialogue_raw.lower()
    if "[music]" in text_lower or "[song]" in text_lower: events.append("background_music")
    if "[scream]" in text_lower or "[shout]" in text_lower: events.append("screaming")
    if "[gun]" in text_lower or "[shoot]" in text_lower: events.append("explosion")
    if "[footstep]" in text_lower or "[step]" in text_lower: events.append("footsteps")
    if "[wind]" in text_lower or "[rain]" in text_lower: events.append("ambient_nature")
    if "[car]" in text_lower or "[engine]" in text_lower: events.append("vehicle_engine")
    if "[applause]" in text_lower: events.append("applause")
    if "[crowd]" in text_lower: events.append("crowd_noise")
    if events: return events
    return ["speech"]  # default


def _detect_music(dialogue_raw) -> str:
    """Detect background music from dialogue markers."""
    if not dialogue_raw:
        return ""
    if isinstance(dialogue_raw, list):
        dialogue_raw = " ".join(str(d) for d in dialogue_raw)
    else:
        dialogue_raw = str(dialogue_raw)
    text_lower = dialogue_raw.lower()
    if "[epic music]" in text_lower: return "epic_orchestral"
    if "[sad music]" in text_lower: return "sad_piano"
    if "[tense music]" in text_lower: return "suspense_tension"
    if "[romantic music]" in text_lower: return "romantic_soft"
    if "[action music]" in text_lower: return "action_drums"
    if "[comedic music]" in text_lower: return "comedic_light"
    if "[horror music]" in text_lower: return "horror_dissonant"
    if "[music]" in text_lower: return "background_music"
    return ""


def _infer_narrative(situation: str, description: str) -> str:
    """Infer narrative arc label."""
    s = f"{situation} {description}".lower()
    if any(w in s for w in ["intro", "opening", "beginning"]): return "introduction"
    if any(w in s for w in ["conflict", "fight", "argument", "tension"]): return "rising_action"
    if any(w in s for w in ["kiss", "win", "celebrate", "resolution"]): return "climax"
    if any(w in s for w in ["ending", "final", "conclusion"]): return "resolution"
    if any(w in s for w in ["exposition", "setup"]): return "exposition"
    if any(w in s for w in ["transition", "montage"]): return "transition"
    return "scene"


def _parse_causal(interactions: list) -> list:
    """Parse causal relations from VideoRag interactions."""
    if not isinstance(interactions, list):
        return []
    relations = []
    for interaction in interactions:
        if isinstance(interaction, str):
            relations.append({"type": "interaction", "relation": interaction})
        elif isinstance(interaction, dict):
            rel = interaction.get("relation", "") or interaction.get("type", "")
            if rel:
                relations.append({"type": "interaction", "relation": str(rel)})
    return relations


# ── Keyframe Copying ──────────────────────────────────────────────────────────

def copy_keyframes(movie_id: str, dry_run: bool = True) -> list:
    """Copy keyframes from VideoRag to project_ky4 keyframe dir. Returns list of new paths."""
    src_dir = VIDEORAG_KEYFRAMES / movie_id
    if not src_dir.exists():
        logger.warning(f"  No keyframes for {movie_id} in VideoRag")
        return []

    dst_dir = OUTPUT_KEYFRAMES / movie_id
    src_frames = sorted(src_dir.glob("*.jpg"))

    if dry_run:
        logger.info(f"  Would copy {len(src_frames)} keyframes: {src_dir} → {dst_dir}")
        return [str(f) for f in src_frames]

    dst_dir.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for src in src_frames:
        dst = dst_dir / src.name
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)
        new_paths.append(str(dst))

    logger.info(f"  Copied {len(new_paths)} keyframes: {movie_id}/")
    return new_paths


# ── Main Import ───────────────────────────────────────────────────────────────

def import_videorag(dry_run: bool = True, copy_keyframes_flag: bool = True,
                    enrich: bool = False, build_index: bool = False) -> None:
    """Main import pipeline."""

    print("\n" + "=" * 70)
    print("  📦 VideoRag → project_ky4 Importer")
    print("=" * 70)
    print(f"  Source:     {VIDEORAG_CHUNKS_DIR}")
    print(f"  Dry run:    {dry_run}")
    print(f"  Enrich L2-L5: {enrich}")
    print(f"  Build index:  {build_index}")
    print(f"  Output:     {OUTPUT_CHUNKS}")
    print("=" * 70 + "\n")

    # Step 1: Load VideoRag chunks
    all_chunks_file = VIDEORAG_CHUNKS_DIR / "all_chunks.json"
    if not all_chunks_file.exists():
        logger.error(f"VideoRag all_chunks.json not found: {all_chunks_file}")
        return

    with open(all_chunks_file, encoding="utf-8") as f:
        vr_data = json.load(f)

    vr_chunks = vr_data.get("chunks", vr_data.get("chunks", []))
    if not isinstance(vr_chunks, list):
        vr_chunks = vr_data.get("chunks", [])

    # Get movie metadata from all_chunks
    metadata = vr_data.get("metadata", {})

    logger.info(f"Loaded {len(vr_chunks)} VideoRag chunks")

    # Step 2: Group by movie
    movies_map = {}
    for chunk in vr_chunks:
        mid = chunk.get("movie_id", "unknown")
        if mid not in movies_map:
            movies_map[mid] = {"title": chunk.get("title", mid), "chunks": []}
        movies_map[mid]["chunks"].append(chunk)

    logger.info(f"Movies: {len(movies_map)}")
    for mid, mdata in sorted(movies_map.items()):
        logger.info(f"  {mid}: {mdata['title']} ({len(mdata['chunks'])} chunks)")

    if dry_run:
        logger.info("\n✅ Dry run complete — re-run with --execute to import")
        return

    # Step 3: Transform chunks
    print("\n📋 Transforming to 5-Layer schema...")
    transformed_chunks = []
    keyframe_copy_results = {}

    for movie_id, mdata in movies_map.items():
        title = mdata["title"]

        # Copy keyframes
        if copy_keyframes_flag:
            copied = copy_keyframes(movie_id, dry_run=False)
            keyframe_copy_results[movie_id] = copied

        # Transform each chunk
        for vr_chunk in mdata["chunks"]:
            tc = transform_videorag_chunk(vr_chunk, movie_title=title)

            # Update keyframe paths with new location
            old_paths = vr_chunk.get("keyframe_paths", [])
            if old_paths and keyframe_copy_results.get(movie_id):
                # Map old filename to new location
                old_filenames = {Path(p).name: p for p in old_paths}
                new_kf_paths = []
                for copied_path in keyframe_copy_results[movie_id]:
                    fname = Path(copied_path).name
                    if fname in old_filenames:
                        new_kf_paths.append(copied_path)
                tc["keyframe_paths"] = new_kf_paths
                tc["num_keyframes"] = len(new_kf_paths)

            transformed_chunks.append(tc)

    logger.info(f"Transformed {len(transformed_chunks)} chunks")

    # Step 4: Optional Groq L2-L5 enrichment
    if enrich:
        _enrich_with_groq(transformed_chunks)

    # Step 5: Save output
    OUTPUT_CHUNKS.mkdir(parents=True, exist_ok=True)
    OUTPUT_GRAPHS.mkdir(parents=True, exist_ok=True)

    out_file = OUTPUT_CHUNKS / "all_chunks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "source": "videorag",
                "version": "1.0",
                "num_chunks": len(transformed_chunks),
                "num_movies": len(movies_map),
                "movies": list(movies_map.keys()),
                "enriched": enrich,
            },
            "chunks": transformed_chunks,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved: {out_file} ({len(transformed_chunks)} chunks)")

    # Step 6: Build FAISS visual index
    if build_index:
        _build_visual_index(transformed_chunks, keyframe_copy_results)

    # Step 7: Stats
    _print_stats(transformed_chunks)


def _enrich_with_groq(chunks: list) -> None:
    """Enrich L2-L5 fields using Groq Llama 4 Scout VLM."""
    import os
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — skipping enrichment")
        return

    from groq import Groq
    client = Groq(api_key=api_key)

    logger.info(f"\n🔮 Groq L2-L5 Enrichment ({len(chunks)} chunks)...")

    # Process in batches
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        prompt = _build_enrichment_prompt(batch)
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a movie analyst. Return valid JSON array. For each chunk, fill L2-L5 fields.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            content = response.choices[0].message.content

            # Parse JSON response
            import re
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                enriched = json.loads(json_match.group())
                for j, chunk in enumerate(batch):
                    if j < len(enriched):
                        e = enriched[j]
                        # Merge
                        for k in ["emotional_tone", "narrative_arc", "vision_setting",
                                  "character_emotions", "causal_relations"]:
                            if k in e and e[k]:
                                chunks[i + j][k] = e[k]
            logger.info(f"  Enriched {min(i+batch_size, len(chunks))}/{len(chunks)}")
        except Exception as ex:
            logger.warning(f"  Batch {i}: {ex}")
            continue

    logger.info("  ✅ Groq enrichment complete")


def _build_enrichment_prompt(batch: list) -> str:
    """Build prompt for L2-L5 enrichment."""
    items = []
    for j, c in enumerate(batch):
        items.append(f"""Chunk {j}:
  ID: {c.get('chunk_id')}
  Movie: {c.get('title')}
  Description: {c.get('description')}
  Situation: {c.get('situation')}
  Characters: {c.get('characters')}
  Dialogue: {c.get('dialogue_text', '')[:200]}""")

    return f"""Analyze these movie scenes and return a JSON array with enriched metadata:

[
  {{
    "emotional_tone": "positive_happy|negative_sad|neutral|tense|positive_funny|...",
    "narrative_arc": "introduction|rising_action|climax|resolution|transition",
    "vision_setting": "beach|indoor_home|outdoor_urban|underwater|...",
    "character_emotions": {{"CharacterName": "emotion"}},
    "causal_relations": [{{"cause": "...", "effect": "..."}}]
  }}
]

Scenes:
{chr(10).join(items)}

Return ONLY the JSON array."""


def _build_visual_index(chunks: list, keyframe_results: dict) -> None:
    """Build CLIP FAISS visual index from real keyframes."""
    import faiss
    import torch
    import open_clip

    logger.info("\n🔢 Building CLIP visual index from real keyframes...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_meta = []

    for chunk in chunks:
        kf_paths = chunk.get("keyframe_paths", [])
        if not kf_paths:
            continue

        # Use first keyframe
        kf_path = Path(kf_paths[0])
        if not kf_path.exists():
            continue

        try:
            from PIL import Image
            img = Image.open(kf_path).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.encode_image(img_input)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb = emb.float().cpu().numpy()

            all_embeddings.append(emb.squeeze())
            all_meta.append({
                "chunk_id": chunk["chunk_id"],
                "movie_id": chunk["movie_id"],
                "title": chunk.get("title", ""),
                "description": chunk.get("description", ""),
                "keyframe_path": str(kf_path),
            })
        except Exception as e:
            logger.warning(f"  Failed on {kf_path}: {e}")
            continue

    if not all_embeddings:
        logger.warning("  No embeddings generated")
        return

    embeddings = np.array(all_embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    OUTPUT_INDEXES.mkdir(parents=True, exist_ok=True)
    idx_path = OUTPUT_INDEXES / "videorag_visual.faiss"
    meta_path = OUTPUT_INDEXES / "videorag_visual_map.json"

    faiss.write_index(index, str(idx_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)

    size_mb = idx_path.stat().st_size // (1024 * 1024)
    logger.info(f"  ✅ Visual index: {index.ntotal} vectors, {size_mb}MB → {idx_path}")


def _print_stats(chunks: list) -> None:
    """Print coverage statistics."""
    print("\n📊 Field Coverage:")
    all_keys = set()
    for c in chunks:
        all_keys.update(c.keys())

    # 5-Layer fields
    layer_fields = {
        "L1": ["chunk_id", "movie_id", "start_seconds", "end_seconds", "duration"],
        "L2": ["description", "situation", "vision_setting", "vision_actions", "emotional_tone"],
        "L3": ["text", "dialogue_text", "speaker", "audio_events", "background_music"],
        "L4": ["characters", "cast_in_scene", "character_emotions", "action_labels", "face_tracking_ids"],
        "L5": ["narrative_arc", "causal_relations", "scene_graph", "script_heading", "screenplay_context"],
    }

    for layer, fields in layer_fields.items():
        covered = sum(1 for f in fields if any(c.get(f) for c in chunks))
        pct = int(100 * covered / len(fields)) if fields else 0
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"  {layer}: [{bar}] {covered}/{len(fields)} fields ({pct}%)")

    # Real vs placeholder
    real_desc = sum(1 for c in chunks if c.get("description"))
    real_chars = sum(1 for c in chunks if c.get("characters"))
    real_dialogue = sum(1 for c in chunks if c.get("dialogue_text"))
    real_kf = sum(1 for c in chunks if c.get("keyframe_paths"))
    print(f"\n  Real data:")
    print(f"    Descriptions:     {real_desc}/{len(chunks)} ({int(100*real_desc/len(chunks))}%)")
    print(f"    Characters:       {real_chars}/{len(chunks)} ({int(100*real_chars/len(chunks))}%)")
    print(f"    Dialogue:         {real_dialogue}/{len(chunks)} ({int(100*real_dialogue/len(chunks))}%)")
    print(f"    Keyframes:       {real_kf}/{len(chunks)} ({int(100*real_kf/len(chunks))}%)")

    # Movies
    movies = set(c.get("movie_id") for c in chunks)
    print(f"\n  Movies: {len(movies)}")
    for mid in sorted(movies):
        cnt = sum(1 for c in chunks if c.get("movie_id") == mid)
        title = next((c.get("title", mid) for c in chunks if c.get("movie_id") == mid), mid)
        print(f"    {mid}: {title} ({cnt} chunks)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import VideoRag chunks into project_ky4 5-Layer schema"
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Show what would be imported without importing")
    parser.add_argument("--execute", action="store_true",
                        help="Actually import (default is dry-run)")
    parser.add_argument("--no-keyframes", dest="no_keyframes", action="store_true",
                        help="Skip keyframe copying")
    parser.add_argument("--enrich", action="store_true",
                        help="Also call Groq API for L2-L5 enrichment")
    parser.add_argument("--build-index", dest="build_index", action="store_true",
                        help="Build CLIP visual FAISS index")
    parser.add_argument("--videorag-root", type=Path,
                        default=VIDEORAG_ROOT,
                        help="Path to VideoRag project")

    args = parser.parse_args()
    dry_run = not args.execute

    # Override VideoRag root if provided
    vr_root = args.videorag_root
    vr_chunks_dir = vr_root / "data" / "temporal_chunks"
    vr_keyframes = vr_root / "data" / "trailer_keyframes"

    import_videorag(
        dry_run=dry_run,
        copy_keyframes_flag=not args.no_keyframes,
        enrich=args.enrich,
        build_index=args.build_index,
    )


if __name__ == "__main__":
    main()
