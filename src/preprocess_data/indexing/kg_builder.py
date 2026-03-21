"""
Knowledge Graph Builder (Cross-Modal)

Fuses Visual Narratives (from VLM) and Movie Transcripts to build a 
semantic Knowledge Graph using Gemini.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from preprocess_data.config import PreprocessConfig as Cfg
from movierag.indexing.graph_indexer import GraphIndexer

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self, movie_id: str):
        self.movie_id = movie_id
        self.index_dir = Cfg.get_index_dir()
        self.indexer = GraphIndexer(
            index_dir=str(self.index_dir),
            index_name=f"{movie_id}_kg"
        )

    def build(self) -> bool:
        """Execute cross-modal KG construction."""
        logger.info(f"🚀 Building Cross-Modal KG for {self.movie_id}...")
        
        # 1. Load Visual Narratives (Step 6a output)
        vlm_path = Cfg.get_shot_keyf_dir() / self.movie_id / "vlm_temporal_descriptions.json"
        if not vlm_path.exists():
            logger.error(f"  ❌ Visual narratives not found at {vlm_path}")
            return False
            
        with open(vlm_path, "r", encoding="utf-8") as f:
            vlm_data = json.load(f)
            visual_scenes = vlm_data.get("scenes", {})

        # 2. Load Temporal Chunks (Step 7 output) to get transcripts
        chunks_path = Cfg.get_temporal_chunks_dir() / f"{self.movie_id}_chunks.json"
        if not chunks_path.exists():
            logger.error(f"  ❌ Temporal chunks not found at {chunks_path}")
            return False
            
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # 3. Fuse data per scene
        fused_scene_data = []
        for i, chunk in enumerate(chunks):
            # We assume chunk index matches scene order or we use scene_id/clip_id
            # For new videos, chunk_id is f"{movie_id}_chunk_{idx:04d}"
            # VLM descriptions use scene_id keys (e.g. "scene_0")
            
            scene_id = chunk.get("scene_label") or f"scene_{i}"
            if scene_id not in visual_scenes:
                # Try fallback keys
                scene_id = f"scene_{i}"
            
            visual_entry = visual_scenes.get(scene_id, {}) or {}
            visual_desc = visual_entry.get("actions", "")
            setting = visual_entry.get("setting", "")
            objects = visual_entry.get("objects", []) or []
            if isinstance(objects, str):
                objects = [objects]
            cast_characters = [
                cast.get("character", "")
                for cast in (chunk.get("cast_in_scene", []) or [])
                if cast.get("character")
            ]
            chunk_characters = []
            for name in (
                (chunk.get("characters", []) or [])
                + (chunk.get("script_characters", []) or [])
                + cast_characters
            ):
                name = str(name or "").strip()
                if name and name not in chunk_characters:
                    chunk_characters.append(name)
             
            fused_scene_data.append({
                "scene_idx": i,
                "chunk_id": chunk.get("chunk_id", f"{self.movie_id}_chunk_{i:04d}"),
                "scene_label": chunk.get("scene_label", ""),
                "script_primary_heading": chunk.get("script_primary_heading", ""),
                "script_location": chunk.get("script_location", ""),
                "visual_description": f"{setting}. {visual_desc}",
                "transcript": chunk.get("dialogue_full_text", "") or chunk.get("dialogue_text", ""),
                "movie_id": self.movie_id,
                "characters": chunk_characters,
                "script_characters": chunk.get("script_characters", []) or [],
                "cast_in_scene": chunk.get("cast_in_scene", []) or [],
                "vision_setting": chunk.get("vision_setting", "") or setting,
                "vision_actions": chunk.get("vision_actions", "") or visual_desc,
                "vision_objects": chunk.get("vision_objects", []) or objects,
                "screenplay_context_excerpt": chunk.get("screenplay_context_excerpt", ""),
                "description": chunk.get("description", ""),
                "situation": chunk.get("situation", ""),
            })

        # 4. Build Index
        self.indexer.build_cross_modal_index(fused_scene_data)
        
        logger.info(f"  ✅ KG construction complete for {self.movie_id}")
        return True
