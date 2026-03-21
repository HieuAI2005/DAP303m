"""
FAISS Index Builder

Reads temporal chunks (all_chunks.json) and builds a FAISS visual index
with enriched metadata for each keyframe.

Adapted from: scripts/reindex_temporal.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

from preprocess_data.config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class FaissBuilder:
    """Build FAISS visual index from temporal chunks."""

    def __init__(self, index_dir: str = None):
        self.index_dir = Path(index_dir) if index_dir else Cfg.get_index_dir()
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _flatten_chunks(chunks: List[Dict]) -> List[Dict]:
        items = []
        for chunk in chunks:
            for kf_path in chunk.get("keyframe_paths", []):
                resolved_path = Cfg.resolve_keyframe_path(
                    str(chunk.get("movie_id", "")),
                    str(kf_path),
                )
                items.append(
                    {
                        "id": f"{chunk['chunk_id']}_{Path(kf_path).stem}",
                        "path": resolved_path,
                        "movie_id": chunk["movie_id"],
                        "chunk_id": chunk["chunk_id"],
                        "start_time": chunk.get("start_time", ""),
                        "end_time": chunk.get("end_time", ""),
                        "start_seconds": chunk.get("start_seconds", 0),
                        "end_seconds": chunk.get("end_seconds", 0),
                        "description": chunk.get("description", ""),
                        "dialogue_text": chunk.get("dialogue_text", ""),
                        "characters": chunk.get("characters", []),
                        "cast_in_scene": chunk.get("cast_in_scene", []),
                        "situation": chunk.get("situation", ""),
                        "scene_label": chunk.get("scene_label", ""),
                        "chunk_source": chunk.get("chunk_source", ""),
                        "parent_scene_id": chunk.get("parent_scene_id", ""),
                        "parent_clip_id": chunk.get("parent_clip_id", ""),
                        "script_scene_uid": chunk.get("script_scene_uid", ""),
                        "scene_type": chunk.get("scene_type", ""),
                        "environment": chunk.get("environment", ""),
                        "script_time_of_day": chunk.get("script_time_of_day", ""),
                        "character_type": chunk.get("character_type", ""),
                        "script_location": chunk.get("script_location", ""),
                        "script_characters": chunk.get("script_characters", []),
                        "script_scene_refs": chunk.get("script_scene_refs", []),
                        "script_scene_count": chunk.get("script_scene_count", 0),
                        "script_primary_heading": chunk.get("script_primary_heading", ""),
                        "script_headings": chunk.get("script_headings", []),
                        "dominant_script_scene_ref": chunk.get(
                            "dominant_script_scene_ref"
                        ),
                        "dominant_script_overlap_sec": chunk.get(
                            "dominant_script_overlap_sec", 0.0
                        ),
                        "alignment_confidence": chunk.get(
                            "alignment_confidence", 0.0
                        ),
                        "dialogue_excerpt": chunk.get("dialogue_excerpt", ""),
                        "subtitle_dialogue_excerpt": chunk.get(
                            "subtitle_dialogue_excerpt", ""
                        ),
                        "dialogue_source": chunk.get("dialogue_source", ""),
                        "dialogue_full_text": chunk.get("dialogue_full_text", ""),
                        "screenplay_action_excerpt": chunk.get(
                            "screenplay_action_excerpt", ""
                        ),
                        "screenplay_dialogue_turns": chunk.get(
                            "screenplay_dialogue_turns", []
                        ),
                        "screenplay_dialogue_excerpt": chunk.get(
                            "screenplay_dialogue_excerpt", ""
                        ),
                        "screenplay_context_excerpt": chunk.get(
                            "screenplay_context_excerpt", ""
                        ),
                        "screenplay_evidence": chunk.get("screenplay_evidence", ""),
                        "vision_setting": chunk.get("vision_setting", ""),
                        "vision_actions": chunk.get("vision_actions", ""),
                        "vision_objects": chunk.get("vision_objects", []),
                        "visual_focus": chunk.get("visual_focus", ""),
                        "vision_source": chunk.get("vision_source", ""),
                        "timestamp_source": chunk.get("timestamp_source", ""),
                        "title": chunk.get("title", ""),
                    }
                )
        return items

    def build(self, chunks_path: Path = None) -> Dict:
        """
        Read all_chunks.json, flatten to per-keyframe items,
        encode with CLIP, and build FAISS index.
        """
        chunks_path = chunks_path or (Cfg.get_temporal_chunks_dir() / "all_chunks.json")

        # Load chunks
        logger.info(f"Loading temporal chunks from: {chunks_path}")
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = data.get("chunks", []) if isinstance(data, dict) else data
        logger.info(f"  Total chunks: {len(chunks)}")

        # Flatten: one item per keyframe
        items = self._flatten_chunks(chunks)

        logger.info(f"  Total keyframe items: {len(items)}")
        if not items:
            logger.warning("  No items to index!")
            return {"items": 0}

        # Import indexer
        sys.path.insert(0, str(Cfg.SRC_DIR))
        from movierag.indexing.visual_indexer import VisualIndexer

        indexer = VisualIndexer(str(self.index_dir))
        indexer.build_index(items, id_key="id", path_key="path")
        indexer.save()
        scene_items = len(getattr(indexer, "_scene_metadata", []) or [])

        logger.info(
            f"  ✅ FAISS index built: {len(items)} frame vectors, {scene_items} scene vectors → {self.index_dir}"
        )
        return {
            "items": len(items),
            "scene_items": scene_items,
            "index_dir": str(self.index_dir),
        }

    def build_incremental(self, movie_id: str) -> Dict:
        """
        Add a single movie's chunks to the existing FAISS index.
        For new video ingest without rebuilding the entire index.
        """
        chunks_path = Cfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json"
        if not chunks_path.exists():
            logger.warning(f"No chunks found for {movie_id}")
            return {"items": 0}

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        items = self._flatten_chunks(chunks)

        if not items:
            return {"items": 0}

        sys.path.insert(0, str(Cfg.SRC_DIR))
        from movierag.indexing.visual_indexer import VisualIndexer

        indexer = VisualIndexer(str(self.index_dir))
        combined_items = {}

        # Rebuild from existing metadata plus the new movie's items.
        # Replace stale entries for the same movie instead of stacking old and new runs.
        if indexer.load():
            logger.info(
                f"  Loaded existing index, merging {len(items)} items for {movie_id}"
            )
            for meta in indexer._metadata:
                meta_item = dict(meta)
                if meta_item.get("movie_id") == movie_id:
                    continue
                item_id = meta_item.get("id")
                if item_id:
                    combined_items[item_id] = meta_item

        for item in items:
            combined_items[item["id"]] = item

        merged_items = list(combined_items.values())
        indexer.build_index(merged_items, id_key="id", path_key="path")
        indexer.save()
        total_scene_items = len(getattr(indexer, "_scene_metadata", []) or [])

        logger.info(
            f"  ✅ Visual index now stores {len(merged_items)} frame vectors and {total_scene_items} scene vectors after merging {movie_id}"
        )
        return {
            "items": len(items),
            "movie_id": movie_id,
            "total_items": len(merged_items),
            "total_scene_items": total_scene_items,
        }
