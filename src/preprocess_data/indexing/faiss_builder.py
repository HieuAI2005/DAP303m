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

        Uses FAISS reconstruct() to preserve existing vectors for items that
        have no image path (e.g. ActivityNet frames), instead of re-encoding
        the entire index from disk images.
        """
        chunks_path = Cfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json"
        if not chunks_path.exists():
            logger.warning(f"No chunks found for {movie_id}")
            return {"items": 0}

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        new_items = self._flatten_chunks(chunks)

        if not new_items:
            return {"items": 0}

        sys.path.insert(0, str(Cfg.SRC_DIR))
        import numpy as np
        import faiss as faiss_lib
        from movierag.indexing.visual_indexer import VisualIndexer

        indexer = VisualIndexer(str(self.index_dir))

        if not indexer.load():
            # No existing index — build fresh from new items only
            indexer.build_index(new_items, id_key="id", path_key="path")
            indexer.save()
            total_scene_items = len(getattr(indexer, "_scene_metadata", []) or [])
            logger.info(
                f"  ✅ Visual index built with {len(new_items)} frame vectors for {movie_id}"
            )
            return {
                "items": len(new_items),
                "movie_id": movie_id,
                "total_items": len(new_items),
                "total_scene_items": total_scene_items,
            }

        # Existing index loaded — reconstruct vectors for items we keep,
        # then encode only the new movie's items from images.
        existing_dim = indexer._index.d
        keep_indices = []   # FAISS vector positions in existing index
        keep_metadata = []  # metadata dicts for kept items

        for i, meta in enumerate(indexer._metadata):
            if meta.get("movie_id") == movie_id:
                continue  # drop stale entries for this movie
            keep_indices.append(i)
            keep_metadata.append(dict(meta))

        logger.info(
            f"  Reconstructing {len(keep_indices)} existing vectors, "
            f"encoding {len(new_items)} new items for {movie_id}"
        )

        # Reconstruct existing vectors from FAISS (avoids re-encoding from images)
        if keep_indices:
            existing_vecs = np.zeros((len(keep_indices), existing_dim), dtype=np.float32)
            for j, faiss_idx in enumerate(keep_indices):
                indexer._index.reconstruct(int(faiss_idx), existing_vecs[j])
        else:
            existing_vecs = np.zeros((0, existing_dim), dtype=np.float32)

        # Encode only the new movie's keyframes
        new_paths = [item["path"] for item in new_items]
        new_vecs = indexer.encoder.encode_images(new_paths, normalize=True)

        if len(new_vecs) == 0:
            logger.warning(f"  No new vectors encoded for {movie_id}")
            return {"items": 0}

        # Build new metadata list
        new_metadata = []
        for item in new_items:
            meta = {
                "id": item["id"],
                "path": item["path"],
                "movie_id": item.get("movie_id", "unknown"),
                **{k: v for k, v in item.items() if k not in ("id", "path", "movie_id")},
            }
            new_metadata.append(meta)

        all_metadata = keep_metadata + new_metadata

        # Combine vectors and build new flat index
        if len(existing_vecs) > 0:
            all_vecs = np.vstack([existing_vecs, new_vecs]).astype(np.float32)
        else:
            all_vecs = new_vecs.astype(np.float32)

        new_index = faiss_lib.IndexFlatIP(existing_dim)
        new_index.add(all_vecs)

        indexer._index = new_index
        indexer._metadata = all_metadata
        indexer.save()
        total_scene_items = len(getattr(indexer, "_scene_metadata", []) or [])

        logger.info(
            f"  ✅ Visual index now stores {len(all_metadata)} frame vectors and "
            f"{total_scene_items} scene vectors after merging {movie_id}"
        )

        # Also update the knowledge (text) index with the new movie's chunks
        knowledge_result = self._update_knowledge_index(movie_id)

        return {
            "items": len(new_items),
            "movie_id": movie_id,
            "total_items": len(all_metadata),
            "total_scene_items": total_scene_items,
            "knowledge_added": knowledge_result.get("added", 0),
            "knowledge_total": knowledge_result.get("total", 0),
        }

    def _update_knowledge_index(self, movie_id: str) -> Dict:
        """Update the knowledge (SentenceTransformer text) FAISS index for movie_id."""
        chunks_path = Cfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json"
        if not chunks_path.exists():
            logger.warning(f"  No chunks file for {movie_id} — knowledge index not updated")
            return {"added": 0, "total": 0}

        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            if isinstance(chunks, dict):
                chunks = chunks.get("chunks", [])
        except Exception as exc:
            logger.warning(f"  Failed to load chunks for knowledge index update: {exc}")
            return {"added": 0, "total": 0}

        try:
            from movierag.indexing.knowledge_indexer import KnowledgeIndexer

            ki = KnowledgeIndexer(
                index_dir=str(self.index_dir),
                index_name="knowledge_videorag",
            )
            result = ki.build_incremental(movie_id, chunks)
            logger.info(
                f"  ✅ Knowledge index updated: +{result.get('added', 0)} chunks "
                f"→ {result.get('total', 0)} total for {movie_id}"
            )
            return result
        except Exception as exc:
            logger.warning(f"  Knowledge index update failed for {movie_id}: {exc}")
            return {"added": 0, "total": 0}
