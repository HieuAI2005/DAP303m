"""
Dedicated text index for screenplay-derived script sub-scenes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from preprocess_data.config import PreprocessConfig as Cfg
from movierag.indexing.knowledge_indexer import KnowledgeIndexer, TextSearchResult

logger = logging.getLogger(__name__)


class ScriptSceneIndexer:
    """Semantic text retrieval over derived script sub-scenes."""

    def __init__(
        self,
        index_dir: str,
        index_name: str = "script_scene_index",
        encoder=None,
    ):
        self._indexer = KnowledgeIndexer(
            index_dir=index_dir, index_name=index_name, encoder=encoder
        )
        self.index_dir = self._indexer.index_dir
        self.index_name = index_name
        self.index_path = self._indexer.index_path
        self.metadata_path = self._indexer.metadata_path

    @property
    def _is_loaded(self) -> bool:
        return self._indexer._is_loaded

    @property
    def _metadata(self) -> List[Dict[str, Any]]:
        return self._indexer._metadata

    def load(self) -> bool:
        return self._indexer.load()

    def ensure_loaded(self) -> None:
        self._indexer.ensure_loaded()

    def search(
        self, query: str, k: int = 6, movie_id: Optional[str] = None
    ) -> List[TextSearchResult]:
        search_k = max(k * 8, 24)
        base_results = self._search_with_recovery(
            query=query,
            k=search_k,
            movie_id=movie_id,
        )
        candidate_results: Dict[str, TextSearchResult] = {
            result.clip_id: result for result in base_results
        }

        for metadata in self._get_metadata_rows():
            if movie_id and metadata.get("movie_id") != movie_id:
                continue
            bonus = self._lexical_bonus(query, metadata)
            if bonus < 0.35:
                continue
            clip_id = metadata.get("clip_id", "")
            if not clip_id:
                continue
            candidate_results.setdefault(
                clip_id,
                TextSearchResult(
                    movie_id=metadata.get("movie_id", "unknown"),
                    clip_id=clip_id,
                    text=metadata.get("text", ""),
                    score=0.0,
                    metadata=metadata,
                ),
            )

        if not candidate_results:
            return []

        reranked = sorted(
            candidate_results.values(),
            key=lambda result: (
                result.score
                + self._lexical_bonus(query, result.metadata)
                + self._alignment_bonus(result.metadata),
                self._lexical_bonus(query, result.metadata)
                + self._alignment_bonus(result.metadata),
                result.score,
            ),
            reverse=True,
        )
        return reranked[:k]

    def _search_with_recovery(
        self, query: str, k: int, movie_id: Optional[str]
    ) -> List[TextSearchResult]:
        try:
            return self._indexer.search(query, k=k, movie_id=movie_id)
        except Exception as exc:
            logger.error(
                "Script-scene semantic search failed; attempting reload + lexical fallback: %s",
                exc,
                exc_info=True,
            )
            self._reset_loaded_state()
            try:
                self.load()
                return self._indexer.search(query, k=k, movie_id=movie_id)
            except Exception as retry_exc:
                logger.error(
                    "Script-scene search retry failed; continuing with lexical-only candidates: %s",
                    retry_exc,
                    exc_info=True,
                )
                return []

    def _reset_loaded_state(self) -> None:
        self._indexer._index = None
        self._indexer._metadata = []
        self._indexer._is_loaded = False

    def _get_metadata_rows(self) -> List[Dict[str, Any]]:
        if self._indexer._metadata:
            return self._indexer._metadata
        try:
            self._indexer.ensure_loaded()
            return self._indexer._metadata
        except Exception as exc:
            logger.warning(
                "Script-scene index metadata not available in-memory; loading JSON metadata directly: %s",
                exc,
            )
        try:
            if self.metadata_path.exists():
                return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception as meta_exc:
            logger.error(
                "Failed to load script-scene metadata JSON fallback: %s",
                meta_exc,
                exc_info=True,
            )
        return []

    def build_incremental(self, movie_id: str) -> Dict[str, Any]:
        subscene_path = Cfg.get_script_subscenes_dir() / f"{movie_id}_script_subscenes.json"
        if not subscene_path.exists():
            logger.warning(f"No script sub-scenes found for {movie_id}")
            return {"items": 0}

        try:
            subscenes = json.loads(subscene_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to read script sub-scenes for {movie_id}: {exc}")
            return {"items": 0}

        documents = [
            self._to_document(subscene)
            for subscene in subscenes
            if subscene.get("indexable")
        ]
        documents = [doc for doc in documents if doc]
        if not documents:
            logger.warning(f"No indexable script sub-scenes found for {movie_id}")
            return {"items": 0, "movie_id": movie_id}

        combined_docs: Dict[str, Dict[str, Any]] = {}
        if self._indexer.load():
            for metadata in self._indexer._metadata:
                if metadata.get("movie_id") == movie_id:
                    continue
                clip_id = metadata.get("clip_id")
                if not clip_id:
                    continue
                combined_docs[clip_id] = {
                    "movie_id": metadata.get("movie_id", "unknown"),
                    "clip_id": clip_id,
                    "text": metadata.get("text", ""),
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key not in {"movie_id", "clip_id", "text"}
                    },
                }

        for document in documents:
            combined_docs[document["clip_id"]] = document

        merged_documents = list(combined_docs.values())
        self._indexer.build_index(merged_documents)
        logger.info(
            "Built script-scene index with %s total vectors after merging %s",
            len(merged_documents),
            movie_id,
        )
        return {
            "items": len(documents),
            "movie_id": movie_id,
            "total_items": len(merged_documents),
        }

    @staticmethod
    def _compose_text(subscene: Dict[str, Any]) -> str:
        characters = ", ".join(subscene.get("script_characters", [])[:8])
        screenplay_turns = subscene.get("screenplay_dialogue_turns", []) or []
        screenplay_turns_text = " | ".join(screenplay_turns[:6])
        pieces = [
            f"Heading: {subscene.get('script_heading', '')}",
            f"Heading exact: {subscene.get('script_heading', '')}",
            f"Location: {subscene.get('script_location', '')}",
            f"Location exact: {subscene.get('script_location', '')}",
            f"Time of day: {subscene.get('script_time_of_day', '')}",
            f"Characters: {characters}",
            f"Alignment confidence: {subscene.get('alignment_confidence', subscene.get('confidence_score', 0.0))}",
            f"Screenplay action: {subscene.get('screenplay_action_excerpt', '')}",
            f"Screenplay dialogue: {subscene.get('screenplay_dialogue_excerpt', '')}",
            f"Screenplay context: {subscene.get('screenplay_context_excerpt', '')}",
            f"Screenplay turns: {screenplay_turns_text}",
            f"Dialogue: {subscene.get('dialogue_excerpt', '')}",
            f"Semantic description: {subscene.get('semantic_description', '')}",
            f"Semantic label: {subscene.get('semantic_scene_label', '')}",
        ]
        return "\n".join(piece for piece in pieces if piece.strip())

    @staticmethod
    def _normalize_text(value: str) -> str:
        import re

        return re.sub(r"[^\w\s]", " ", str(value or "").lower()).strip()

    @classmethod
    def _lexical_bonus(cls, query: str, metadata: Dict[str, Any]) -> float:
        query_norm = cls._normalize_text(query)
        if not query_norm:
            return 0.0

        heading = cls._normalize_text(metadata.get("script_heading", ""))
        location = cls._normalize_text(metadata.get("script_location", ""))
        time_of_day = cls._normalize_text(metadata.get("script_time_of_day", ""))
        characters = cls._normalize_text(
            " ".join(metadata.get("script_characters", []) or [])
        )
        action_excerpt = cls._normalize_text(
            metadata.get("screenplay_action_excerpt", "")
        )
        dialogue_excerpt = cls._normalize_text(
            metadata.get("screenplay_dialogue_excerpt", "")
        )
        context_excerpt = cls._normalize_text(
            metadata.get("screenplay_context_excerpt", "")
        )
        semantic_label = cls._normalize_text(metadata.get("semantic_scene_label", ""))
        combined = " ".join(
            part
            for part in (
                heading,
                location,
                time_of_day,
                characters,
                action_excerpt,
                dialogue_excerpt,
                context_excerpt,
                semantic_label,
            )
            if part
        )
        if not combined:
            return 0.0

        query_tokens = [token for token in query_norm.split() if token]
        combined_tokens = set(combined.split())
        bonus = 0.0

        if query_norm in heading or query_norm in location:
            bonus += 1.20
        elif query_norm in combined:
            bonus += 0.75

        if query_tokens:
            overlap = sum(1 for token in query_tokens if token in combined_tokens)
            bonus += 0.30 * (overlap / len(query_tokens))

        if any(token in {"night", "day", "morning", "evening", "afternoon"} for token in query_tokens):
            if any(token in time_of_day for token in query_tokens):
                bonus += 0.30

        if any(token in {"who", "ai", "character", "appears", "xuat", "present"} for token in query_tokens):
            char_overlap = sum(1 for token in query_tokens if token in characters.split())
            if char_overlap:
                bonus += 0.25

        if any(token in {"quote", "dialogue", "line", "noi", "said", "says"} for token in query_tokens):
            if any(token in dialogue_excerpt for token in query_tokens):
                bonus += 0.20

        return bonus

    @staticmethod
    def _alignment_bonus(metadata: Dict[str, Any]) -> float:
        anchor_quality = str(metadata.get("anchor_quality", "linear") or "linear")
        confidence = float(
            metadata.get(
                "alignment_confidence", metadata.get("confidence_score", 0.0)
            )
            or 0.0
        )
        bonus = min(0.30, confidence * 0.25)
        if anchor_quality == "full":
            bonus += 0.18
        elif anchor_quality == "partial":
            bonus += 0.10
        elif confidence < 0.35:
            bonus -= 0.05
        return bonus

    def _to_document(self, subscene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        subscene_id = subscene.get("subscene_id")
        if not subscene_id:
            return None

        text = self._compose_text(subscene).strip()
        if not text:
            return None

        metadata = dict(subscene)
        metadata["category"] = "script_subscene"
        metadata["title"] = subscene.get("movie_id", "")

        return {
            "movie_id": subscene.get("movie_id", "unknown"),
            "clip_id": subscene_id,
            "text": text,
            "metadata": metadata,
        }
