"""
Knowledge Indexer for MovieGraphs Text Search.

Uses CLIP text encoder + FAISS for semantic search over
textual descriptions extracted from MovieGraphs.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

from movierag.indexing.clip_encoder import CLIPEncoder

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TextSearchResult:
    """Result from text search."""

    movie_id: str
    clip_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class KnowledgeIndexer:
    """
    Indexes textual knowledge from MovieGraphs for semantic search.

    Uses CLIP text encoder for embeddings and FAISS for similarity search.
    Supports queries like "What happened in the scene with X and Y?"
    """

    def __init__(
        self,
        index_dir: str,
        index_name: str = "knowledge_index",
        encoder: Optional[CLIPEncoder] = None,
    ):
        """
        Initialize the knowledge indexer.

        Args:
            index_dir: Directory to store/load index files
            index_name: Base name for index files
            encoder: CLIPEncoder instance (creates new one if None)
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_name = index_name
        self.index_path = self.index_dir / f"{index_name}.faiss"
        self.metadata_path = self.index_dir / f"{index_name}_metadata.json"

        # Initialize encoder
        self.encoder = encoder or CLIPEncoder()
        self._st_encoder = None  # lazy-loaded SentenceTransformer for 384-dim indexes

        # Index state
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._is_loaded = False

    def _metadata_candidates(self) -> List[Path]:
        return [
            self.metadata_path,
            self.index_dir / f"{self.index_name}_map.json",
        ]

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the knowledge index from textual documents.

        Args:
            documents: List of dicts with 'text', 'movie_id', 'clip_id', 'metadata'
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS is not installed. Install with: pip install faiss-cpu"
            )

        logger.info(f"Building knowledge index from {len(documents)} documents...")

        # Extract texts and metadata
        texts = []
        self._metadata = []

        for doc in documents:
            text = doc.get("text", "")
            if not text.strip():
                continue

            texts.append(text)
            self._metadata.append(
                {
                    "movie_id": doc.get("movie_id", "unknown"),
                    "clip_id": doc.get("clip_id", "unknown"),
                    "text": text,
                    **doc.get("metadata", {}),
                }
            )

        if not texts:
            logger.error("No valid documents to index")
            return

        # Encode all texts
        logger.info(f"Encoding {len(texts)} text documents...")
        embeddings = self.encoder.encode_texts(texts, normalize=True)

        if len(embeddings) == 0:
            logger.error("No texts could be encoded")
            return

        # Build FAISS index
        dim = embeddings.shape[1]
        logger.info(f"Building FAISS index with {len(embeddings)} vectors of dim {dim}")

        # Use IndexFlatIP for cosine similarity (vectors are normalized)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))

        self._is_loaded = True

        # Save index
        self.save()

        logger.info(f"Knowledge index built with {self._index.ntotal} documents")

    def save(self) -> None:
        """Save index and metadata to disk."""
        if self._index is None:
            logger.warning("No index to save")
            return

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))
        logger.info(f"Saved FAISS index to {self.index_path}")

        # Save metadata
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata to {self.metadata_path}")

    def load(self) -> bool:
        """Load index and metadata from disk."""
        if not FAISS_AVAILABLE or faiss is None:
            raise RuntimeError(
                "FAISS is not installed in the active environment. Install `faiss-cpu` or `faiss-gpu`."
            )
        metadata_path = next(
            (path for path in self._metadata_candidates() if path.exists()),
            None,
        )
        if not self.index_path.exists() or metadata_path is None:
            logger.warning(f"Index files not found at {self.index_dir}")
            return False

        # Load FAISS index
        self._index = faiss.read_index(str(self.index_path))
        logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors (dim={self._index.d})")

        # If index is 384-dim (SentenceTransformer), load ST encoder for query encoding
        if self._index.d == 384:
            if _ST_AVAILABLE and self._st_encoder is None:
                logger.info("Index is 384-dim — using SentenceTransformer for query encoding")
                self._st_encoder = _SentenceTransformer("all-MiniLM-L6-v2")
            elif not _ST_AVAILABLE:
                logger.warning("Index is 384-dim but sentence_transformers not installed; search may fail")

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        if metadata_path != self.metadata_path:
            logger.info(f"Loaded metadata fallback from {metadata_path}")

        self._is_loaded = True
        return True

    def ensure_loaded(self) -> None:
        """Ensure index is loaded."""
        if not self._is_loaded:
            if not self.load():
                raise RuntimeError("Index not found. Build index first.")

    def search(
        self,
        query: str,
        k: int = 10,
        movie_id: Optional[str] = None,
    ) -> List[TextSearchResult]:
        """
        Search for documents matching the query.

        Args:
            query: Natural language query
            k: Number of results to return
            movie_id: Optional filter to search within a specific movie

        Returns:
            List of TextSearchResult objects
        """
        self.ensure_loaded()

        # Encode query using the appropriate encoder based on index dimension
        if self._st_encoder is not None and self._index is not None and self._index.d == 384:
            emb = self._st_encoder.encode([query], normalize_embeddings=True)
            query_embedding = emb.astype(np.float32)
        else:
            query_embedding = self.encoder.encode_text(query, normalize=True)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search — scan larger pool when filtering by movie_id so sparse movies
        # are represented even when many other chunks dominate global top-k.
        search_k = min(k * 50, self._index.ntotal) if movie_id else k
        distances, indices = self._index.search(
            query_embedding, min(search_k, self._index.ntotal)
        )

        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            metadata = self._metadata[idx]

            # Filter by movie_id if specified; skip generic trailer-only chunks
            # when annotation chunks for the same movie exist in the result set.
            if movie_id and metadata.get("movie_id") != movie_id:
                continue

            # Prefer annotation chunks over content-sparse trailer placeholders
            text = metadata.get("text", "")
            if text in ("", "Trailer scene 1 | [NO_DIALOGUE] | trailer") and len(results) >= k:
                continue

            results.append(
                TextSearchResult(
                    movie_id=metadata.get("movie_id", "unknown"),
                    clip_id=metadata.get("clip_id") or metadata.get("chunk_id", f"idx_{idx}"),
                    text=text,
                    score=float(distances[0][i]),
                    metadata=metadata,
                )
            )

            if len(results) >= k:
                break

        return results

    def search_multi(
        self,
        queries: List[str],
        k: int = 5,
    ) -> Dict[str, List[TextSearchResult]]:
        """
        Search for multiple queries at once.

        Args:
            queries: List of query strings
            k: Number of results per query

        Returns:
            Dict mapping query to list of results
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, k=k)
        return results

    @property
    def num_documents(self) -> int:
        """Get number of indexed documents."""
        if self._index is None:
            return 0
        return self._index.ntotal

    @staticmethod
    def _compose_chunk_text(chunk: dict) -> str:
        """Compose searchable text from a temporal chunk dict."""
        parts = []
        if chunk.get("description"):
            parts.append(str(chunk["description"]).strip())
        if chunk.get("situation"):
            parts.append(str(chunk["situation"]).strip())
        dlg = chunk.get("dialogue_text") or chunk.get("dialogue") or ""
        if dlg and dlg not in ("[PENDING_SRT_ALIGNMENT]", "[NO_DIALOGUE]"):
            parts.append(str(dlg).strip())
        chars = chunk.get("characters") or []
        if isinstance(chars, list) and chars:
            parts.append(" ".join(str(c) for c in chars))
        if chunk.get("vision_setting"):
            parts.append(str(chunk["vision_setting"]).strip())
        if chunk.get("narrative_arc"):
            parts.append(str(chunk["narrative_arc"]).strip())
        return " | ".join(p for p in parts if p)

    def build_incremental(self, movie_id: str, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Add or replace a single movie's chunks in the existing knowledge index.

        Removes any prior entries for *movie_id*, encodes the new chunks with
        the SentenceTransformer encoder, appends them, and writes the updated
        index + metadata back to disk.

        Returns a dict with ``{"added": N, "total": M}``.
        """
        if not FAISS_AVAILABLE or faiss is None:
            logger.warning("FAISS unavailable — skipping knowledge index update for %s", movie_id)
            return {"added": 0, "total": 0}

        if not chunks:
            logger.warning("No chunks provided for %s — skipping knowledge index update", movie_id)
            return {"added": 0, "total": 0}

        # Load existing index (or start fresh if it doesn't exist yet)
        already_loaded = self._is_loaded
        if not already_loaded:
            self.load()  # best-effort; may return False without crashing

        # Determine encoder and dim
        if self._st_encoder is None and _ST_AVAILABLE:
            self._st_encoder = _SentenceTransformer("all-MiniLM-L6-v2")

        index_dim = 384  # SentenceTransformer all-MiniLM-L6-v2

        # Build new entries
        new_texts: List[str] = []
        new_meta: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = chunk.get("text") or self._compose_chunk_text(chunk)
            if not text.strip():
                continue
            meta: Dict[str, Any] = {
                "chunk_id": chunk.get("chunk_id", ""),
                "movie_id": chunk.get("movie_id", movie_id),
                "title": chunk.get("title", ""),
                "text": text,
                "description": chunk.get("description", ""),
                "vision_setting": chunk.get("vision_setting", ""),
                "vision_actions": chunk.get("vision_actions", []),
                "emotional_tone": chunk.get("emotional_tone", ""),
                "situation": chunk.get("situation", ""),
                "dialogue_text": chunk.get("dialogue_text") or chunk.get("dialogue", ""),
                "speaker": chunk.get("speaker", ""),
                "audio_events": chunk.get("audio_events", ""),
                "characters": chunk.get("characters", []),
                "cast_in_scene": chunk.get("cast_in_scene", []),
                "narrative_arc": chunk.get("narrative_arc", ""),
                "causal_relations": chunk.get("causal_relations", []),
                "screenplay_context": chunk.get("screenplay_context", ""),
                "source": chunk.get("source", "ingest"),
                "start_seconds": chunk.get("start_seconds"),
                "end_seconds": chunk.get("end_seconds"),
                "keyframe_paths": chunk.get("keyframe_paths", []),
            }
            new_texts.append(text)
            new_meta.append(meta)

        if not new_texts:
            logger.warning("No valid texts in chunks for %s", movie_id)
            return {"added": 0, "total": self.num_documents}

        # Encode new texts
        if self._st_encoder is not None:
            import numpy as _np
            new_vecs = self._st_encoder.encode(new_texts, normalize_embeddings=True).astype(_np.float32)
        else:
            logger.warning("SentenceTransformer not available — cannot update knowledge index")
            return {"added": 0, "total": self.num_documents}

        # Keep existing entries that belong to OTHER movies
        if self._index is not None and self._metadata:
            import numpy as _np
            existing_dim = self._index.d
            keep_meta = [m for m in self._metadata if m.get("movie_id") != movie_id]
            keep_indices = [i for i, m in enumerate(self._metadata) if m.get("movie_id") != movie_id]
            if keep_indices:
                kept_vecs = _np.zeros((len(keep_indices), existing_dim), dtype=_np.float32)
                for j, idx in enumerate(keep_indices):
                    self._index.reconstruct(int(idx), kept_vecs[j])
                all_vecs = _np.vstack([kept_vecs, new_vecs])
            else:
                all_vecs = new_vecs
                keep_meta = []
        else:
            import numpy as _np
            all_vecs = new_vecs
            keep_meta = []

        # Rebuild index
        new_index = faiss.IndexFlatIP(index_dim)
        new_index.add(all_vecs)
        self._index = new_index
        self._metadata = keep_meta + new_meta
        self._is_loaded = True

        # Persist to disk
        self.save()

        added = len(new_meta)
        total = new_index.ntotal
        logger.info(
            "Knowledge index updated: +%d chunks for %s → %d total vectors", added, movie_id, total
        )
        return {"added": added, "total": total}

    def build_from_loaders(
        self,
        unified_loader=None,
        subtitle_loader=None,
    ) -> None:
        """
        Convenience method to build the knowledge index from all available data loaders.

        Args:
            unified_loader: UnifiedLoader instance (provides plot, cast, script, subtitle docs)
            subtitle_loader: SubtitleLoader instance (provides timestamp-aligned dialog docs)
        """
        all_docs = []

        if unified_loader is not None:
            docs = unified_loader.get_all_textual_documents()
            logger.info(f"Collected {len(docs)} documents from UnifiedLoader")
            all_docs.extend(docs)

        if subtitle_loader is not None:
            docs = subtitle_loader.get_all_textual_documents()
            logger.info(f"Collected {len(docs)} documents from SubtitleLoader")
            all_docs.extend(docs)

        if not all_docs:
            logger.error("No documents collected from any loader")
            return

        logger.info(
            f"Building unified knowledge index from {len(all_docs)} total documents"
        )
        self.build_index(all_docs)
