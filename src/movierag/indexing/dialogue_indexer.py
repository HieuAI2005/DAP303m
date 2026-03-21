import os
import json
import logging
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

try:
    from google import genai
except ImportError:
    genai = None

from movierag.config import get_config

logger = logging.getLogger(__name__)


class DialogueIndexer:
    """
    Ingests and embeds granular dialogue from annotation JSONs (.sentences),
    subtitles (.srt), and scripts (.script).
    Using sentence-transformers or Gemini Embeddings for FAISS semantic search.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.config = get_config()
        # Fallback to the known Gemini key if not in env
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        # Target index files
        self.index_path = self.config.paths.index_dir / "movie_dialogue_index.faiss"
        self.meta_path = self.config.paths.index_dir / "movie_dialogue_meta.json"

        # Initialize Embedding Model (Gemini text-embedding-004)
        if genai and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.embed_model = "text-embedding-004"
            self.dim_size = 768  # text-embedding-004 dimension
            logger.info("DialogueIndexer initialized with Gemini Embeddings.")
        else:
            logger.error("No Gemini API key found for DialogueIndexer. Cannot embed.")
            self.client = None
            self.dim_size = 768

        self.index = None
        self.metadata = []
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Loads FAISS index from disk or creates a new one."""
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(
                    f"Loaded Dialogue Index with {self.index.ntotal} vectors from {self.index_path}"
                )
            except Exception as e:
                logger.warning(f"Error loading index: {e}. Recreating...")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        self.index = faiss.IndexFlatIP(self.dim_size)
        self.metadata = []

    def save(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Saved Dialogue Index with {self.index.ntotal} vectors to {self.index_path}"
        )

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Get Gemini embeddings for a batch of texts."""
        if not self.client:
            return np.zeros((len(texts), self.dim_size), dtype=np.float32)

        try:
            import time

            time.sleep(0.5)  # Prevent rate limit
            response = self.client.models.embed_content(
                model=self.embed_model,
                contents=texts,
            )
            embeddings = [emb.values for emb in response.embeddings]
            vecs = np.array(embeddings, dtype=np.float32)

            # Normalize for Inner Product (Cosine Similarity)
            faiss.normalize_L2(vecs)
            return vecs

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return np.zeros((len(texts), self.dim_size), dtype=np.float32)

    def extract_dialogues_from_json(
        self, json_path: str, movie_id: str
    ) -> List[Dict[str, Any]]:
        """
        Parses annotation JSON structure which has shots and sentences.
        Example item: {"shot": 1531, "duration": [6995, 6997], "sentences": ["[grunts]"]}
        """
        chunks = []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "cast" in data:
                # Need to find the shots list, the schema is likely deeply nested or adjacent
                pass

            # Full annotation file parsing logic depending on user schema
            # Let's read file as a long list if it has objects with 'sentences'

            # Simple recursive search for objects with 'sentences' key
            def find_dialogues(node):
                if isinstance(node, dict):
                    if "sentences" in node and isinstance(node["sentences"], list):
                        text = " ".join(node["sentences"]).strip()
                        if text:
                            chunks.append(
                                {
                                    "movie_id": movie_id,
                                    "shot_id": str(node.get("shot", "")),
                                    "text": text,
                                    "start_time": node.get("duration", [0, 0])[0],
                                    "end_time": node.get("duration", [0, 0])[1]
                                    if len(node.get("duration", [])) > 1
                                    else 0,
                                }
                            )
                    for v in node.values():
                        find_dialogues(v)
                elif isinstance(node, list):
                    for item in node:
                        find_dialogues(item)

            find_dialogues(data)

        except Exception as e:
            logger.error(f"Error parsing {json_path}: {e}")

        return chunks

    def process_all_annotations(self):
        """Iterates over all JSON files in annotation_dir to build DB."""
        if not self.config.paths.annotation_dir.exists():
            logger.error("Annotation directory not found.")
            return

        all_chunks = []
        for file_path in self.config.paths.annotation_dir.glob("*.json"):
            movie_id = file_path.stem
            chunks = self.extract_dialogues_from_json(str(file_path), movie_id)
            all_chunks.extend(chunks)
            logger.info(f"Parsed {len(chunks)} dialog chunks from {movie_id}")

        if not all_chunks:
            logger.warning("No dialog chunks extracted.")
            return

        # Batch embed and insert
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            if not texts:
                continue

            try:
                embeddings = self._embed_texts(texts)
                self.index.add(embeddings)
                self.metadata.extend(batch)
                logger.info(
                    f"Embedded batch {i // batch_size + 1}, total: {len(self.metadata)}"
                )
            except Exception as e:
                logger.error(f"Failed to index batch {i}: {e}")

        self.save()
        logger.info(
            f"✨ Dialogue ingestion complete! Total vectors: {self.index.ntotal}"
        )

    def search(
        self, query: str, k: int = 5, movie_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search the dialogue index."""
        if not self.index or self.index.ntotal == 0:
            return []

        try:
            vec = self._embed_texts([query])

            # Request more candidates if filtering by movie_id
            search_k = k * 10 if movie_id else k
            scores, indices = self.index.search(vec, search_k)

            results = []
            for j, i in enumerate(indices[0]):
                if i != -1:
                    item = self.metadata[i].copy()

                    # Filter by movie_id if provided
                    if movie_id and item.get("movie_id") != movie_id:
                        continue

                    item["score"] = float(scores[0][j])
                    results.append(item)

                    if len(results) >= k:
                        break

            return results
        except Exception as e:
            logger.error(f"Dialogue search failed: {e}")
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    indexer = DialogueIndexer()
    indexer.process_all_annotations()
