import os
import json
import logging
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from movierag.indexing.visual_indexer import VisualIndexer

logger = logging.getLogger(__name__)


def _map_encode_chunk(
    chunk_items: List[Tuple[int, Dict[str, Any]]], path_key: str, model_id: str
) -> List[Tuple[int, np.ndarray]]:
    """
    Map Phase Worker: Loads a chunk of images (or extracts from video) and encodes them.
    Returns a list of (original_index, embedding) tuples.
    """
    try:
        from movierag.indexing.clip_encoder import CLIPEncoder
        import cv2
        from PIL import Image

        # Initialize encoder locally (using CPU to avoid VRAM collisions)
        encoder = CLIPEncoder(model_name=model_id, device="cpu", local_files_only=False)

        results = []
        for i, item in chunk_items:
            path = item.get(path_key)
            if not path or not os.path.exists(path):
                continue

            # If path is a video file (from our Hybrid extraction plan)
            if path.endswith(".mp4") or path.endswith(".mkv"):
                timestamp = item.get("timestamp_sec", 0)
                cap = cv2.VideoCapture(path)

                # Seek to exact timestamp (milliseconds)
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    emb = encoder.encode_images([pil_img], normalize=True)
                    if emb is not None and len(emb) > 0:
                        results.append((i, emb[0]))

            # If path is a static image (MovieNet fallback)
            else:
                emb = encoder.encode_images([path], normalize=True)
                if emb is not None and len(emb) > 0:
                    results.append((i, emb[0]))

        return results
    except Exception as e:
        logger.error(f"Error in Map worker: {e}")
        return []


class ParallelVisualIndexer(VisualIndexer):
    """
    Implements MapReduce parallel indexing for Long-form Videos (inspired by MR.Video CVPR2025).
    Dramatically accelerates the extraction of visual features for large movie datasets
    by distributing the encoding workload across multiple CPU cores.
    """

    def __init__(
        self,
        index_dir: str,
        index_name: str = "parallel_visual_index",
        encoder=None,
        num_workers: int = 4,
    ):
        super().__init__(index_dir, index_name, encoder)
        self.num_workers = int(os.environ.get("MOVIERAG_WORKERS", num_workers))

    def build_index_parallel(
        self,
        items: List[Dict[str, Any]],
        id_key: str = "keyframe_id",
        path_key: str = "keyframe_path",
        movie_id_key: str = "movie_id",
        batch_size: int = 32,
    ) -> None:
        """
        Parallel MapReduce implementation of build_index.
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS is not installed. Install with: pip install faiss-cpu"
            )

        if not items:
            logger.warning("No items provided to build_index_parallel.")
            return

        logger.info(
            f"Starting MR.Video Parallel Indexing on {len(items)} items using {self.num_workers} workers..."
        )

        # 1. Compile Metadata Sequentially
        self._metadata = []
        for item in items:
            self._metadata.append(
                {
                    "id": item.get(id_key, "unknown"),
                    "path": item.get(path_key, ""),
                    "movie_id": item.get(movie_id_key, "unknown"),
                    **{
                        k: v
                        for k, v in item.items()
                        if k not in [id_key, path_key, movie_id_key]
                    },
                }
            )

        # 2. Chunking for Map Phase
        indexed_items = list(enumerate(items))
        chunks = [
            indexed_items[i : i + batch_size]
            for i in range(0, len(indexed_items), batch_size)
        ]
        logger.info(f"Split data into {len(chunks)} chunks for Map Phase.")

        model_id = (
            self.encoder.model_name if self.encoder else "openai/clip-vit-base-patch32"
        )
        all_embeddings = [None] * len(items)
        successful_count = 0

        # Pre-fetch model to HF cache on the main thread before launching workers
        logger.info(f"Pre-fetching HuggingFace model cache for '{model_id}'...")
        try:
            from movierag.indexing.clip_encoder import CLIPEncoder

            CLIPEncoder(model_name=model_id, device="cpu")._ensure_model_loaded()
        except Exception as e:
            logger.warning(f"Could not pre-fetch model on main thread: {e}")

        # 3. MAP PHASE: Parallel Execution
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_chunk = {
                executor.submit(_map_encode_chunk, chunk, path_key, model_id): i
                for i, chunk in enumerate(chunks)
            }

            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    results = future.result()
                    for idx, emb in results:
                        all_embeddings[idx] = emb
                        successful_count += 1
                    logger.debug(
                        f"Map Phase: Completed chunk {chunk_id + 1}/{len(chunks)}"
                    )
                except Exception as exc:
                    logger.error(f"Chunk {chunk_id} generated an exception: {exc}")

        logger.info(
            f"Map Phase Complete. Successfully encoded {successful_count}/{len(items)} images."
        )

        # 4. REDUCE PHASE: Filter invalid results and build FAISS index
        logger.info("Starting Reduce Phase: Building unified FAISS Index...")
        valid_embeddings = []
        valid_metadata = []

        for i, emb in enumerate(all_embeddings):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_metadata.append(self._metadata[i])

        if not valid_embeddings:
            logger.error("No valid embeddings generated, aborting index build.")
            return

        self._metadata = valid_metadata
        self._index = None  # Reset base index just in case

        embeddings_np = np.vstack(valid_embeddings).astype(np.float32)
        dim = embeddings_np.shape[1]

        # Note: CLIP features are already L2 normalized by encode_images(normalize=True),
        # so IndexFlatIP effectively computes Cosine Similarity.
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings_np)
        self._is_loaded = True

        self.save()
        logger.info(
            f"✅ Reduce Phase Complete. Unified map-reduced index built with {self._index.ntotal} vectors (Dim: {dim})."
        )

        # --- Hierarchical Scene Aggregation (L1) ---
        from movierag.config import get_config

        cfg = get_config()
        window = cfg.index.scene_window_size  # default 5

        logger.info(
            f"Building Hierarchical L1 Scene Index (window={window} shots per scene)..."
        )

        # Group frames by movie_id → sorted shots
        movie_shots: Dict[str, Dict[str, List]] = {}
        for idx, meta in enumerate(valid_metadata):
            m_id = meta.get("movie_id", "unknown")
            shot_id = meta.get("shot_id", f"unknown_shot_{idx}")
            if m_id not in movie_shots:
                movie_shots[m_id] = {}
            if shot_id not in movie_shots[m_id]:
                movie_shots[m_id][shot_id] = {"embs": [], "meta": meta.copy()}
            movie_shots[m_id][shot_id]["embs"].append(valid_embeddings[idx])

        scene_embeddings = []
        scene_metadata = []

        for m_id, shots_dict in movie_shots.items():
            # Sort shots by their numeric index
            sorted_shots = sorted(
                shots_dict.items(),
                key=lambda x: int(x[0].split("_")[-1]) if "_" in x[0] else 0,
            )

            # Group shots into scenes of `window` size
            for scene_idx in range(0, len(sorted_shots), window):
                scene_shots = sorted_shots[scene_idx : scene_idx + window]

                # Collect all frame embeddings within this scene window
                all_embs = []
                shot_ids_in_scene = []
                for shot_key, shot_data in scene_shots:
                    all_embs.extend(shot_data["embs"])
                    shot_ids_in_scene.append(shot_key)

                if not all_embs:
                    continue

                # Mean pooling over all frames in the scene
                arr = np.vstack(all_embs)
                mean_pooled = np.mean(arr, axis=0)
                norm = np.linalg.norm(mean_pooled)
                if norm > 0:
                    mean_pooled = mean_pooled / norm
                scene_embeddings.append(mean_pooled)

                # Scene metadata
                first_meta = scene_shots[0][1]["meta"].copy()
                scene_metadata.append(
                    {
                        "id": f"{m_id}_scene_{scene_idx // window:04d}",
                        "movie_id": m_id,
                        "is_scene": True,
                        "num_frames": len(all_embs),
                        "num_shots": len(scene_shots),
                        "shot_ids": shot_ids_in_scene,
                        "start_frame": first_meta.get("start_frame", 0),
                        "end_frame": scene_shots[-1][1]["meta"].get("end_frame", 0),
                        "timestamp": first_meta.get("timestamp", 0),
                        "timestamp_end": scene_shots[-1][1]["meta"].get(
                            "timestamp_end", 0
                        ),
                    }
                )

        if scene_embeddings:
            scene_embs_np = np.vstack(scene_embeddings).astype(np.float32)
            scene_index = faiss.IndexFlatIP(dim)
            scene_index.add(scene_embs_np)

            # Save the hierarchical scene index
            scene_index_path = self.index_dir / f"{self.index_name}_scenes.faiss"
            scene_meta_path = self.index_dir / f"{self.index_name}_scenes_meta.json"
            faiss.write_index(scene_index, str(scene_index_path))
            with open(scene_meta_path, "w", encoding="utf-8") as f:
                json.dump(scene_metadata, f, indent=2)

            logger.info(
                f"✅ L1 Scene Index: {scene_index.ntotal} scenes "
                f"from {len(movie_shots)} movies (window={window})"
            )
