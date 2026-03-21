"""
movierag/preprocessing/visual_pruner/pruner_v3.py
Main Visual Pruner Orchestrator (Version 3.0 Logic)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from dataclasses import dataclass

from .quality_metrics import QualityMetrics
from .trajectory_encoder import TrajectoryAwareEncoder
from .dynamic_segmenter import DynamicTemporalSegmenter
from .relevance_predictor import QueryFrameRelevancePredictor
from .coverage_optimizer import InformationCoverageOptimizer
from .token_merger import SpatiotemporalTokenMerger

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration cho Visual Pruner."""
    # Layer 1: Deduplication
    traj_weight: float = 0.3
    similarity_threshold: float = 0.98  # Thắt chặt cực hạn để gộp các khung hình gần giống nhau
    
    # Layer 2: Quality & Density
    target_fps: float = 1.0
    quality_threshold: float = 0.7      # Chỉ lấy ảnh cực kỳ sắc nét
    min_frames_per_shot: int = 1
    max_frames_per_scene: int = 3       # Giới hạn tối thiểu (Cực kỳ tiết kiệm chi phí VLM)
    
    # Query settings
    query_relevance_weight: float = 0.4
    default_query: str = "main characters and important actions"
    
    # Performance
    n_workers: int = 4
    enable_tracking: bool = True


class VisualPruner:
    """
    Advanced Visual Pruner - Research-backed frame selection.
    
    LAYERS:
    1. Trajectory-Aware Deduplication: Clean redundant frames with motion context.
    2. Information Coverage Optimization: Select frames for VLM high-quality context.
    """

    def __init__(self, config: Optional[PruningConfig] = None, clip_encoder: Any = None):
        self.config = config or PruningConfig()
        self.clip_encoder = clip_encoder
        
        # Initialize modules
        self.quality_tester = QualityMetrics()
        self.traj_encoder = TrajectoryAwareEncoder(
            clip_encoder=self.clip_encoder, 
            enable_tracking=self.config.enable_tracking
        )
        self.segmenter = DynamicTemporalSegmenter(
            similarity_threshold=self.config.similarity_threshold
        )
        self.relevance_predictor = QueryFrameRelevancePredictor()
        self.optimizer = InformationCoverageOptimizer(
            max_frames=self.config.max_frames_per_scene
        )
        self.token_merger = SpatiotemporalTokenMerger(merge_ratio=0.3)

    def prune_scene(
        self,
        movie_id: str,
        scene_idx: int,
        frame_paths: List[str],
        shot_map: List[int],  # New: mapping each frame path to a shot_id
        timestamps: Optional[List[float]] = None,
        query: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Prune frames in a scene into two layers: Vector Clean and VLM Quality.
        """
        if not frame_paths:
            return [], []
        
        logger.info(f"🔍 Pruning Scene {scene_idx} ({len(frame_paths)} frames)...")
        
        try:
            # 1. Quality Filtering
            qualities = self.quality_tester.compute_quality_batch(
                frame_paths, n_workers=self.config.n_workers
            )
            valid_indices = [i for i, q in enumerate(qualities) if q.get("composite", 0) > 0.1]
            
            if not valid_indices:
                valid_indices = [len(frame_paths) // 2]  # Fallback to middle frame
            
            frame_paths = [frame_paths[i] for i in valid_indices]
            qualities = [qualities[i] for i in valid_indices]
            shot_map = [shot_map[i] for i in valid_indices]  # Fix: align shot_map
            
            # 2. Embedding & Trajectory
            # Read images for trajectory (limit count if needed, but for now full)
            imgs = [cv2.imread(p) for p in frame_paths]
            imgs = [cv2.resize(img, (224, 224)) for img in imgs if img is not None]
            
            encoding = self.traj_encoder.encode_with_trajectory(
                imgs, image_paths=frame_paths
            )
            embeddings = encoding["fused"]
            
            # 2.5 Token Merging (Feature-level deduplication)
            embeddings = self.token_merger.merge_tokens(embeddings)
            
            # 3. Dynamic Segmentation
            segments = self.segmenter.segment(embeddings)
            
            # 4. Layer 1 Selection: Vector Clean (Highly diverse frames)
            vector_clean_indices = self._select_layer1(embeddings, segments)
            
            # 5. Layer 2 Selection: VLM Quality (Relevance-aware optimization)
            query = query or self.config.default_query
            rel_scores, _ = self.relevance_predictor.predict_relevance(
                embeddings, query, video_context=embeddings, clip_encoder=self.clip_encoder
            )
            vlm_indices = self.optimizer.optimize_selection(embeddings, rel_scores)
            
            # Map back to original data
            layer1_frames = self._format_results(
                vector_clean_indices,
                frame_paths,
                shot_map,
                qualities,
                rel_scores,
                layer=1,
                scene_idx=scene_idx,
                timestamps=timestamps,
            )
            layer2_frames = self._format_results(
                vlm_indices,
                frame_paths,
                shot_map,
                qualities,
                rel_scores,
                layer=2,
                scene_idx=scene_idx,
                timestamps=timestamps,
            )
            
            logger.info(
                f"  ✅ Scene {scene_idx}: {len(frame_paths)} -> "
                f"L1: {len(layer1_frames)}, L2: {len(layer2_frames)}"
            )
            
            return layer1_frames, layer2_frames
            
        except Exception as e:
            logger.error(f"Pruning failed for Scene {scene_idx}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Basic fallback with mapping
            middle = len(frame_paths) // 2
            fallback = [{
                "frame_path": frame_paths[middle],
                "shot_id": int(shot_map[middle]),
                "scene_id": int(scene_idx),
                "quality": {"composite": 1.0},
                "relevance": 1.0,
                "pruning_layer": 0,
                "timestamp": 0.0
            }]
            return fallback, fallback

    def _select_layer1(
        self,
        embeddings: np.ndarray,
        segments: List[Dict[str, Any]]
    ) -> List[int]:
        """Select frames per segment for Layer 1. Pick only the most representative one."""
        indices = []
        for seg in segments:
            # Pick center frame as the most representative
            mid_idx = (seg["start"] + seg["end"]) // 2
            indices.append(mid_idx)
            
            # Only pick an extra frame if segment is extremely long (> 15 images)
            if seg["length"] >= 15:
                # Pick one more at 3/4 position
                indices.append((seg["start"] + 3 * seg["end"]) // 4)
                
        return sorted(list(set(indices)))

    def _format_results(
        self,
        indices: List[int],
        paths: List[str],
        shot_map: List[int],
        qualities: List[Dict[str, Any]],
        rel_scores: np.ndarray,
        layer: int,
        scene_idx: int,
        timestamps: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Format selection results into list of dicts."""
        results = []
        for idx in indices:
            if idx < len(paths):
                results.append({
                    "frame_path": str(paths[idx]),
                    "shot_id": int(shot_map[idx]),
                    "scene_id": int(scene_idx),
                    "quality": qualities[idx],
                    "relevance": float(rel_scores[idx]),
                    "pruning_layer": layer,
                    "timestamp": (
                        float(timestamps[idx])
                        if timestamps is not None and idx < len(timestamps)
                        else idx * (1.0 / self.config.target_fps)
                    ),
                })
        return results
