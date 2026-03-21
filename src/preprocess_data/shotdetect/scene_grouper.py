"""
movierag/preprocessing/auto_annotator/scene_grouper.py
Group shots into physical scenes based on visual similarity
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .shot_detector import ShotInfo

logger = logging.getLogger(__name__)


@dataclass
class SceneInfo:
    """Thông tin một physical scene (group of shots)."""
    scene_id: str
    scene_idx: int
    shot_indices: List[int]
    shot_range: Tuple[int, int]  # [start_shot_idx, end_shot_idx]
    frame_range: Tuple[int, int]  # [start_frame, end_frame]
    start_sec: float
    end_sec: float
    duration_sec: float
    num_shots: int
    visual_coherence: float = 1.0  # 0-1, higher = more coherent


@dataclass
class SceneGroupingConfig:
    """Configuration cho scene grouping."""
    # Grouping strategy
    strategy: str = "similarity"  # similarity, count, time
    
    # Similarity-based params
    similarity_threshold: float = 0.7  # CLIP embedding similarity
    min_shots_per_scene: int = 1
    max_shots_per_scene: int = 20
    
    # Time-based params
    max_scene_duration_sec: float = 120.0
    min_scene_duration_sec: float = 2.0
    
    # Count-based params
    max_shots_count: int = 10


class SceneGrouper:
    """
    Group shots into physical scenes.
    
    STRATEGIES:
    1. Similarity-based: Group shots with high visual similarity
    2. Count-based: Group N consecutive shots
    3. Time-based: Group shots within time window
    """

    def __init__(self, config: Optional[SceneGroupingConfig] = None):
        self.config = config or SceneGroupingConfig()
        self._clip_encoder = None

    def group_shots(
        self,
        shots: List[ShotInfo],
        video_path: Optional[str] = None,
        thumbnail_dir: Optional[str] = None
    ) -> List[SceneInfo]:
        """
        Group shots into physical scenes.
        
        Args:
            shots: List of detected shots
            video_path: Optional video path for embedding extraction
            thumbnail_dir: Optional directory with shot thumbnails
        
        Returns:
            List of SceneInfo objects
        """
        if not shots:
            logger.warning("No shots to group")
            return []
        
        logger.info(f"📦 Grouping {len(shots)} shots into scenes...")
        
        if self.config.strategy == "similarity":
            scenes = self._group_by_similarity(shots, thumbnail_dir)
        elif self.config.strategy == "count":
            scenes = self._group_by_count(shots)
        elif self.config.strategy == "time":
            scenes = self._group_by_time(shots)
        else:
            scenes = self._group_by_similarity(shots, thumbnail_dir)
        
        logger.info(f"  ✅ Created {len(scenes)} physical scenes")
        return scenes

    def _group_by_similarity(
        self,
        shots: List[ShotInfo],
        thumbnail_dir: Optional[str]
    ) -> List[SceneInfo]:
        """Group shots based on visual embedding similarity."""
        if not thumbnail_dir:
            logger.warning("No thumbnail dir, falling back to count-based grouping")
            return self._group_by_count(shots)
        
        try:
            from pathlib import Path
            from movierag.indexing.clip_encoder import CLIPEncoder
            
            # Get thumbnail paths
            thumbnail_paths = []
            for shot in shots:
                if shot.thumbnail_path and Path(shot.thumbnail_path).exists():
                    thumbnail_paths.append(shot.thumbnail_path)
                else:
                    # Try to construct path
                    expected_path = Path(thumbnail_dir) / f"shot_{shot.shot_idx}-0.jpg"
                    if expected_path.exists():
                        thumbnail_paths.append(str(expected_path))
                    else:
                        thumbnail_paths.append(None)
            
            # Filter shots with valid thumbnails
            valid_indices = [i for i, p in enumerate(thumbnail_paths) if p is not None]
            valid_paths = [p for p in thumbnail_paths if p is not None]
            
            if len(valid_paths) < 2:
                logger.warning("Not enough valid thumbnails, using count-based")
                return self._group_by_count(shots)
            
            # Compute embeddings
            self._clip_encoder = CLIPEncoder()
            embeddings = self._clip_encoder.encode_images(valid_paths, normalize=True)
            
            # Compute pairwise similarities (consecutive shots)
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1])
                similarities.append(sim)
            
            # Find scene boundaries (where similarity drops below threshold)
            boundaries = [0]  # First shot always starts a scene
            for i, sim in enumerate(similarities):
                if sim < self.config.similarity_threshold:
                    boundaries.append(i + 1)
            boundaries.append(len(valid_indices))  # End
            
            # Create scenes
            scenes = []
            for scene_idx in range(len(boundaries) - 1):
                start_idx = boundaries[scene_idx]
                end_idx = boundaries[scene_idx + 1]
                
                # Map back to original shot indices
                shot_indices = valid_indices[start_idx:end_idx]
                
                if shot_indices:
                    scene = self._create_scene_from_shots(
                        shots, shot_indices, scene_idx
                    )
                    # Compute visual coherence
                    if end_idx - start_idx > 1:
                        scene_similarities = similarities[start_idx:end_idx - 1]
                        scene.visual_coherence = float(np.mean(scene_similarities))
                    scenes.append(scene)
            
            return scenes
            
        except Exception as e:
            logger.error(f"Similarity-based grouping failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._group_by_count(shots)

    def _group_by_count(self, shots: List[ShotInfo]) -> List[SceneInfo]:
        """Group shots by fixed count."""
        scenes = []
        current_shots = []
        scene_idx = 0
        
        for shot in shots:
            current_shots.append(shot)
            
            if len(current_shots) >= self.config.max_shots_count:
                scene = self._create_scene_from_shots(
                    shots, [s.shot_idx for s in current_shots], scene_idx
                )
                scenes.append(scene)
                current_shots = []
                scene_idx += 1
        
        # Remaining shots
        if current_shots:
            scene = self._create_scene_from_shots(
                shots, [s.shot_idx for s in current_shots], scene_idx
            )
            scenes.append(scene)
        
        return scenes

    def _group_by_time(self, shots: List[ShotInfo]) -> List[SceneInfo]:
        """Group shots by time window."""
        scenes = []
        current_shots = []
        scene_idx = 0
        scene_start_time = None
        
        for shot in shots:
            if scene_start_time is None:
                scene_start_time = shot.start_sec
            
            current_shots.append(shot)
            current_duration = shot.end_sec - scene_start_time
            
            if current_duration >= self.config.max_scene_duration_sec:
                scene = self._create_scene_from_shots(
                    shots, [s.shot_idx for s in current_shots], scene_idx
                )
                scenes.append(scene)
                current_shots = []
                scene_start_time = None
                scene_idx += 1
        
        # Remaining shots
        if current_shots:
            scene = self._create_scene_from_shots(
                shots, [s.shot_idx for s in current_shots], scene_idx
            )
            scenes.append(scene)
        
        return scenes

    def _create_scene_from_shots(
        self,
        all_shots: List[ShotInfo],
        shot_indices: List[int],
        scene_idx: int
    ) -> SceneInfo:
        """Create SceneInfo from a group of shot indices."""
        if not shot_indices:
            raise ValueError("Empty shot indices")
        
        # Get shots
        selected_shots = [all_shots[i] for i in shot_indices if i < len(all_shots)]
        
        if not selected_shots:
            selected_shots = [all_shots[min(shot_indices[0], len(all_shots) - 1)]]
        
        first_shot = selected_shots[0]
        last_shot = selected_shots[-1]
        
        return SceneInfo(
            scene_id=f"scene_{scene_idx}",
            scene_idx=scene_idx,
            shot_indices=shot_indices,
            shot_range=(first_shot.shot_idx, last_shot.shot_idx),
            frame_range=(first_shot.start_frame, last_shot.end_frame),
            start_sec=first_shot.start_sec,
            end_sec=last_shot.end_sec,
            duration_sec=round(last_shot.end_sec - first_shot.start_sec, 3),
            num_shots=len(shot_indices)
        )
