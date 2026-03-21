"""
movierag/preprocessing/visual_pruner/trajectory_encoder.py
Trajectory-Aware Embedding (TrajTok-inspired)
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict
import cv2

logger = logging.getLogger(__name__)


class TrajectoryAwareEncoder:
    """
    Combine CLIP embedding with trajectory tokens.
    """

    def __init__(
        self,
        clip_encoder: Any = None,
        tracker_type: str = "simple",
        enable_tracking: bool = True
    ):
        self.clip_encoder = clip_encoder
        self.tracker_type = tracker_type
        self.enable_tracking = enable_tracking

    def encode_with_trajectory(
        self,
        frames: List[np.ndarray],
        image_paths: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode frames with trajectory-aware embeddings.
        """
        n_frames = len(frames)
        
        # Step 1: CLIP semantic embeddings
        if image_paths and self.clip_encoder:
            semantic_embs = self.clip_encoder.encode_images(
                image_paths, normalize=True, show_progress=False
            )
        else:
            logger.warning("No CLIP encoder, using random embeddings")
            semantic_embs = np.random.randn(n_frames, 512)
            semantic_embs = semantic_embs / np.linalg.norm(semantic_embs, axis=1, keepdims=True)
        
        result = {"semantic": semantic_embs}
        
        # Step 2: Trajectory extraction (if enabled)
        if self.enable_tracking:
            trajectories = self._extract_trajectories(frames)
            trajectory_tokens = self._trajectories_to_tokens(trajectories)
            
            # Step 3: Fuse semantic + trajectory
            fused_embs = self._fuse_embeddings(semantic_embs, trajectory_tokens)
            
            result["trajectory"] = trajectory_tokens
            result["fused"] = fused_embs
            result["trajectories"] = trajectories
        else:
            result["trajectory"] = np.zeros((n_frames, 10))
            result["fused"] = semantic_embs
            result["trajectories"] = {}
        
        return result

    def _extract_trajectories(
        self,
        frames: List[np.ndarray]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract object trajectories using simple tracking.
        """
        trajectories = defaultdict(list)
        
        if len(frames) < 2:
            return dict(trajectories)
        
        try:
            bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            track_id = 0
            
            for frame_idx, frame in enumerate(frames):
                if frame is None:
                    continue
                
                fg_mask = bg_subtractor.apply(frame)
                contours, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Minimum object size
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        point = {
                            "frame": frame_idx,
                            "track_id": track_id,
                            "bbox": [x, y, w, h],
                            "center": [x + w/2, y + h/2],
                            "area": area,
                            "velocity": [0, 0]
                        }
                        trajectories[track_id].append(point)
                        track_id += 1
            
            for t_id, points in trajectories.items():
                for i in range(1, len(points)):
                    prev_center = np.array(points[i-1]["center"])
                    curr_center = np.array(points[i]["center"])
                    velocity = curr_center - prev_center
                    points[i]["velocity"] = velocity.tolist()
            
        except Exception as e:
            logger.warning(f"Trajectory extraction failed: {e}")
        
        return dict(trajectories)

    def _trajectories_to_tokens(
        self,
        trajectories: Dict[int, List[Dict[str, Any]]]
    ) -> np.ndarray:
        """
        Convert trajectories to fixed-size tokens.
        """
        tokens = []
        
        for track_id, points in trajectories.items():
            if len(points) < 2:
                continue
            
            positions = np.array([p["center"] for p in points])
            velocities = np.array([p.get("velocity", [0, 0]) for p in points])
            
            trajectory_length = np.sum(
                np.linalg.norm(np.diff(positions, axis=0), axis=1)
            )
            avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
            
            accelerations = np.diff(velocities, axis=0)
            max_acceleration = np.max(np.linalg.norm(accelerations, axis=1)) if len(accelerations) > 0 else 0
            
            token = np.array([
                trajectory_length,
                avg_velocity,
                max_acceleration,
                0.0, # placeholder for direction changes
                np.mean(positions[:, 0]),  # Centroid x
                np.mean(positions[:, 1]),  # Centroid y
                np.std(positions[:, 0]),   # Variance x
                np.std(positions[:, 1]),   # Variance y
                len(points),               # Duration
                trajectory_length / max(len(points), 1)  # Speed
            ])
            tokens.append(token)
        
        if not tokens:
            return np.zeros((0, 10))
        
        return np.array(tokens)

    def _fuse_embeddings(
        self,
        semantic_embs: np.ndarray,
        trajectory_tokens: np.ndarray
    ) -> np.ndarray:
        """
        Fuse semantic and trajectory embeddings.
        """
        n_frames = len(semantic_embs)
        
        if len(trajectory_tokens) == 0:
            return semantic_embs
        
        mean_trajectory = np.mean(trajectory_tokens, axis=0, keepdims=True)
        trajectory_features = np.tile(mean_trajectory, (n_frames, 1))
        
        # Pad or truncate
        d_semantic = semantic_embs.shape[1]
        d_traj = trajectory_features.shape[1]
        
        if d_traj < d_semantic:
            padding = np.zeros((n_frames, d_semantic - d_traj))
            trajectory_features = np.concatenate([trajectory_features, padding], axis=1)
        elif d_traj > d_semantic:
            trajectory_features = trajectory_features[:, :d_semantic]
            
        semantic_norm = semantic_embs / (np.linalg.norm(semantic_embs, axis=1, keepdims=True) + 1e-8)
        trajectory_norm = trajectory_features / (np.linalg.norm(trajectory_features, axis=1, keepdims=True) + 1e-8)
        
        # Ensure correct shapes for broadcasting (N, D) + (N, D)
        if semantic_norm.ndim == 1:
            semantic_norm = semantic_norm.reshape(n_frames, -1)
        if trajectory_norm.ndim == 1:
            trajectory_norm = trajectory_norm.reshape(n_frames, -1)
            
        fused = 0.7 * semantic_norm + 0.3 * trajectory_norm
        fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)
        
        return fused
