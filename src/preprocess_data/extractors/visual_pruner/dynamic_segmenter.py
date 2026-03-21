"""
movierag/preprocessing/visual_pruner/dynamic_segmenter.py
Dynamic Density-Based Temporal Segmentation (FastVID-inspired)
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DynamicTemporalSegmenter:
    """
    Adaptive temporal segmentation based on visual density.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_segment_length: int = 3,
        max_segment_length: int = 30,
        density_window: int = 5,
        boundary_percentile: float = 15
    ):
        self.similarity_threshold = similarity_threshold
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.density_window = density_window
        self.boundary_percentile = boundary_percentile

    def segment(
        self,
        embeddings: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Segment video based on visual density.
        """
        n_frames = len(embeddings)
        
        if n_frames < 2:
            return [{
                "start": 0, "end": n_frames, "frames": list(range(n_frames)),
                "avg_similarity": 1.0, "density": 1.0
            }]
        
        similarities = self._compute_temporal_similarities(embeddings)
        boundaries = self._detect_density_boundaries(similarities)
        segments = self._create_segments(
            boundaries, n_frames, embeddings, timestamps
        )
        
        return segments

    def _compute_temporal_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity between consecutive frames."""
        n_frames = len(embeddings)
        similarities = np.zeros(n_frames - 1)
        
        for i in range(n_frames - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities[i] = sim
        
        return similarities

    def _detect_density_boundaries(
        self,
        similarities: np.ndarray
    ) -> List[int]:
        """Detect segment boundaries using density analysis."""
        if len(similarities) < self.density_window:
            return [0, len(similarities) + 1]
        
        cumsum = np.cumsum(np.insert(similarities, 0, 0))
        window = min(self.density_window, len(similarities))
        density = (cumsum[window:] - cumsum[:-window]) / window
        
        gradient = np.diff(density)
        threshold = np.percentile(gradient, self.boundary_percentile)
        boundary_candidates = np.where(gradient < threshold)[0]
        
        boundaries = [0]
        last_boundary = -self.min_segment_length
        
        for candidate in boundary_candidates:
            if candidate - last_boundary >= self.min_segment_length:
                boundaries.append(candidate + 1)
                last_boundary = candidate
        
        boundaries.append(len(similarities) + 1)
        return boundaries

    def _create_segments(
        self,
        boundaries: List[int],
        n_frames: int,
        embeddings: np.ndarray,
        timestamps: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Create segment dictionaries from boundaries."""
        segments = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = min(boundaries[i + 1], n_frames)
            
            if end <= start:
                continue
            
            segment_length = end - start
            if segment_length < self.min_segment_length and segments:
                segments[-1]["end"] = end
                segments[-1]["frames"].extend(range(start, end))
                continue
            
            if segment_length > self.max_segment_length:
                sub_segments = self._split_long_segment(start, end, embeddings)
                segments.extend(sub_segments)
                continue
            
            segment_embs = embeddings[start:end]
            avg_similarity = self._compute_segment_similarity(segment_embs)
            
            segment = {
                "start": start, "end": end, "frames": list(range(start, end)),
                "avg_similarity": float(avg_similarity), "density": float(avg_similarity),
                "length": segment_length
            }
            if timestamps is not None:
                segment["start_time"] = float(timestamps[start])
                segment["end_time"] = float(timestamps[end - 1])
            
            segments.append(segment)
        
        return segments

    def _compute_segment_similarity(self, embeddings: np.ndarray) -> float:
        """Compute average pairwise similarity within segment."""
        if len(embeddings) < 2:
            return 1.0
        n = len(embeddings)
        total_sim, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, min(i + 5, n)):
                sim = np.dot(embeddings[i], embeddings[j])
                total_sim += sim
                count += 1
        return total_sim / count if count > 0 else 1.0

    def _split_long_segment(
        self,
        start: int,
        end: int,
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Split a segment that exceeds max length."""
        segment_length = end - start
        num_splits = (segment_length - 1) // self.max_segment_length + 1
        split_size = segment_length // num_splits
        
        sub_segments = []
        for i in range(num_splits):
            sub_start = start + i * split_size
            sub_end = start + (i + 1) * split_size if i < num_splits - 1 else end
            sub_embs = embeddings[sub_start:sub_end]
            sub_segments.append({
                "start": sub_start, "end": sub_end, "frames": list(range(sub_start, sub_end)),
                "avg_similarity": self._compute_segment_similarity(sub_embs),
                "density": self._compute_segment_similarity(sub_embs),
                "length": sub_end - sub_start
            })
        return sub_segments
