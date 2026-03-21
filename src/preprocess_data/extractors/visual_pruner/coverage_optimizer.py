"""
movierag/preprocessing/visual_pruner/coverage_optimizer.py
Information Coverage Optimization (AKS-inspired)
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class InformationCoverageOptimizer:
    """
    Multi-objective optimization for frame selection.
    """

    def __init__(
        self,
        relevance_weight: float = 0.5,
        coverage_weight: float = 0.3,
        diversity_weight: float = 0.2,
        max_frames: int = 50
    ):
        self.relevance_weight = relevance_weight
        self.coverage_weight = coverage_weight
        self.diversity_weight = diversity_weight
        self.max_frames = max_frames

    def optimize_selection(
        self,
        frame_embeddings: np.ndarray,
        relevance_scores: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select frames to maximize multi-objective score using greedy approach.
        """
        n_frames = len(frame_embeddings)
        if n_frames <= self.max_frames:
            return list(range(n_frames))
        
        selected_indices = []
        remaining_indices = set(range(n_frames))
        
        # Start with the most relevant frame
        first_frame = int(np.argmax(relevance_scores))
        selected_indices.append(first_frame)
        remaining_indices.remove(first_frame)
        
        while len(selected_indices) < self.max_frames and remaining_indices:
            best_score, best_frame = -np.inf, None
            
            # Sub-sample candidates for speed if many frames
            candidates = list(remaining_indices)
            if len(candidates) > 100:
                step = len(candidates) // 100
                candidates = candidates[::step]
            
            for candidate in candidates:
                score = self._compute_multi_objective_score(
                    candidate, selected_indices, frame_embeddings,
                    relevance_scores, timestamps
                )
                if score > best_score:
                    best_score, best_frame = score, candidate
            
            if best_frame is not None:
                selected_indices.append(best_frame)
                remaining_indices.remove(best_frame)
            else:
                break
        
        return sorted(selected_indices)

    def _compute_multi_objective_score(
        self,
        candidate: int,
        selected: List[int],
        embeddings: np.ndarray,
        relevance_scores: np.ndarray,
        timestamps: Optional[np.ndarray]
    ) -> float:
        """Compute multi-objective score for candidate frame."""
        relevance = relevance_scores[candidate]
        coverage = self._compute_coverage_gain(candidate, selected, embeddings)
        diversity = self._compute_diversity_score(
            candidate, selected, embeddings, timestamps
        )
        return (self.relevance_weight * relevance +
                self.coverage_weight * coverage +
                self.diversity_weight * diversity)

    def _compute_coverage_gain(
        self,
        candidate: int,
        selected: List[int],
        embeddings: np.ndarray
    ) -> float:
        """Compute information coverage gain using 1 - max similarity."""
        if not selected:
            return 1.0
        candidate_emb = embeddings[candidate]
        selected_embs = embeddings[selected]
        similarities = np.dot(selected_embs, candidate_emb)
        return 1.0 - float(similarities.max())

    def _compute_diversity_score(
        self,
        candidate: int,
        selected: List[int],
        embeddings: np.ndarray,
        timestamps: Optional[np.ndarray]
    ) -> float:
        """Compute diversity score (semantic + temporal)."""
        if not selected:
            return 1.0
        
        candidate_emb = embeddings[candidate]
        selected_embs = embeddings[selected]
        
        # Mean semantic diversity
        semantic_diversity = 1.0 - float(np.mean(np.dot(selected_embs, candidate_emb)))
        
        if timestamps is not None:
            # Temporal diversity (distance to nearest selected frame)
            time_distances = np.abs(timestamps[selected] - timestamps[candidate])
            time_range = timestamps.max() - timestamps.min() + 1e-8
            temporal_diversity = float(time_distances.min()) / time_range
        else:
            temporal_diversity = 0.5
        
        return 0.7 * semantic_diversity + 0.3 * temporal_diversity
