"""
movierag/preprocessing/visual_pruner/token_merger.py
Spatiotemporal Token Merging (ToMe-inspired)
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SpatiotemporalTokenMerger:
    """
    Merge redundant visual tokens in space and time.
    """

    def __init__(
        self,
        merge_ratio: float = 0.5,
        token_size: int = 16,
        time_window: int = 3
    ):
        self.merge_ratio = merge_ratio
        self.token_size = token_size
        self.time_window = time_window

    def merge_tokens(
        self,
        embeddings: np.ndarray,
        spatial_resolution: Tuple[int, int] = (14, 14)
    ) -> np.ndarray:
        """
        Merge redundant embeddings (tokens) across time.
        """
        n_frames = len(embeddings)
        if n_frames < 2:
            return embeddings
        
        merged_embs = []
        for i in range(n_frames):
            start = max(0, i - self.time_window // 2)
            end = min(n_frames, i + self.time_window // 2 + 1)
            
            # Context tokens from neighboring frames
            window_embs = embeddings[start:end]
            
            # Simple average merging for current token
            merged_token = np.mean(window_embs, axis=0)
            merged_embs.append(merged_token)
            
        return np.array(merged_embs)

    def select_tokens_by_attention(
        self,
        embeddings: np.ndarray,
        attention_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select most informative tokens based on attention or density.
        """
        if attention_scores is None:
            # Use self-density as proxy for attention
            attention_scores = self._compute_self_density(embeddings)
            
        n_retain = int(len(embeddings) * (1.0 - self.merge_ratio))
        retain_indices = np.argsort(attention_scores)[-max(1, n_retain):]
        
        return embeddings[sorted(retain_indices)]

    def _compute_self_density(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute density of tokens in embedding space."""
        n = len(embeddings)
        density = np.zeros(n)
        for i in range(n):
            sims = np.dot(embeddings, embeddings[i])
            density[i] = sims.mean()
        return density
