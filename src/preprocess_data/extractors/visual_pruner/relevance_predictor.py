"""
movierag/preprocessing/visual_pruner/relevance_predictor.py
Query-Frame Relevance Predictor (KVTP-inspired)
"""

import logging
from typing import Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class QueryFrameRelevancePredictor:
    """
    Query-aware frame relevance prediction.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        target_retention: float = 0.3,
        enable_context_fusion: bool = True,
        local_context_window: int = 5
    ):
        self.temperature = temperature
        self.target_retention = target_retention
        self.enable_context_fusion = enable_context_fusion
        self.local_context_window = local_context_window
        self._clip_encoder = None

    def predict_relevance(
        self,
        frame_embeddings: np.ndarray,
        query: str,
        video_context: Optional[np.ndarray] = None,
        clip_encoder: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict relevance scores for each frame based on query.
        """
        n_frames = len(frame_embeddings)
        if clip_encoder:
            self._clip_encoder = clip_encoder
        
        query_embedding = self._encode_query(query)
        base_logits = self._compute_similarity(frame_embeddings, query_embedding)
        
        if self.enable_context_fusion and video_context is not None:
            enhanced_logits = self._apply_context_fusion(
                frame_embeddings, base_logits, video_context, query_embedding
            )
        else:
            enhanced_logits = base_logits
        
        soft_scores = self._apply_temperature(enhanced_logits, self.temperature)
        pruning_rates = self._scores_to_pruning_rates(
            soft_scores, self.target_retention
        )
        
        return soft_scores, pruning_rates

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding."""
        if self._clip_encoder:
            try:
                embedding = self._clip_encoder.encode_texts([query], normalize=True)
                return embedding[0]
            except Exception as e:
                logger.warning(f"Query encoding failed: {e}")
        
        return np.random.randn(512)

    def _compute_similarity(
        self,
        embeddings: np.ndarray,
        query_embedding: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between frames and query."""
        return np.dot(embeddings, query_embedding)

    def _apply_context_fusion(
        self,
        frame_embeddings: np.ndarray,
        base_logits: np.ndarray,
        video_context: np.ndarray,
        query_embedding: np.ndarray
    ) -> np.ndarray:
        """Apply local and global context fusion."""
        n_frames = len(frame_embeddings)
        local_contexts = self._compute_local_contexts(frame_embeddings)
        
        # Correct similarity: how similar is the LOCAL CONTEXT to the query
        local_sims = np.dot(local_contexts, query_embedding)
        
        # Weighted fusion of logits (ensure all are (N,) or scalars)
        base_logits = base_logits.flatten()
        local_sims = local_sims.flatten()
        fused_logits = (0.5 * base_logits + 0.3 * local_sims + 0.2 * base_logits.mean())
        return fused_logits

    def _compute_local_contexts(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute local context for each frame."""
        n_frames = len(embeddings)
        contexts = np.zeros_like(embeddings)
        for i in range(n_frames):
            start = max(0, i - self.local_context_window // 2)
            end = min(n_frames, i + self.local_context_window // 2 + 1)
            contexts[i] = np.mean(embeddings[start:end], axis=0)
        return contexts

    def _apply_temperature(
        self,
        logits: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature-controlled softmax."""
        if temperature <= 0:
            scores = np.zeros_like(logits)
            scores[np.argmax(logits)] = 1.0
            return scores
        
        exp_logits = np.exp(logits / temperature)
        return exp_logits / (exp_logits.sum() + 1e-8)

    def _scores_to_pruning_rates(
        self,
        scores: np.ndarray,
        target_retention: float
    ) -> np.ndarray:
        """Convert relevance scores to pruning rates."""
        max_rate, min_rate = 0.9, 0.1
        score_range = scores.max() - scores.min() + 1e-8
        norm_scores = (scores - scores.min()) / score_range
        pruning_rates = max_rate - (max_rate - min_rate) * norm_scores
        
        current_retention = 1.0 - pruning_rates.mean()
        if current_retention > 0:
            adjustment = target_retention / current_retention
            pruning_rates = np.clip(pruning_rates * adjustment, min_rate, max_rate)
        
        return pruning_rates
