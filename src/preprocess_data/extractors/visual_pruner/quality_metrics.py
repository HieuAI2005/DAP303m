"""
movierag/preprocessing/visual_pruner/quality_metrics.py
Multi-dimensional quality assessment for frames
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class QualityMetrics:
    """
    Multi-dimensional quality metrics for frame assessment.
    """

    @staticmethod
    def compute_sharpness(image: np.ndarray) -> float:
        """
        Compute sharpness using Laplacian variance.
        """
        if image is None or image.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_brightness(image: np.ndarray) -> float:
        """
        Compute mean brightness.
        """
        if image is None or image.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        
        # Penalty for under/over exposure
        if mean_brightness < 30 or mean_brightness > 220:
            return mean_brightness * 0.5
        
        return mean_brightness

    @staticmethod
    def compute_contrast(image: np.ndarray) -> float:
        """
        Compute contrast using intensity standard deviation.
        """
        if image is None or image.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    @staticmethod
    def compute_noise_level(image: np.ndarray) -> float:
        """
        Estimate noise level using high-frequency content.
        """
        if image is None or image.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        return float(np.std(noise))

    @staticmethod
    def compute_quality_score(image_path: str) -> Dict[str, Any]:
        """
        Compute comprehensive quality score for an image.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    "sharpness": 0.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "noise": 0.0,
                    "composite": 0.0,
                    "valid": False
                }
            
            sharpness = QualityMetrics.compute_sharpness(image)
            brightness = QualityMetrics.compute_brightness(image)
            contrast = QualityMetrics.compute_contrast(image)
            noise = QualityMetrics.compute_noise_level(image)
            
            # Normalize metrics to [0, 1]
            sharpness_norm = min(sharpness / 500.0, 1.0)
            brightness_norm = min(brightness / 255.0, 1.0)
            contrast_norm = min(contrast / 128.0, 1.0)
            noise_penalty = max(0.0, 1.0 - noise / 50.0)
            
            # Composite score with weights
            composite = (
                0.35 * sharpness_norm +
                0.20 * brightness_norm +
                0.25 * contrast_norm +
                0.20 * noise_penalty
            )
            
            return {
                "sharpness": sharpness,
                "brightness": brightness,
                "contrast": contrast,
                "noise": noise,
                "composite": composite,
                "valid": True
            }
            
        except Exception as e:
            logger.warning(f"Quality computation failed for {image_path}: {e}")
            return {
                "sharpness": 0.0, "composite": 0.0, "valid": False
            }

    @staticmethod
    def compute_quality_batch(image_paths: List[str], n_workers: int = 4) -> List[Dict[str, Any]]:
        """Compute quality scores for multiple images in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        
        results = [None] * len(image_paths)
        
        def compute(idx_path):
            idx, path = idx_path
            return idx, QualityMetrics.compute_quality_score(path)
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for idx, quality in executor.map(compute, enumerate(image_paths)):
                results[idx] = quality
        
        return results
