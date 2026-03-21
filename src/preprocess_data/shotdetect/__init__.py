"""
movierag/preprocessing/auto_annotator/__init__.py
"""
from .core import AutoAnnotator
from .shot_detector import ShotDetector, ShotInfo, DetectionConfig

__all__ = ["AutoAnnotator", "ShotDetector", "ShotInfo", "DetectionConfig"]
