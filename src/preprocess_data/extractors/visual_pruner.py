"""
Unified Visual Pruner Orchestrator.

Consolidates Layer 1 (Deduplication) and Layer 2 (Quality/Relevance) selection.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .visual_pruner.pruner_v3 import VisualPruner, PruningConfig

logger = logging.getLogger(__name__)

# Re-export for easier access
__all__ = ["VisualPruner", "PruningConfig"]
