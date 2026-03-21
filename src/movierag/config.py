"""
Configuration management for MovieRAG.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathConfig:
    """Paths configuration."""

    # Project root (config.py → movierag → src → project_root)
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    # Data directories
    data_dir: Path = field(init=False)
    rawdata_dir: Path = field(init=False)
    movienet_dir: Path = field(init=False)
    shot_keyf_dir: Path = field(init=False)
    shot_txt_dir: Path = field(init=False)
    raw_videos_dir: Path = field(init=False)
    movie_subset_dir: Path = field(init=False)
    annotation_dir: Path = field(init=False)
    meta_dir: Path = field(init=False)
    script_dir: Path = field(init=False)
    subtitle_dir: Path = field(init=False)
    index_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.rawdata_dir = self.data_dir / "rawdata"
        self.movienet_dir = self.data_dir / "movienet"

        # MovieNet visual data
        self.shot_keyf_dir = self.movienet_dir / "shot_keyf"
        self.shot_txt_dir = self.movienet_dir / "shot_txt"
        self.raw_videos_dir = self.data_dir / "raw_videos"

        # Movie subset data (19 movies)
        self.movie_subset_dir = self.project_root / "movie_data_subset_20"
        self.annotation_dir = self.movie_subset_dir / "annotation"
        self.meta_dir = self.movie_subset_dir / "meta"
        self.script_dir = self.movie_subset_dir / "script"
        self.subtitle_dir = self.movie_subset_dir / "subtitle"

        # Index storage
        self.index_dir = self.data_dir / "indexes"

        # Create directories if they don't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.rawdata_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model configuration."""

    # CLIP settings
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_batch_size: int = 32

    # FAISS settings
    faiss_index_type: str = "IndexFlatIP"  # Inner Product for cosine similarity

    # Device
    device: str = field(default_factory=lambda: "cuda" if _cuda_available() else "cpu")


@dataclass
class IndexConfig:
    """Index configuration."""

    # Visual index (legacy)
    visual_index_name: str = "visual_index.faiss"
    visual_index_map_name: str = "visual_index_map.json"

    # Hierarchical indexes (HierVL/SceneRAG)
    frame_index_name: str = "movie_frame_index.faiss"
    scene_index_name: str = "movie_scene_index.faiss"
    knowledge_index_name: str = "movie_knowledge_index.faiss"

    # Scene aggregation
    scene_window_size: int = 5  # Group N consecutive shots into a scene

    # Keyframe extraction
    keyframe_fps: float = 1.0  # Extract 1 frame per second
    max_frames_per_video: int = 100


@dataclass
class Config:
    """Main configuration class."""

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()

        # Override with environment variables if present
        if os.getenv("MOVIERAG_DATA_DIR"):
            config.paths.data_dir = Path(os.getenv("MOVIERAG_DATA_DIR"))

        if os.getenv("MOVIERAG_DEVICE"):
            config.model.device = os.getenv("MOVIERAG_DEVICE")

        return config


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
