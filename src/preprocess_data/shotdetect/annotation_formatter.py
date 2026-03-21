"""
movierag/preprocessing/auto_annotator/annotation_formatter.py
Format annotations to MovieNet-compatible JSON
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .shot_detector import ShotInfo
from .scene_grouper import SceneInfo

logger = logging.getLogger(__name__)


class AnnotationFormatter:
    """
    Format shot/scene detection results to MovieNet-compatible JSON.
    
    MOVIEFORMAT SPEC:
    - movie_id: Unique identifier
    - fps: Video frame rate
    - duration: Video duration in seconds
    - total_shots: Number of shots
    - total_scenes: Number of physical scenes
    - raw_shots: List of shot dictionaries
    - scene: List of scene dictionaries
    - auto_generated: Flag for auto-generated annotations
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def format_annotation(
        self,
        movie_id: str,
        shots: List[ShotInfo],
        scenes: List[SceneInfo],
        video_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format annotation to MovieNet JSON structure.
        
        Args:
            movie_id: Movie identifier
            shots: List of ShotInfo objects
            scenes: List of SceneInfo objects
            video_info: Optional video metadata (fps, duration, etc.)
        
        Returns:
            Annotation dictionary
        """
        video_info = video_info or {}
        fps = video_info.get("fps", 30.0)
        duration = video_info.get("duration", shots[-1].end_sec if shots else 0)
        
        annotation = {
            "movie_id": movie_id,
            "fps": fps,
            "duration": round(duration, 3),
            "total_shots": len(shots),
            "total_scenes": len(scenes),
            "auto_generated": True,
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "raw_shots": self._format_shots(shots),
            "scene": self._format_scenes(scenes)
        }
        
        return annotation

    def _format_shots(self, shots: List[ShotInfo]) -> List[Dict[str, Any]]:
        """Format ShotInfo objects to dictionaries."""
        formatted = []
        for shot in shots:
            shot_dict = {
                "shot_idx": shot.shot_idx,
                "start_frame": shot.start_frame,
                "end_frame": shot.end_frame,
                "start_sec": shot.start_sec,
                "end_sec": shot.end_sec,
                "duration_sec": shot.duration_sec
            }
            if shot.thumbnail_path:
                shot_dict["thumbnail_path"] = shot.thumbnail_path
            formatted.append(shot_dict)
        return formatted

    def _format_scenes(self, scenes: List[SceneInfo]) -> List[Dict[str, Any]]:
        """Format SceneInfo objects to dictionaries."""
        formatted = []
        for scene in scenes:
            scene_dict = {
                "id": scene.scene_id,
                "scene_idx": scene.scene_idx,
                "shot": list(scene.shot_range),
                "frame": list(scene.frame_range),
                "start_seconds": scene.start_sec,
                "end_seconds": scene.end_sec,
                "duration_sec": scene.duration_sec,
                "num_shots": scene.num_shots,
                "visual_coherence": round(scene.visual_coherence, 4)
            }
            formatted.append(scene_dict)
        return formatted

    def save_annotation(
        self,
        annotation: Dict[str, Any],
        movie_id: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """Save annotation to JSON file."""
        output_dir = output_dir or self.output_dir
        if output_dir is None:
            raise ValueError("Output directory not specified")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{movie_id}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved annotation to {output_path}")
        return output_path

    def load_annotation(self, movie_id: str, annotation_dir: Path) -> Optional[Dict[str, Any]]:
        """Load existing annotation from JSON file."""
        annotation_path = annotation_dir / f"{movie_id}.json"
        
        if not annotation_path.exists():
            logger.warning(f"Annotation not found: {annotation_path}")
            return None
        
        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)
            logger.info(f"📖 Loaded annotation for {movie_id}")
            return annotation
        except Exception as e:
            logger.error(f"Failed to load annotation: {e}")
            return None
