"""
movierag/preprocessing/auto_annotator/core.py
Main AutoAnnotator class that orchestrates all components
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .shot_detector import ShotDetector, ShotInfo, DetectionConfig
from .scene_grouper import SceneGrouper, SceneInfo, SceneGroupingConfig
from .annotation_formatter import AnnotationFormatter

logger = logging.getLogger(__name__)


class AutoAnnotator:
    """
    Auto Annotator - Generate annotation JSON from shot detection on raw video.
    
    PIPELINE:
    1. Detect shots using PySceneDetect
    2. Group shots into physical scenes
    3. Format to MovieNet-compatible JSON
    4. Save annotation file
    """

    def __init__(
        self,
        detection_config: Optional[DetectionConfig] = None,
        grouping_config: Optional[SceneGroupingConfig] = None,
        output_dir: Optional[Path] = None
    ):
        self.detection_config = detection_config or DetectionConfig()
        self.grouping_config = grouping_config or SceneGroupingConfig()
        self.output_dir = output_dir
        
        # Initialize components
        self.shot_detector = ShotDetector(self.detection_config)
        self.scene_grouper = SceneGrouper(self.grouping_config)
        self.formatter = AnnotationFormatter(self.output_dir)

    def annotate(
        self,
        movie_id: str,
        video_path: Optional[Path] = None,
        force_regenerate: bool = False,
        existing_annotation_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Detect shots and generate annotation JSON for a movie.
        """
        # Check for existing annotation
        if not force_regenerate and existing_annotation_dir:
            existing = self.formatter.load_annotation(movie_id, existing_annotation_dir)
            if existing:
                logger.info(f"✅ Using existing annotation for {movie_id}")
                return existing
        
        if video_path is None or not video_path.exists():
            logger.error(f"❌ No video for {movie_id} at {video_path}")
            return {}
        
        logger.info(f"🎬 Auto-annotating: {movie_id}")
        
        try:
            # Step 1: Get video info
            video_info = self._get_video_info(video_path)
            fps = video_info.get("fps", 30.0)
            
            # Step 2: Detect shots
            thumbnail_dir = self.output_dir / movie_id / "shot_images" if self.output_dir else None
            shots = self.shot_detector.detect_shots(
                video_path=video_path,
                movie_id=movie_id,
                output_dir=thumbnail_dir
            )
            
            if not shots:
                logger.warning(f"  No shots detected, using fixed interval")
                shots = self.shot_detector._fixed_interval_shots(video_path, fps=fps)
            
            # Step 3: Group shots into scenes
            scenes = self.scene_grouper.group_shots(
                shots=shots,
                video_path=str(video_path),
                thumbnail_dir=str(thumbnail_dir) if thumbnail_dir else None
            )
            
            # Step 4: Format annotation
            annotation = self.formatter.format_annotation(
                movie_id=movie_id,
                shots=shots,
                scenes=scenes,
                video_info=video_info
            )
            
            # Step 5: Save annotation
            if self.output_dir:
                ann_dir = self.output_dir / "annotations"
                ann_path = self.formatter.save_annotation(annotation, movie_id, ann_dir)
                annotation["annotation_path"] = str(ann_path)
            
            logger.info(
                f"✅ Auto-annotation complete: "
                f"{len(shots)} shots → {len(scenes)} scenes"
            )
            
            return annotation
            
        except Exception as e:
            logger.error(f"Annotation failed for {movie_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata using ffprobe."""
        import subprocess
        
        try:
            # Get duration
            duration_result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, timeout=30
            )
            duration = float(duration_result.stdout.strip())
            
            # Get fps
            fps_result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, timeout=30
            )
            fps_str = fps_result.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            return {"fps": fps, "duration": duration}
            
        except Exception as e:
            logger.warning(f"Failed to get video info: {e}")
            return {"fps": 30.0, "duration": 300.0}
