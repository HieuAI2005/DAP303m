"""
movierag/preprocessing/auto_annotator/shot_detector.py
Core shot detection using PySceneDetect with multiple detector strategies
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ShotInfo:
    """Thông tin một shot/cut."""
    shot_idx: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    thumbnail_path: Optional[str] = None


@dataclass
class DetectionConfig:
    """Configuration cho shot detection."""
    # Detector type
    detector_type: str = "adaptive"  # adaptive, content, threshold
    
    # AdaptiveDetector params
    adaptive_threshold: float = 3.0
    min_scene_len: int = 15
    min_cut_len: int = 2
    
    # ContentDetector params
    content_threshold: float = 27.0
    show_progress: bool = False
    
    # Thumbnail config
    save_thumbnails: bool = True
    thumbnails_per_shot: int = 3
    thumbnail_output_dir: Optional[str] = None


class ShotDetector:
    """
    Shot detection sử dụng PySceneDetect với multiple strategies.
    
    SUPPORTED DETECTORS:
    - AdaptiveDetector: Phát hiện cut dựa trên luminance change adaptive
    - ContentDetector: Phát hiện dựa trên histogram difference
    - ThresholdDetector: Simple threshold-based detection
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self._video = None
        self._scene_manager = None

    def detect_shots(
        self,
        video_path: Path,
        movie_id: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> List[ShotInfo]:
        """
        Detect shots từ video file.
        
        Args:
            video_path: Path đến video file
            movie_id: Movie identifier cho thumbnail naming
            output_dir: Output directory cho thumbnails
        
        Returns:
            List of ShotInfo objects
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"🎬 Detecting shots for {video_path.name}...")
        
        try:
            from scenedetect import SceneManager, open_video
            
            # Initialize video
            self._video = open_video(str(video_path))
            fps = self._video.frame_rate
            
            # Initialize scene manager with appropriate detector
            self._scene_manager = SceneManager()
            detector = self._get_detector()
            self._scene_manager.add_detector(detector)
            
            # Detect scenes
            logger.info(f"  Running {self.config.detector_type} detector...")
            self._scene_manager.detect_scenes(
                self._video,
                show_progress=self.config.show_progress
            )
            
            scene_list = self._scene_manager.get_scene_list()
            logger.info(f"  Detected {len(scene_list)} shots")
            
            # Save thumbnails if enabled
            thumbnail_paths = []
            if self.config.save_thumbnails and output_dir:
                thumbnail_paths = self._save_thumbnails(
                    scene_list, movie_id, output_dir
                )
            
            # Convert to ShotInfo objects
            shots = []
            for i, scene in enumerate(scene_list):
                start_time, end_time = scene
                start_sec = start_time.get_seconds()
                end_sec = end_time.get_seconds()
                duration_sec = end_sec - start_sec
                
                shot = ShotInfo(
                    shot_idx=i,
                    start_frame=start_time.get_frames(),
                    end_frame=end_time.get_frames() - 1,
                    start_sec=round(start_sec, 3),
                    end_sec=round(end_sec, 3),
                    duration_sec=round(duration_sec, 3),
                    thumbnail_path=thumbnail_paths[i] if i < len(thumbnail_paths) else None
                )
                
                # Filter very short shots
                if duration_sec >= 0.3:  # Minimum 300ms
                    shots.append(shot)
            
            logger.info(f"  ✅ Final: {len(shots)} valid shots (filtered <300ms)")
            return shots
            
        except ImportError as e:
            logger.error(f"PySceneDetect not installed: {e}")
            logger.info("  Falling back to fixed-interval shots...")
            return self._fixed_interval_shots(video_path, fps=30.0)
            
        except Exception as e:
            logger.error(f"Shot detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._fixed_interval_shots(video_path, fps=30.0)
        
        finally:
            # VideoStream handles its own resources in newer scenedetect
            pass

    def _get_detector(self):
        """Get detector based on config."""
        if self.config.detector_type == "adaptive":
            from scenedetect.detectors import AdaptiveDetector
            return AdaptiveDetector(
                adaptive_threshold=self.config.adaptive_threshold,
                min_scene_len=self.config.min_scene_len
            )
        elif self.config.detector_type == "content":
            from scenedetect.detectors import ContentDetector
            return ContentDetector(
                threshold=self.config.content_threshold,
                min_scene_len=self.config.min_scene_len
            )
        elif self.config.detector_type == "threshold":
            from scenedetect.detectors import ThresholdDetector
            return ThresholdDetector(
                min_scene_len=self.config.min_scene_len
            )
        else:
            from scenedetect.detectors import AdaptiveDetector
            return AdaptiveDetector()

    def _save_thumbnails(
        self,
        scene_list: List,
        movie_id: Optional[str],
        output_dir: Path
    ) -> List[str]:
        """Save thumbnails for each shot."""
        try:
            from scenedetect.scene_manager import save_images
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate template
            movie_prefix = f"{movie_id}_" if movie_id else ""
            template = f"{movie_prefix}shot_$SCENE_NUMBER-$IMAGE_NUMBER.jpg"
            
            # Save thumbnails (1 per shot for efficiency)
            save_images(
                scene_list=scene_list,
                video=self._video,
                num_images=self.config.thumbnails_per_shot,
                output_dir=str(output_dir),
                image_name_template=template,
                show_progress=False
            )
            
            # Generate paths
            thumbnail_paths = []
            for i in range(len(scene_list)):
                for j in range(self.config.thumbnails_per_shot):
                    # PySceneDetect numbers scenes from 1
                    path = output_dir / f"{movie_prefix}shot_{i+1:03d}-{j+1:02d}.jpg"
                    # Note: Template might vary based on num_images. 
                    # We'll just list directory for exact paths if needed.
                # Simplification for mapping back:
                # Based on PySceneDetect default naming for save_images
                # We'll try to find them.
            
            # Better way: list the output dir
            all_files = sorted(list(output_dir.glob("*.jpg")))
            return [str(f) for f in all_files]
            
        except Exception as e:
            logger.warning(f"Thumbnail saving failed: {e}")
            return []

    def _fixed_interval_shots(
        self,
        video_path: Path,
        fps: float = 30.0,
        interval_sec: float = 5.0
    ) -> List[ShotInfo]:
        """Fallback: create shots at fixed intervals."""
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True
            )
            duration = float(result.stdout.strip())
        except:
            duration = 300.0  # Default 5 minutes
        
        shots = []
        t = 0.0
        shot_idx = 0
        
        while t < duration:
            end_t = min(t + interval_sec, duration)
            shot = ShotInfo(
                shot_idx=shot_idx,
                start_frame=int(t * fps),
                end_frame=int(end_t * fps) - 1,
                start_sec=round(t, 3),
                end_sec=round(end_t, 3),
                duration_sec=round(end_t - t, 3)
            )
            shots.append(shot)
            shot_idx += 1
            t += interval_sec
        
        logger.info(f"  Created {len(shots)} fixed-interval shots")
        return shots
