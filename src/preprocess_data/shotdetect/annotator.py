"""
Auto Annotator — Shot Detection → Annotation JSON

Uses movienet_tools shotdetect (ContentDetectorHSVLUV) or ffmpeg I-frame
fallback to detect shot boundaries and generate annotation JSON equivalent
to MovieNet's annotation format.

For new videos without any pre-existing annotation data.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from preprocess_data.config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class AutoAnnotator:
    """Generate annotation JSON from shot detection on raw video."""

    def __init__(self):
        # We now explicitly rely on PySceneDetect which is vastly superior to crude ffmpeg
        pass

    def annotate(self, movie_id: str, video_path: Path = None) -> Dict:
        """
        Detect shots and generate annotation JSON for a movie.

        Returns: annotation dict compatible with MovieNet format
        """
        if video_path is None:
            video_path = Cfg.get_video_path(movie_id)
        if video_path is None or not video_path.exists():
            logger.error(f"  ❌ No video for {movie_id}")
            return {}

        logger.info(f"  🎬 Auto-annotating: {movie_id}")

        # Get video info
        from ..extractors.keyframe_extractor import KeyframeExtractor

        info = KeyframeExtractor.get_video_info(video_path)
        fps = info["fps"]
        duration = info["duration"]

        # Detect shots and potential physical scenes using PySceneDetect directly
        shots, physical_scenes = self._detect_with_scenedetect(video_path, fps, movie_id)

        if not shots:
            logger.warning(f"  No shots detected, using fixed interval")
            shots = self._fixed_interval_shots(duration, fps)
            physical_scenes = self._group_shots_into_scenes(shots, fps)

        # Build annotation JSON
        annotation = {
            "movie_id": movie_id,
            "fps": fps,
            "duration": duration,
            "total_shots": len(shots),
            "total_scenes": len(physical_scenes),
            "auto_generated": True,
            "raw_shots": shots,
            "scene": physical_scenes, # These are physical scenes (Camera Cuts + Visual Change)
        }

        # Save
        out_dir = Cfg.get_annotation_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        ann_path = out_dir / f"{movie_id}.json"
        ann_path.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(
            f"  ✅ Auto-annotation: {len(shots)} shots → {len(physical_scenes)} scenes → {ann_path}"
        )
        return annotation

    def _detect_with_scenedetect(self, video_path: Path, fps: float, movie_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Use PySceneDetect ContentDetector to accurately find shots and save thumbnails."""
        logger.info("  Using PySceneDetect (AdaptiveDetector) with Image Saving...")
        try:
            from scenedetect import SceneManager, open_video, AdaptiveDetector
            from scenedetect.scene_manager import save_images
            
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(AdaptiveDetector())
            
            # Perform detection
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
            
            # Save thumbnails for EACH shot
            # Logic: Save 1, 2, or more depending on duration
            image_dir = Cfg.OUTPUT_DIR / movie_id / "shot_images" if Cfg.OUTPUT_DIR else Cfg.PROJECT_ROOT / "shot_images" / movie_id
            image_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"  📸 Saving thumbnails to {image_dir}...")
            try:
                video.seek(0)  # reset stream position after detect_scenes()
                save_images(
                    scene_list=scene_list,
                    video=video,
                    num_images=3,
                    output_dir=str(image_dir),
                    image_name_template=f"{movie_id}_shot_$SCENE_NUMBER-$IMAGE_NUMBER",
                    show_progress=False
                )
                saved = list(image_dir.glob("*.jpg"))
                logger.info(f"  📸 Saved {len(saved)} thumbnails")
            except Exception as img_err:
                logger.warning(f"  save_images failed: {img_err} — keyframe_extractor will fallback")
            
            shots = []
            for i, scene in enumerate(scene_list):
                start_time, end_time = scene
                start_sec, end_sec = start_time.get_seconds(), end_time.get_seconds()
                
                if (end_sec - start_sec) >= 0.5:
                    shots.append({
                        "shot_idx": len(shots),
                        "start_frame": start_time.get_frames(),
                        "end_frame": end_time.get_frames() - 1,
                        "start_sec": round(start_sec, 3),
                        "end_sec": round(end_sec, 3),
                    })

            # Detect "Physical Scenes" (Groups of shots with large visual change)
            # We can use a higher threshold ContentDetector or just group by shot count for now
            # but user likes "Tool" scenes, so let's try a second pass with higher threshold
            scene_manager_phys = SceneManager()
            scene_manager_phys.add_detector(AdaptiveDetector(adaptive_threshold=8.0)) # Higher threshold
            video.seek(0)
            scene_manager_phys.detect_scenes(video)
            phys_list = scene_manager_phys.get_scene_list()
            
            physical_scenes = []
            for i, p_scene in enumerate(phys_list):
                p_start, p_end = p_scene
                # Map this physical scene to shot indices
                shot_indices = []
                for s_idx, s in enumerate(shots):
                    # Check if shot center is within physical scene
                    s_mid = (s["start_sec"] + s["end_sec"]) / 2.0
                    if p_start.get_seconds() <= s_mid <= p_end.get_seconds():
                        shot_indices.append(s_idx)
                
                if shot_indices:
                    physical_scenes.append({
                        "id": f"scene_{i}",
                        "shot": [shot_indices[0], shot_indices[-1]],
                        "frame": [shots[shot_indices[0]]["start_frame"], shots[shot_indices[-1]]["end_frame"]],
                        "start_seconds": shots[shot_indices[0]]["start_sec"],
                        "end_seconds": shots[shot_indices[-1]]["end_sec"],
                    })

            return shots, physical_scenes
        except Exception as e:
            logger.warning(f"  PySceneDetect failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return [], []

    @staticmethod
    def _fixed_interval_shots(
        duration: float, fps: float, interval: float = 5.0
    ) -> List[Dict]:
        """Fallback: create shots at fixed intervals."""
        shots = []
        t = 0.0
        while t < duration:
            end_t = min(t + interval, duration)
            shots.append(
                {
                    "shot_idx": len(shots),
                    "start_frame": int(t * fps),
                    "end_frame": int(end_t * fps) - 1,
                    "start_sec": round(t, 3),
                    "end_sec": round(end_t, 3),
                }
            )
            t += interval
        return shots

    @staticmethod
    def _group_shots_into_scenes(
        shots: List[Dict], fps: float, max_shots_per_scene: int = 8
    ) -> List[Dict]:
        """Group consecutive shots into scenes."""
        if not shots:
            return []

        scenes = []
        current_shots = []

        for shot in shots:
            current_shots.append(shot)
            if len(current_shots) >= max_shots_per_scene:
                scenes.append(_make_scene(scenes, current_shots, fps))
                current_shots = []

        if current_shots:
            scenes.append(_make_scene(scenes, current_shots, fps))

        return scenes


def _make_scene(existing_scenes, shot_group, fps):
    """Create a scene dict from a group of shots."""
    idx = len(existing_scenes)
    first, last = shot_group[0], shot_group[-1]
    start_frame = first.get("start_frame", 0)
    end_frame = last.get("end_frame", 0)
    return {
        "id": f"scene_{idx}",
        "shot": [first["shot_idx"], last["shot_idx"]],
        "frame": [start_frame, end_frame],
        "start_seconds": first.get("start_sec", round(start_frame / fps, 3)),
        "end_seconds": last.get("end_sec", round(end_frame / fps, 3)),
    }
