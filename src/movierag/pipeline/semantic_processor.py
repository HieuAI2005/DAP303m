"""
Semantic Video Processing for Continuous-Time Contexts.
Uses PySceneDetect to find color-gradient scene cuts, providing
physically accurate temporal boundaries instead of static 1-FPS chunking.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

logger = logging.getLogger(__name__)


class SemanticVideoProcessor:
    """
    Parses long native MP4 videos into continuous segments (Scenes)
    based on visual semantic boundaries (cuts/fades) rather than static chunking.
    Extracts representative keyframes from each semantic scene.
    """

    def __init__(self, threshold: float = 27.0):
        """
        Args:
            threshold: PySceneDetect ContentDetector sensitivity. Lower = more scenes.
        """
        self.threshold = threshold

    def detect_scenes(self, video_path: str) -> List[Tuple[float, float, int, int]]:
        """
        Perform semantic scene detection on a video.

        Returns:
            List of Tuples: (start_time_sec, end_time_sec, start_frame, end_frame)
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []

        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))

        video_manager.set_downscale_factor()
        video_manager.start()

        logger.info(
            f"Running continuous semantic scene detection on {os.path.basename(video_path)}..."
        )
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()

        # Convert to a standard list format
        scenes = []
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            scenes.append((start_time, end_time, start_frame, end_frame))

        logger.info(f"Detected {len(scenes)} physical scene boundaries.")
        return scenes

    def process_video_to_semantic_frames(
        self, video_path: str, output_dir: str, frames_per_scene: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Detects semantic scenes, extracts N frames from each, and generates the continuous-time mapping.

        Args:
            video_path: Path to the input .mp4
            output_dir: Where to save the temporary extracted JPEG frames
            frames_per_scene: Number of equidistant frames to extract per scene segment

        Returns:
            A list of semantic segments:
            [
               {"scene_id": int, "start_time": float, "end_time": float, "frame_paths": [str]}
            ]
        """
        scenes = self.detect_scenes(video_path)
        if not scenes:
            return []

        os.makedirs(output_dir, exist_ok=True)
        vid_cap = cv2.VideoCapture(video_path)

        semantic_segments = []

        for scene_idx, (start_sec, end_sec, start_frame, end_frame) in enumerate(
            scenes
        ):
            scene_duration = end_frame - start_frame
            if scene_duration <= 0:
                continue

            # Pick equidistant frames inside the scene barrier
            step = max(1, scene_duration // frames_per_scene)
            frame_indices = [start_frame + (i * step) for i in range(frames_per_scene)]

            # Ensure we don't pick outside the boundary
            frame_indices = [min(idx, end_frame - 1) for idx in frame_indices]

            saved_paths = []
            for count, idx in enumerate(set(frame_indices)):
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = vid_cap.read()
                if success:
                    out_path = os.path.join(
                        output_dir, f"scene_{scene_idx}_frame_{count}.jpg"
                    )
                    cv2.imwrite(out_path, frame)
                    saved_paths.append(out_path)

            if saved_paths:
                semantic_segments.append(
                    {
                        "scene_id": scene_idx,
                        "start_time": start_sec,
                        "end_time": end_sec,
                        "frame_paths": saved_paths,
                    }
                )

        vid_cap.release()
        return semantic_segments
