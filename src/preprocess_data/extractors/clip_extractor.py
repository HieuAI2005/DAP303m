import json
import logging
import subprocess
from pathlib import Path

from preprocess_data.config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)

class ClipExtractor:
    """Extracts physical video clips for each semantic scene using FFmpeg."""

    def __init__(self):
        pass

    def extract_clips(self, movie_id: str, video_path: Path, force: bool = False) -> bool:
        # Try both flat path (from Cfg.get_annotation_dir) and movie-id-nested path
        ann_path = Cfg.get_annotation_dir() / f"{movie_id}.json"
        if not ann_path.exists():
            # Try nested: OUTPUT_DIR/movie_id/annotation/movie_id.json
            if Cfg.OUTPUT_DIR is not None:
                ann_path = Cfg.OUTPUT_DIR / movie_id / "annotation" / f"{movie_id}.json"
        
        if not ann_path.exists():
            logger.error(f"Cannot extract clips without annotation file: {ann_path}")
            return False
            
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load annotation file: {e}")
            return False
            
        scenes = ann_data.get("story", [])
        if not scenes:
            scenes = ann_data.get("scenes", [])
        if not scenes:
            scenes = ann_data.get("scene", [])
            
        if not scenes:
            logger.warning("No scenes or story found in annotation.")
            return False
            
        # Determine clips directory from OUTPUT_DIR or fall back to sibling of annotation dir
        if Cfg.OUTPUT_DIR is not None:
            clips_dir = Cfg.OUTPUT_DIR / movie_id / "clips"
        else:
            # Fallback: write clips next to annotation, in data/movie_output/<movie_id>/clips
            clips_dir = ann_path.parent.parent / movie_id / "clips"
        
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_count = 0
        total_scenes = len(scenes)
        
        for i, scene in enumerate(scenes):
            scene_id = scene.get("id")
            start_sec = scene.get("start_seconds")
            end_sec = scene.get("end_seconds")
            
            if scene_id is None or start_sec is None or end_sec is None:
                continue
                
            clip_name = f"{movie_id}_{scene_id}.mp4"
            clip_path = clips_dir / clip_name
            
            if clip_path.exists() and not force:
                continue
                
            duration = end_sec - start_sec
            
            # Using precise cut with re-encoding (veryfast preset to speed up, low crf as it's just for analysis)
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start_sec),
                "-i", str(video_path),
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "28",
                "-c:a", "aac",
                str(clip_path)
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                extracted_count = extracted_count + 1
                if i % 10 == 0 or i == total_scenes - 1:
                    logger.info(f"  Progress: Extracted {i+1}/{total_scenes} clips...")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract {clip_name}: {e}")
                
        logger.info(f"Clip extraction complete. Extracted {extracted_count}/{total_scenes} new clips.")
        return True
