import os
import subprocess
import logging
from typing import List

logger = logging.getLogger(__name__)


def extract_keyframes_from_video(
    video_path: str, output_dir: str, num_frames: int = 5
) -> List[str]:
    """
    Extracts N evenly spaced keyframes from an uploaded video using ffmpeg.

    Args:
        video_path: Path to the uploaded MP4 file.
        output_dir: Directory to save the extracted frames.
        num_frames: Total number of frames to extract.

    Returns:
        List of absolute paths to the extracted frame images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # First get video duration
    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    try:
        duration_str = subprocess.check_output(duration_cmd).decode("utf-8").strip()
        duration = float(duration_str)
    except Exception as e:
        logger.warning(f"Failed to get video duration: {e}. Falling back to 10s.")
        duration = 10.0

    if duration <= 0:
        duration = 10.0

    # Calculate interval
    interval = duration / (num_frames + 1)

    extracted_frames = []
    for i in range(1, num_frames + 1):
        timestamp = i * interval
        out_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")

        extract_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-q:v",
            "2",
            out_path,
        ]

        try:
            subprocess.run(
                extract_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            if os.path.exists(out_path):
                extracted_frames.append(out_path)
        except Exception as e:
            logger.error(f"Failed to extract frame {i} at {timestamp}s: {e}")

    logger.info(f"Extracted {len(extracted_frames)} frames from {video_path}")
    return extracted_frames
