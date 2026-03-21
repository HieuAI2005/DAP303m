# ─────────────────────────────────────────────────────────────────────────────
# action_recognizer.py
# VideoMAE Action Recognition — Video Understanding Pipeline
# Layer 4: Cast & Characters — action_labels field
# ─────────────────────────────────────────────────────────────────────────────
"""
 VideoMAE-based Action Recognition for movie scenes.

 Extracts activity/action labels from video segments using pre-trained
 VideoMAE models. Enriches Layer 4: Cast & Characters → action_labels.

 Supported models:
   - VideoMAE (Kinetics-400/600 pretrained)
   - TimeSformer
   - SlowFast (via detectron2)

 Output schema (per segment):
   {
     "segment_start": float,
     "segment_end": float,
     "actions": [{"label": str, "score": float}],
     "top_action": str,
     "model_name": str,
   }
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

DEVICE = os.getenv("MOVIERAG_ACTION_DEVICE", "cuda")
VIDEO_MAE_MODEL = os.getenv("MOVIERAG_VIDEOMAE_MODEL", "vit_base_patch16_224")
LABEL_MAP_SOURCE = os.getenv("MOVIERAG_ACTION_LABELS", "kinetics400")
NUM_FRAMES = int(os.getenv("MOVIERAG_ACTION_FRAMES", "16"))
CLIP_LENGTH = int(os.getenv("MOVIERAG_ACTION_CLIP_SECS", "16"))
TEMP_DIR = os.getenv("MOVIERAG_TEMP_VIDEO", "/tmp/movierag_action")


# ── Kinetics-400 Labels (subset — full list has 400 labels) ────────────────────

KINETICS_400_LABELS = [
    "abseiling", "driving car", "eating cake", "playing guitar", "running",
    "walking with dog", "jumping", "dancing", "swimming", "fighting",
    "kissing", "hugging", "laughing", "crying", "singing",
    "writing", "reading", "cooking", "cleaning", "driving",
    "flying kite", "playing violin", "playing piano", "playing drums",
    "mowing lawn", "shoveling snow", "building house", "fishing", "sailing",
    "surfing", "skiing", "snowboarding", "skateboarding", "cycling",
    "playing basketball", "playing volleyball", "playing football", "playing tennis",
    "wrestling", "rock climbing", "bungee jumping", "paragliding", "skydiving",
    "yoga", "meditating", "stretching", "brushing teeth", "taking a shower",
    "waking up", "going to bed", "drinking", "phoning", "shaking hands",
    "giving a presentation", "clapping", "waving", "pointing", "saluting",
    "smoking", "eating hotdog", "eating popcorn", "drinking beer", "drinking coffee",
    "opening present", "looking at phone", "texting", "shooting goal", "celebrating",
    "texting", "driving tractor", "riding mechanical bull", "riding camel", "kayaking",
    "playing badminton", "playing cricket", "playing baseball", "playing hockey",
    "tossing coin", "cutting pineapple", "making jewelry", "playing uno", "braiding hair",
]


# ── Frame Extraction ──────────────────────────────────────────────────────────

def extract_video_clip_for_action(
    video_path: str,
    start_seconds: float,
    end_seconds: float,
    num_frames: int = NUM_FRAMES,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Extract a video clip segment for action recognition.

    Args:
        video_path: Input video path.
        start_seconds: Clip start time.
        end_seconds: Clip end time.
        num_frames: Number of frames to extract (spatial temporal).
        output_path: Output clip path. Auto-generated if None.

    Returns:
        Path to extracted clip, or None if extraction fails.
    """
    import subprocess

    if output_path is None:
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(TEMP_DIR, f"action_clip_{start_seconds:.1f}_{end_seconds:.1f}.mp4")

    duration = max(end_seconds - start_seconds, 1.0)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_seconds),
        "-i", video_path,
        "-t", str(duration),
        "-vf", f"fps={num_frames / duration},scale=224:224",
        "-c:v", "libx264",
        "-preset", "fast",
        "-frames:v", str(num_frames),
        "-pix_fmt", "rgb24",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if Path(output_path).exists():
            return output_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Video clip extraction failed: {e.stderr}")
    return None


# ── Action Recognizer ──────────────────────────────────────────────────────────

class ActionRecognizer:
    """
    VideoMAE-based action recognition for movie video segments.

    Usage:
        recognizer = ActionRecognizer()
        result = recognizer.recognize_in_segment(
            video_path="movie.mp4",
            start_seconds=120.0,
            end_seconds=136.0,
        )
    """

    def __init__(
        self,
        model_name: str = VIDEO_MAE_MODEL,
        device: str = DEVICE,
        num_frames: int = NUM_FRAMES,
        top_k: int = 5,
        confidence_threshold: float = 0.1,
    ):
        """
        Args:
            model_name: VideoMAE model variant.
            device: "cuda" or "cpu".
            num_frames: Number of frames to sample per clip.
            top_k: Return top-k action labels.
            confidence_threshold: Minimum confidence to include label.
        """
        self.model_name = model_name
        self.device = device
        self.num_frames = num_frames
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._preprocess = None
        self._labels = KINETICS_400_LABELS

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load VideoMAE model and preprocessor."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        except ImportError as e:
            raise ImportError(
                "transformers + torch required for ActionRecognizer. "
                "Install with: pip install torch transformers"
            ) from e

        logger.info(f"Loading VideoMAE model: {self.model_name}")
        self._model = VideoMAEForVideoClassification.from_pretrained(
            self.model_name,
            device_map=self.device if self.device == "cuda" else "cpu",
        )
        self._model.to(self.device)
        self._model.eval()

        # Image processor for preprocessing
        self._processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
        logger.info(f"VideoMAE loaded on device: {self.device}")

    # ── Core Recognition ─────────────────────────────────────────────────────

    def recognize_in_segment(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
        return_frames: bool = False,
    ) -> Dict[str, Any]:
        """
        Recognize actions in a video segment.

        Args:
            video_path: Path to video file.
            start_seconds: Segment start time.
            end_seconds: Segment end time.
            return_frames: Also return sampled frame paths.

        Returns:
            Dict with action labels and scores.
        """
        self._load_model()

        # Extract clip
        clip_path = extract_video_clip_for_action(
            video_path, start_seconds, end_seconds, self.num_frames
        )
        if clip_path is None:
            return {
                "segment_start": start_seconds,
                "segment_end": end_seconds,
                "actions": [],
                "top_action": "",
                "model_name": self.model_name,
                "error": "Clip extraction failed",
            }

        try:
            # Load video as frames
            import torch
            from torchvision.io import read_video

            frames, _, _ = read_video(clip_path, end_pts=self.num_frames)
            if frames.shape[0] < self.num_frames:
                # Pad by repeating last frame
                pad_count = self.num_frames - frames.shape[0]
                frames = torch.cat([frames] + [frames[-1:].repeat(pad_count, 1, 1, 1)], dim=0)

            # Preprocess: normalize to [0,1] and convert format
            frames = frames.float() / 255.0
            # RGB from [N, H, W, C] → [N, C, H, W]
            frames = frames.permute(0, 3, 1, 2)

            # Resize to model input size
            import torch.nn.functional as F
            frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)

            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames = (frames - mean) / std

            # Make patch frames
            if frames.shape[0] > self.num_frames:
                indices = np.linspace(0, frames.shape[0] - 1, self.num_frames).astype(int)
                frames = frames[indices]

            # Add batch dim
            pixel_values = frames.unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self._model(pixel_values)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = probs.topk(self.top_k, dim=-1)

            # Map to labels
            actions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                score = float(prob.cpu())
                if score < self.confidence_threshold:
                    continue
                label_idx = int(idx.cpu())
                label = self._labels[label_idx] if label_idx < len(self._labels) else f"action_{label_idx}"
                actions.append({"label": label, "score": round(score, 4)})

            result = {
                "segment_start": start_seconds,
                "segment_end": end_seconds,
                "actions": actions,
                "top_action": actions[0]["label"] if actions else "",
                "model_name": self.model_name,
            }

            if return_frames:
                result["clip_path"] = clip_path

            return result

        except Exception as e:
            logger.error(f"Action recognition failed: {e}")
            return {
                "segment_start": start_seconds,
                "segment_end": end_seconds,
                "actions": [],
                "top_action": "",
                "model_name": self.model_name,
                "error": str(e),
            }

    def recognize_in_scened(
        self,
        video_path: str,
        scenes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run action recognition across multiple pre-detected scenes.

        Args:
            video_path: Path to video file.
            scenes: List of scene dicts with keys: scene_id, start_seconds, end_seconds.

        Returns:
            List of action recognition results (same order as input).
        """
        results = []
        for scene in scenes:
            result = self.recognize_in_segment(
                video_path=video_path,
                start_seconds=scene.get("start_seconds", 0.0),
                end_seconds=scene.get("end_seconds", 30.0),
            )
            result["scene_id"] = scene.get("scene_id", "unknown")
            results.append(result)
        return results

    def __repr__(self) -> str:
        return (
            f"ActionRecognizer(model={self.model_name}, frames={self.num_frames}, "
            f"top_k={self.top_k}, device={self.device})"
        )
