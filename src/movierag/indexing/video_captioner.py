# ─────────────────────────────────────────────────────────────────────────────
# video_captioner.py
# Video Captioning Module — Video Understanding Pipeline
# Layer 2: Semantic Description — VLM-generated scene descriptions
# ─────────────────────────────────────────────────────────────────────────────
"""
 Video Captioning using Vision Language Models.

 Generates natural language captions for video segments, enriching
 Layer 2: Semantic Description in the 5-Layer Scene Metadata Model.

 Supports:
   - Single-frame captioning (fast, for dense coverage)
   - Multi-frame captioning (comprehensive, for key scenes)
   - Dense video captioning (ActivityNet-style event captions)
   - Temporal grounding captions (describe what happens when)

 Output schema (per segment):
   {
     "segment_id": str,
     "start_seconds": float,
     "end_seconds": float,
     "caption": str,                    # Generated natural language caption
     "situation": str,                  # Short situation label
     "emotional_tone": str,
     "setting": str,
     "actions": [str],
     "characters_mentioned": [str],
     "model": str,
     "is_key_frame": bool,
   }
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

CAPTION_MODEL = os.getenv("MOVIERAG_CAPTION_MODEL", "qwen-vl")
CAPTION_MAX_TOKENS = int(os.getenv("MOVIERAG_CAPTION_MAX_TOKENS", "256"))
CAPTION_TEMPERATURE = float(os.getenv("MOVIERAG_CAPTION_TEMPERATURE", "0.3"))
SAMPLE_FRAMES_PER_SEGMENT = int(os.getenv("MOVIERAG_CAPTION_SAMPLE_FRAMES", "4"))


# ── Caption Dataclass ──────────────────────────────────────────────────────────

@dataclass
class VideoCaption:
    """Generated caption for a video segment."""
    segment_id: str
    start_seconds: float
    end_seconds: float
    caption: str
    situation: str = ""
    emotional_tone: str = ""
    setting: str = ""
    actions: List[str] = field(default_factory=list)
    characters_mentioned: List[str] = field(default_factory=list)
    model: str = ""
    is_key_frame: bool = False
    raw_response: Optional[str] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "vlm_description": self.caption,
            "situation": self.situation,
            "emotional_tone": self.emotional_tone,
            "vision_setting": self.setting,
            "vision_actions": self.actions,
            "characters": self.characters_mentioned,
            "caption_model": self.model,
        }


# ── Video Captioner ────────────────────────────────────────────────────────────

class VideoCaptioner:
    """
    VLM-based video captioning.

    Usage:
        captioner = VideoCaptioner()
        caption = captioner.caption_segment("movie.mp4", 120.0, 150.0)
    """

    # System prompt for captioning
    CAPTION_PROMPT = """You are a film analyst describing a movie scene.

Describe what happens in this video segment in 1-3 sentences.
Focus on:
- What is the main action or event?
- Who are the characters involved?
- Where does the scene take place?
- What is the emotional tone?

Keep the description concise, factual, and informative.
"""

    DENSE_CAPTION_PROMPT = """You are watching a movie. For each event in the video segment, provide a timestamped description.

Output as JSON array:
[
  {{"timestamp": "00:01", "event": "description"}},
  {{"timestamp": "00:05", "event": "description"}}
]

Focus on: actions, dialogues, character movements, and emotional shifts.
Only describe what is clearly visible/audible.
"""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = CAPTION_MODEL,
        max_tokens: int = CAPTION_MAX_TOKENS,
        temperature: float = CAPTION_TEMPERATURE,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._llm_client = llm_client

    # ── Client ────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._llm_client is None:
            from movierag.generation.universal_client import UniversalLLMClient
            self._llm_client = UniversalLLMClient()
        return self._llm_client

    # ── Frame Extraction ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_frames(
        video_path: str,
        start: float,
        end: float,
        num_frames: int = SAMPLE_FRAMES_PER_SEGMENT,
    ) -> List[str]:
        """Extract evenly-spaced frames from video segment."""
        import subprocess
        from pathlib import Path

        temp_dir = Path("/tmp/movierag_caption")
        temp_dir.mkdir(parents=True, exist_ok=True)

        duration = max(end - start, 1.0)
        frame_paths: List[str] = []

        for i in range(num_frames):
            ts = start + (i / max(num_frames - 1, 1)) * duration
            out_path = str(temp_dir / f"frame_{i:02d}.jpg")

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(ts),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                "-s", "896x504",
                out_path,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                if Path(out_path).exists():
                    frame_paths.append(out_path)
            except subprocess.CalledProcessError:
                continue

        return frame_paths

    # ── Core Captioning ────────────────────────────────────────────────────────

    def caption_segment(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
        segment_id: str = "unknown",
        is_key_frame: bool = False,
    ) -> VideoCaption:
        """
        Generate a caption for a video segment.

        Args:
            video_path: Path to video file.
            start_seconds: Segment start time.
            end_seconds: Segment end time.
            segment_id: Unique segment identifier.
            is_key_frame: Whether this is a key moment scene.

        Returns:
            VideoCaption dataclass.
        """
        import base64

        # Extract frames
        frame_paths = self._extract_frames(video_path, start_seconds, end_seconds)

        if not frame_paths:
            return VideoCaption(
                segment_id=segment_id,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                caption="[No frames extracted]",
                model=self.model,
            )

        client = self._get_client()

        try:
            if len(frame_paths) == 1:
                result = client.generate_vision_content(
                    prompt=self.CAPTION_PROMPT,
                    image_path=frame_paths[0],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            else:
                # Multi-frame: encode to base64
                images_b64 = []
                for fp in frame_paths:
                    with open(fp, "rb") as f:
                        images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
                result = client.generate_multi_vision(
                    prompt=self.CAPTION_PROMPT,
                    images_base64=images_b64,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            if not result or result == "[No VLM available]":
                return VideoCaption(
                    segment_id=segment_id,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    caption="[VLM unavailable]",
                    model=self.model,
                )

            # Parse structured info from response
            parsed = self._parse_caption_response(result)

            return VideoCaption(
                segment_id=segment_id,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                caption=result.strip(),
                situation=parsed.get("situation", ""),
                emotional_tone=parsed.get("emotional_tone", ""),
                setting=parsed.get("setting", ""),
                actions=parsed.get("actions", []),
                characters_mentioned=parsed.get("characters", []),
                model=self.model,
                is_key_frame=is_key_frame,
                raw_response=result,
            )

        except Exception as e:
            logger.error(f"Video captioning failed: {e}")
            return VideoCaption(
                segment_id=segment_id,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                caption=f"[Error: {e}]",
                model=self.model,
            )

    def caption_dense(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
        segment_id: str = "unknown",
    ) -> List[VideoCaption]:
        """
        Generate dense timestamped captions for a video segment.

        Returns multiple VideoCaption objects, one per detected event.
        """
        import base64
        import json
        import re

        frame_paths = self._extract_frames(video_path, start_seconds, end_seconds, num_frames=8)

        if not frame_paths:
            return []

        client = self._get_client()

        try:
            images_b64 = []
            for fp in frame_paths:
                with open(fp, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode("utf-8"))

            result = client.generate_multi_vision(
                prompt=self.DENSE_CAPTION_PROMPT,
                images_base64=images_b64,
                max_tokens=self.max_tokens * 2,
                temperature=self.temperature,
            )

            # Parse JSON array from response
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                events = json.loads(json_match.group(0))
                captions = []
                for ev in events:
                    ts_str = ev.get("timestamp", "00:00")
                    ts_parts = ts_str.split(":")
                    ts_seconds = int(ts_parts[0]) * 60 + int(ts_parts[1])
                    abs_start = start_seconds + ts_seconds
                    captions.append(VideoCaption(
                        segment_id=f"{segment_id}_evt_{len(captions)}",
                        start_seconds=abs_start,
                        end_seconds=abs_start + 5.0,
                        caption=ev.get("event", ""),
                        model=self.model,
                    ))
                return captions

        except Exception as e:
            logger.warning(f"Dense captioning failed: {e}")

        # Fallback: single caption
        single = self.caption_segment(video_path, start_seconds, end_seconds, segment_id)
        return [single]

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_caption_response(self, text: str) -> Dict[str, Any]:
        """Extract structured info from free-form caption text."""
        import re

        result: Dict[str, Any] = {
            "situation": "",
            "emotional_tone": "",
            "setting": "",
            "actions": [],
            "characters": [],
        }

        text_lower = text.lower()

        # Detect situation keywords
        situation_kw = {
            "arguing": ["argue", "argument", "yelling", "shouting"],
            "romantic": ["kiss", "hug", "hold hands", "love", "romantic"],
            "chase": ["chase", "running away", "pursuit", "escape"],
            "comedic": ["laugh", "funny", "joke", "comedic", "slapstick"],
            "dramatic": ["dramatic", "intense", "serious", "dark"],
            "action": ["fight", "explosion", "shoot", "explode", "crash"],
            "sad": ["cry", "tears", "sad", "mourn", "grief"],
        }
        for label, keywords in situation_kw.items():
            if any(kw in text_lower for kw in keywords):
                result["situation"] = label
                break

        # Detect emotional tone
        tone_kw = {
            "tense": ["tense", "nervous", "anxious"],
            "joyful": ["happy", "joy", "laugh", "smile", "delighted"],
            "melancholic": ["sad", "blue", "gloomy", "somber"],
            "exciting": ["exciting", "thrilling", "adrenaline"],
            "calm": ["calm", "peaceful", "quiet", "serene"],
        }
        for label, keywords in tone_kw.items():
            if any(kw in text_lower for kw in keywords):
                result["emotional_tone"] = label
                break

        # Extract capitalized names as characters
        names = re.findall(r'\b([A-Z][a-z]+)\b', text)
        result["characters"] = list(dict.fromkeys(n))[:5]  # dedupe, limit 5

        # Extract action verbs
        action_kw = ["running", "walking", "sitting", "standing", "talking", "singing",
                      "dancing", "fighting", "driving", "eating", "sleeping"]
        result["actions"] = [kw for kw in action_kw if kw in text_lower]

        return result

    # ── Batch Captioning ────────────────────────────────────────────────────────

    def caption_scenes(
        self,
        video_path: str,
        scenes: List[Dict[str, Any]],
    ) -> List[VideoCaption]:
        """
        Generate captions for multiple pre-detected scenes.

        Args:
            video_path: Path to video file.
            scenes: List of scene dicts with keys: scene_id, start_seconds, end_seconds.

        Returns:
            List of VideoCaption objects.
        """
        captions = []
        for scene in scenes:
            cap = self.caption_segment(
                video_path=video_path,
                start_seconds=scene.get("start_seconds", 0.0),
                end_seconds=scene.get("end_seconds", 30.0),
                segment_id=scene.get("scene_id", "unknown"),
                is_key_frame=scene.get("is_key_scene", False),
            )
            captions.append(cap)
        return captions

    def __repr__(self) -> str:
        return f"VideoCaptioner(model={self.model}, max_tokens={self.max_tokens})"
