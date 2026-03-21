# ─────────────────────────────────────────────────────────────────────────────
# vlm_scene_analyzer.py
# VLM-Guided Scene Understanding — Video Understanding Pipeline
# Multi-frame Vision Language Model analysis for deep scene comprehension
# Layer 2: Semantic Description + Layer 4: Cast & Characters
# ─────────────────────────────────────────────────────────────────────────────
"""
 VLM-Guided Scene Understanding using Vision Language Models.

 Key capabilities:
   1. Multi-Frame Scene Analysis — analyze N frames per scene, fuse into coherent narrative
   2. Conflict Detection — compare VLM analysis against FAISS/metadata for consistency
   3. Query Distillation — convert VLM descriptions into search keywords
   4. Scene Description Generation — enrich Layer 2 (Semantic Description) metadata
   5. Character Emotion Tracking — track per-character emotions across scenes
   6. Action Recognition — detect and label physical actions from visual frames

 Output schema (per scene):
   {
     "scene_id": str,
     "movie_id": str,
     "vlm_description": str,           # Detailed VLM-generated description
     "situation": str,                 # e.g. "arguing", "romantic", "chase"
     "vision_setting": str,            # e.g. "beach at sunset"
     "vision_actions": [str],          # e.g. ["running", "fighting"]
     "emotional_tone": str,            # e.g. "tense", "romantic", "comedic"
     "characters_detected": [{"name": str, "emotion": str, "position": str}],
     "notable_objects": [str],
     "camera_style": str,              # e.g. "close-up", "wide shot"
     "vlm_conflict_detected": bool,
     "conflict_details": str | None,
     "distilled_keywords": [str],      # For search expansion
     "frame_count": int,
     "analyzed_at": str,               # ISO timestamp
   }
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

VLM_MODEL = os.getenv("MOVIERAG_VLM_MODEL", "gemini-2.0-flash")
VLM_MAX_FRAMES = int(os.getenv("MOVIERAG_VLM_MAX_FRAMES", "8"))
VLM_MAX_TOKENS = int(os.getenv("MOVIERAG_VLM_MAX_TOKENS", "2048"))
VLM_TEMPERATURE = float(os.getenv("MOVIERAG_VLM_TEMPERATURE", "0.2"))
VLM_BATCH_SIZE = int(os.getenv("MOVIERAG_VLM_BATCH_SIZE", "4"))
CONFLICT_THRESHOLD = float(os.getenv("MOVIERAG_CONFLICT_THRESHOLD", "0.6"))


# ── Scene Analysis Dataclass ───────────────────────────────────────────────────

@dataclass
class VLMAnalysis:
    """Result of VLM scene analysis."""
    scene_id: str
    movie_id: str
    vlm_description: str
    situation: str = ""
    vision_setting: str = ""
    vision_actions: List[str] = field(default_factory=list)
    emotional_tone: str = ""
    characters_detected: List[Dict[str, str]] = field(default_factory=list)
    notable_objects: List[str] = field(default_factory=list)
    camera_style: str = ""
    vlm_conflict_detected: bool = False
    conflict_details: Optional[str] = None
    distilled_keywords: List[str] = field(default_factory=list)
    frame_count: int = 0
    analyzed_at: str = ""
    raw_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to 5-Layer metadata dict for Layer 2 + Layer 4."""
        return {
            "vlm_description": self.vlm_description,
            "situation": self.situation,
            "vision_setting": self.vision_setting,
            "vision_actions": self.vision_actions,
            "emotional_tone": self.emotional_tone,
            "characters": [c["name"] for c in self.characters_detected],
            "character_emotions": {c["name"]: c["emotion"] for c in self.characters_detected},
            "notable_objects": self.notable_objects,
            "camera_style": self.camera_style,
            "vlm_conflict_detected": self.vlm_conflict_detected,
            "vlm_conflict_details": self.conflict_details,
            "distilled_keywords": self.distilled_keywords,
            "vlm_analyzed_at": self.analyzed_at,
        }


# ── Frame Extraction ──────────────────────────────────────────────────────────

def extract_scene_frames(
    video_path: str,
    scene_start: float,
    scene_end: float,
    num_frames: int = 8,
    output_dir: Optional[str] = None,
) -> List[str]:
    """
    Extract N evenly-spaced frames from a video scene segment.

    Args:
        video_path: Path to video file.
        scene_start: Scene start time in seconds.
        scene_end: Scene end time in seconds.
        num_frames: Number of frames to extract.
        output_dir: Directory to save frames. Uses temp dir if None.

    Returns:
        List of paths to extracted frame images.
    """
    import subprocess

    if output_dir is None:
        output_dir = "/tmp/movierag_frames"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = scene_end - scene_start
    if duration <= 0:
        duration = 1.0

    frame_paths: List[str] = []
    for i in range(num_frames):
        # Calculate timestamp: evenly spaced within the scene
        timestamp = scene_start + (i / max(num_frames - 1, 1)) * duration
        output_path = out_dir / f"scene_frame_{i:02d}.jpg"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            "-s", "896x504",  # Reduced resolution for VLM efficiency
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if output_path.exists():
                frame_paths.append(str(output_path))
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to extract frame at {timestamp}s")
            continue

    logger.debug(f"Extracted {len(frame_paths)} frames from [{scene_start:.1f}, {scene_end:.1f}]")
    return frame_paths


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── VLM Scene Analyzer ─────────────────────────────────────────────────────────

class VLMSceneAnalyzer:
    """
    Vision Language Model (VLM) guided scene understanding.

    Supports:
      - Gemini (primary)
      - Groq Vision (fallback)
      - Local VLM via OpenAI-compatible API

    Usage:
        analyzer = VLMSceneAnalyzer()
        result = analyzer.analyze_scene(
            video_path="movie.mp4",
            scene_start=120.0,
            scene_end=150.0,
            scene_id="tt001_scene_001",
            movie_id="tt001",
        )
    """

    # System prompt for VLM scene analysis
    SCENE_ANALYSIS_PROMPT = """You are an expert film analyst watching a movie scene.
Analyze the provided frames carefully and provide structured information.

Respond EXACTLY as this JSON format (no extra text):
{
  "description": "Detailed description of what happens in this scene...",
  "situation": "Single-word situation label: arguing|romantic|chase|comedic|tragic|suspenseful|calm|mysterious|dramatic",
  "setting": "Where and when: e.g. 'beach at sunset', 'dark office at night'",
  "actions": ["action1", "action2", ...],
  "emotional_tone": "e.g. 'tense', 'romantic', 'comedic', 'melancholic'",
  "characters": [
    {"name": "CharacterName", "emotion": "emotion_label", "position": "left|center|right"}
  ],
  "notable_objects": ["object1", "object2"],
  "camera_style": "e.g. 'close-up', 'wide shot', 'over-the-shoulder'"
}
"""

    # Prompt for conflict detection
    CONFLICT_CHECK_PROMPT = """Compare the VLM scene analysis with the expected movie metadata.

VLM Description: {vlm_desc}

Expected Movie: {movie_name}
Expected Characters: {expected_chars}

Question: Does the VLM description match the expected movie? Answer YES or NO and briefly explain.
"""

    # Prompt for query distillation
    DISTILL_PROMPT = """From the following VLM scene description, extract 5-10 search keywords (in English).
These keywords should capture: characters, actions, setting, objects, emotions.

Description: {description}

Output ONLY a comma-separated list of keywords, nothing else.
"""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_frames: int = VLM_MAX_FRAMES,
        max_tokens: int = VLM_MAX_TOKENS,
        temperature: float = VLM_TEMPERATURE,
        batch_size: int = VLM_BATCH_SIZE,
        conflict_threshold: float = CONFLICT_THRESHOLD,
        use_groq_fallback: bool = True,
    ):
        """
        Args:
            llm_client: UniversalLLMClient instance. Created internally if None.
            max_frames: Maximum frames to analyze per scene.
            max_tokens: Max tokens for VLM response.
            temperature: VLM sampling temperature.
            batch_size: Number of frames per VLM batch call.
            conflict_threshold: Score threshold for conflict detection.
            use_groq_fallback: Use Groq as fallback if primary fails.
        """
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.conflict_threshold = conflict_threshold
        self.use_groq_fallback = use_groq_fallback
        self._llm_client = llm_client

    # ── LLM Client ─────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._llm_client is None:
            from movierag.generation.universal_client import UniversalLLMClient
            self._llm_client = UniversalLLMClient()
        return self._llm_client

    # ── Main Analysis Entry Point ──────────────────────────────────────────────

    def analyze_scene(
        self,
        video_path: Optional[str] = None,
        scene_start: float = 0.0,
        scene_end: float = 30.0,
        scene_id: str = "unknown",
        movie_id: str = "unknown",
        frame_paths: Optional[List[str]] = None,
        expected_movie: Optional[str] = None,
        expected_characters: Optional[List[str]] = None,
    ) -> VLMAnalysis:
        """
        Analyze a video scene using VLM.

        Provide either `video_path` + `scene_start/end` OR `frame_paths`.

        Args:
            video_path: Path to video file.
            scene_start: Scene start time in seconds.
            scene_end: Scene end time in seconds.
            scene_id: Unique scene identifier.
            movie_id: Movie identifier.
            frame_paths: Pre-extracted frame paths (skip extraction if provided).
            expected_movie: Movie name for conflict detection.
            expected_characters: List of expected character names.

        Returns:
            VLMAnalysis dataclass with full scene understanding.
        """
        # Extract frames if not provided
        if frame_paths is None:
            if video_path is None:
                return VLMAnalysis(
                    scene_id=scene_id,
                    movie_id=movie_id,
                    vlm_description="",
                    error="Neither video_path nor frame_paths provided.",
                )
            frame_paths = extract_scene_frames(
                video_path, scene_start, scene_end, self.max_frames
            )

        if not frame_paths:
            return VLMAnalysis(
                scene_id=scene_id,
                movie_id=movie_id,
                vlm_description="",
                error="No frames extracted.",
            )

        # Run VLM analysis
        analysis_text, error = self._run_vlm_analysis(frame_paths)

        if error:
            return VLMAnalysis(
                scene_id=scene_id,
                movie_id=movie_id,
                vlm_description="",
                error=error,
            )

        # Parse structured output
        parsed = self._parse_vlm_response(analysis_text)

        # Conflict detection
        conflict_detected = False
        conflict_details = None
        if expected_movie and parsed.get("description"):
            conflict_detected, conflict_details = self._detect_conflict(
                parsed["description"],
                expected_movie,
                expected_characters or [],
            )

        # Query distillation
        distilled_keywords = []
        if parsed.get("description"):
            distilled_keywords = self._distill_keywords(parsed["description"])

        return VLMAnalysis(
            scene_id=scene_id,
            movie_id=movie_id,
            vlm_description=parsed.get("description", ""),
            situation=parsed.get("situation", ""),
            vision_setting=parsed.get("setting", ""),
            vision_actions=parsed.get("actions", []),
            emotional_tone=parsed.get("emotional_tone", ""),
            characters_detected=parsed.get("characters", []),
            notable_objects=parsed.get("notable_objects", []),
            camera_style=parsed.get("camera_style", ""),
            vlm_conflict_detected=conflict_detected,
            conflict_details=conflict_details,
            distilled_keywords=distilled_keywords,
            frame_count=len(frame_paths),
            analyzed_at=datetime.utcnow().isoformat(),
            raw_json=parsed,
        )

    # ── VLM Inference ──────────────────────────────────────────────────────────

    def _run_vlm_analysis(
        self,
        frame_paths: List[str],
    ) -> Tuple[str, Optional[str]]:
        """
        Send frames to VLM and get analysis.

        Strategy:
          - ≤5 frames: single call with all frames
          - >5 frames: batch into chunks, analyze each, then fuse
        """
        client = self._get_client()

        if len(frame_paths) <= 5:
            # Single call
            return self._single_vlm_call(frame_paths)
        else:
            # Batch: split into chunks
            chunks = [
                frame_paths[i:i + self.batch_size]
                for i in range(0, len(frame_paths), self.batch_size)
            ]
            analyses = []
            for chunk in chunks:
                text, err = self._single_vlm_call(chunk)
                if err:
                    return "", err
                analyses.append(text)

            # Fuse analyses
            return self._fuse_analyses(analyses), None

    def _single_vlm_call(
        self,
        frame_paths: List[str],
    ) -> Tuple[str, Optional[str]]:
        """Make a single VLM call with 1-N frames."""
        client = self._get_client()

        try:
            if len(frame_paths) == 1:
                result = client.generate_vision_content(
                    prompt=self.SCENE_ANALYSIS_PROMPT,
                    image_path=frame_paths[0],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            else:
                # Multi-frame: encode all frames as base64
                images_b64 = [encode_image_to_base64(p) for p in frame_paths]
                result = client.generate_multi_vision(
                    prompt=self.SCENE_ANALYSIS_PROMPT,
                    images_base64=images_b64,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

            if not result or result == "[No VLM available]":
                return "", "VLM returned no content"
            return result.strip(), None

        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return "", str(e)

    def _fuse_analyses(self, analyses: List[str]) -> str:
        """
        Fuse multiple VLM analyses (from frame batches) into a single coherent description.

        Uses LLM to synthesize the analyses.
        """
        if len(analyses) == 1:
            return analyses[0]

        client = self._get_client()
        fusion_prompt = (
            "You have analyzed different parts of the same movie scene. "
            "Synthesize all analyses into a single coherent scene description.\n\n"
            + "\n---\n".join(f"[Part {i + 1}]:\n{a}" for i, a in enumerate(analyses))
            + "\n\nOutput a single JSON object following the same format."
        )

        try:
            fused = client.generate_content(
                model=None,
                contents=[{"role": "user", "content": fusion_prompt}],
                max_tokens=self.max_tokens,
                temperature=0.1,
            )
            return fused if fused else analyses[0]
        except Exception as e:
            logger.warning(f"Analysis fusion failed: {e}. Using first analysis.")
            return analyses[0]

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_vlm_response(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse VLM raw text response into structured dict.

        Handles:
          - JSON with/without code fences
          - Partial JSON (best effort)
          - Plain text fallback
        """
        import re

        # Try JSON extraction
        json_match = re.search(r'\{[^{}]*(?:\[[^\]]*\][^{}]*)*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # Handle potential truncated JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common truncation issues
                if not json_str.rstrip().endswith("}"):
                    json_str = json_str.rsplit("}", 1)[0] + "}"
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Fallback: return raw text as description
        logger.warning("VLM response was not valid JSON. Using raw text.")
        return {
            "description": raw_text.strip(),
            "situation": "",
            "setting": "",
            "actions": [],
            "emotional_tone": "",
            "characters": [],
            "notable_objects": [],
            "camera_style": "",
        }

    # ── Conflict Detection ──────────────────────────────────────────────────────

    def _detect_conflict(
        self,
        vlm_description: str,
        expected_movie: str,
        expected_characters: List[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect conflicts between VLM description and expected movie metadata.

        Returns (conflict_detected, conflict_explanation).
        """
        client = self._get_client()

        chars_str = ", ".join(expected_characters) if expected_characters else "unknown"
        prompt = self.CONFLICT_CHECK_PROMPT.format(
            vlm_desc=vlm_description[:500],
            movie_name=expected_movie,
            expected_chars=chars_str,
        )

        try:
            response = client.generate_content(
                model=None,
                contents=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1,
            )
            response_lower = response.lower().strip()
            is_conflict = response_lower.startswith("no")
            details = response.strip() if response else None
            return is_conflict, details
        except Exception as e:
            logger.warning(f"Conflict detection failed: {e}")
            return False, None

    # ── Query Distillation ─────────────────────────────────────────────────────

    def _distill_keywords(self, description: str) -> List[str]:
        """
        Extract search keywords from VLM description for query expansion.
        """
        client = self._get_client()
        prompt = self.DISTILL_PROMPT.format(description=description[:1000])

        try:
            response = client.generate_content(
                model=None,
                contents=[{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=0.1,
            )
            if response:
                keywords = [k.strip().lower() for k in response.split(",") if k.strip()]
                return keywords[:10]
        except Exception as e:
            logger.warning(f"Query distillation failed: {e}")

        return []

    # ── Batch Analysis ─────────────────────────────────────────────────────────

    def analyze_scenes_batch(
        self,
        scenes: List[Dict[str, Any]],
        video_path: Optional[str] = None,
    ) -> List[VLMAnalysis]:
        """
        Analyze multiple scenes in a batch.

        Args:
            scenes: List of scene dicts, each with keys:
              - scene_id, movie_id, start_seconds, end_seconds
            video_path: Path to video (optional; needed for frame extraction).

        Returns:
            List of VLMAnalysis results.
        """
        results = []
        for scene in scenes:
            scene_start = scene.get("start_seconds", 0.0)
            scene_end = scene.get("end_seconds", 30.0)

            result = self.analyze_scene(
                video_path=video_path,
                scene_start=scene_start,
                scene_end=scene_end,
                scene_id=scene.get("scene_id", "unknown"),
                movie_id=scene.get("movie_id", "unknown"),
                expected_movie=scene.get("movie_title"),
                expected_characters=scene.get("characters"),
            )
            results.append(result)

        return results

    def __repr__(self) -> str:
        return (
            f"VLMSceneAnalyzer(frames={self.max_frames}, batch={self.batch_size}, "
            f"model={VLM_MODEL})"
        )
