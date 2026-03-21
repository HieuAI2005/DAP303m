# ─────────────────────────────────────────────────────────────────────────────
# script_aligner.py
# Script-Scene Alignment — Video Understanding Pipeline
# Layer 5: Script & Narrative — screenplay_context + script_heading
# ─────────────────────────────────────────────────────────────────────────────
"""
 Script-Scene Alignment: aligning screenplay/script text with video scenes.

 Key capabilities:
   1. Parse screenplay format (Fountain, PDF, plain text)
   2. Match script headings (INT./EXT.) to video scenes
   3. Align dialogue lines with Whisper transcripts
   4. Enrich Layer 5: Script & Narrative metadata
   5. Build script_heading → scene_id mapping

 Input formats:
   - Plain text screenplay (.txt)
   - PDF scripts
   - JSON structured script data

 Output schema (aligned script-scene):
   {
     "script_heading": "INT. TITANIC - DECK - DAY",
     "scene_id": str,
     "movie_id": str,
     "start_seconds": float,
     "end_seconds": float,
     "characters_in_scene": [str],
     "dialogue_lines": [
       {"speaker": str, "text": str, "start_seconds": float, "end_seconds": float}
     ],
     "screenplay_context": str,   # Preceding action description
     "narrative_arc": str,        # "exposition", "rising_action", "climax", "falling_action", "resolution"
     "confidence": float,
   }
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Screenplay Parsing ─────────────────────────────────────────────────────────

# Regex patterns for screenplay format
SCENE_HEADING_PATTERN = re.compile(
    r'^(INT\.|EXT\.|INT/EXT\.|I/E\.)\s+'
    r'(.+?)\s*[-–—]\s*(.+?)(?:\s*-\s*(.+))?$',
    re.IGNORECASE | re.MULTILINE
)

DIALOGUE_PATTERN = re.compile(
    r'^([A-Z][A-Z\s]+?)$\n^(.+)$',
    re.MULTILINE
)

ACTION_PATTERN = re.compile(
    r'^\s*[\[（\(]?(.*?)[\]）)]\s*$',
    re.MULTILINE
)


@dataclass
class ScriptScene:
    """A parsed scene from a screenplay."""
    heading: str                    # "INT. TITANIC - DECK - DAY"
    location: str                   # "TITANIC"
    setting: str                    # "DECK"
    time_of_day: str                # "DAY"
    scene_type: str                 # "INT" or "EXT"
    sequence: str = ""              # Optional sequence name
    raw_text: str = ""              # Full scene text
    action_description: str = ""    # Action/description lines
    dialogue_lines: List[Dict[str, str]] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)
    order_index: int = 0


@dataclass
class AlignedScriptScene:
    """A script scene aligned to a video scene."""
    script_scene: ScriptScene
    scene_id: str
    movie_id: str
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    start_score: float = 0.0
    alignment_method: str = ""     # "text_matched", "duration_avg", "dialogue_matched"
    narrative_arc: str = ""         # "exposition", "rising_action", "climax", "falling_action", "resolution"
    confidence: float = 0.0

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "script_heading": self.script_scene.heading,
            "screenplay_context": self.script_scene.action_description,
            "narrative_arc": self.narrative_arc,
            "characters_in_scene": self.script_scene.characters,
            "dialogue_lines": self.script_scene.dialogue_lines,
        }


# ── Screenplay Parser ─────────────────────────────────────────────────────────

class ScreenplayParser:
    """
    Parse screenplay/script text into structured scenes.

    Supports standard screenplay formats:
      - INT./EXT. scene headings
      - CHARACTER NAME (uppercase)
      - Dialogue lines
      - Action/description blocks

    Usage:
        parser = ScreenplayParser()
        scenes = parser.parse_file("titanic_script.txt")
    """

    def parse_file(self, path: str) -> List[ScriptScene]:
        """
        Parse a screenplay file into structured scenes.

        Args:
            path: Path to screenplay file (.txt, .fountain, .pdf).

        Returns:
            List of ScriptScene objects.
        """
        suffix = Path(path).suffix.lower()

        if suffix == ".pdf":
            text = self._extract_pdf_text(path)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

        return self.parse_text(text)

    def parse_text(self, text: str) -> List[ScriptScene]:
        """
        Parse screenplay text into structured scenes.

        Args:
            text: Raw screenplay text.

        Returns:
            List of ScriptScene objects.
        """
        scenes: List[ScriptScene] = []
        lines = text.split("\n")

        current_scene: Optional[ScriptScene] = None
        current_action: List[str] = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line:
                continue

            # Check for scene heading
            heading_match = SCENE_HEADING_PATTERN.match(line)
            if heading_match:
                # Save previous scene
                if current_scene:
                    self._finalize_scene(current_scene, current_action)
                    scenes.append(current_scene)

                # Start new scene
                scene_type = heading_match.group(1).upper().replace("/", "").replace(".", "")
                location = heading_match.group(2).strip()
                setting = heading_match.group(3).strip()
                time_of_day = heading_match.group(4).strip() if heading_match.group(4) else "DAY"

                current_scene = ScriptScene(
                    heading=line.upper(),
                    location=location,
                    setting=setting,
                    time_of_day=time_of_day,
                    scene_type=scene_type,
                    order_index=len(scenes),
                )
                current_action = []
                continue

            # Inside a scene
            if current_scene:
                # Check for dialogue
                dialogue_match = DIALOGUE_PATTERN.match(f"{line}\n{lines[i] if i < len(lines) else ''}")
                if dialogue_match and self._is_character_name(dialogue_match.group(1)):
                    speaker = dialogue_match.group(1).strip()
                    dialogue_text = dialogue_match.group(2).strip()
                    current_scene.dialogue_lines.append({
                        "speaker": speaker,
                        "text": dialogue_text,
                    })
                    if speaker not in current_scene.characters:
                        current_scene.characters.append(speaker)
                    i += 1
                    continue

                # Otherwise, action/description
                current_action.append(line)

        # Save last scene
        if current_scene:
            self._finalize_scene(current_scene, current_action)
            scenes.append(current_scene)

        logger.info(f"Parsed {len(scenes)} scenes from screenplay")
        return scenes

    def _is_character_name(self, name: str) -> bool:
        """Check if a string looks like a character name."""
        name = name.strip()
        if not name:
            return False
        # All uppercase, 2-30 chars, letters and spaces only
        return (
            name.isupper()
            and 2 <= len(name) <= 30
            and name.replace(" ", "").replace(".", "").isalpha()
            and not any(kw in name for kw in ("INT", "EXT", "CUT", "FADE", "DISSOLVE"))
        )

    def _finalize_scene(self, scene: ScriptScene, action_lines: List[str]):
        """Finalize scene metadata after parsing."""
        scene.action_description = " ".join(action_lines)

        # Extract characters from dialogue
        for dl in scene.dialogue_lines:
            speaker = dl["speaker"]
            if speaker not in scene.characters:
                scene.characters.append(speaker)

    def _extract_pdf_text(self, path: str) -> str:
        """Extract text from PDF screenplay."""
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text
        except ImportError:
            raise ImportError("pypdf required for PDF parsing. Install: pip install pypdf")
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {e}")


# ── Script-Scene Aligner ───────────────────────────────────────────────────────

class ScriptSceneAligner:
    """
    Align parsed screenplay scenes to video scenes using multiple signals.

    Alignment strategies:
      1. Text similarity: Match script headings to scene descriptions (CLIP/text)
      2. Duration matching: Align by estimated scene duration
      3. Dialogue matching: Match dialogue with Whisper transcripts
      4. Character matching: Ensure character consistency

    Usage:
        aligner = ScriptSceneAligner()
        aligned = aligner.align(
            script_scenes=script_scenes,
            video_scenes=video_scenes,
            movie_id="tt0120338",
        )
    """

    def __init__(
        self,
        clip_encoder: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        dialogue_matcher: Optional[Any] = None,
    ):
        self.clip_encoder = clip_encoder
        self._llm_client = llm_client
        self.dialogue_matcher = dialogue_matcher

    # ── Client ────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._llm_client is None:
            from movierag.generation.universal_client import UniversalLLMClient
            self._llm_client = UniversalLLMClient()
        return self._llm_client

    # ── Main Alignment ────────────────────────────────────────────────────────

    def align(
        self,
        script_scenes: List[ScriptScene],
        video_scenes: List[Dict[str, Any]],
        movie_id: str,
        whisper_transcript: Optional[Dict[str, Any]] = None,
    ) -> List[AlignedScriptScene]:
        """
        Align screenplay scenes to video scenes.

        Args:
            script_scenes: Parsed ScriptScene list from ScreenplayParser.
            video_scenes: List of video scene dicts with keys:
                scene_id, start_seconds, end_seconds, description, characters
            movie_id: Movie identifier.
            whisper_transcript: Whisper transcript result dict (optional).

        Returns:
            List of AlignedScriptScene objects.
        """
        if not script_scenes or not video_scenes:
            return []

        # Strategy 1: Text similarity matching
        text_aligned = self._align_by_text_similarity(script_scenes, video_scenes)

        # Strategy 2: Duration-based filling
        aligned = self._fill_duration_gaps(text_aligned, video_scenes, len(script_scenes))

        # Strategy 3: Dialogue matching (if transcript available)
        if whisper_transcript:
            aligned = self._refine_with_dialogue(aligned, whisper_transcript)

        # Strategy 4: Narrative arc assignment
        aligned = self._assign_narrative_arcs(aligned)

        return aligned

    def _align_by_text_similarity(
        self,
        script_scenes: List[ScriptScene],
        video_scenes: List[Dict[str, Any]],
    ) -> List[Optional[AlignedScriptScene]]:
        """
        Align scenes using text similarity between script and video descriptions.

        Uses LLM for fuzzy matching when CLIP is not available.
        """
        if self.clip_encoder:
            return self._align_with_clip(script_scenes, video_scenes)
        else:
            return self._align_with_llm(script_scenes, video_scenes)

    def _align_with_clip(
        self,
        script_scenes: List[ScriptScene],
        video_scenes: List[Dict[str, Any]],
    ) -> List[Optional[AlignedScriptScene]]:
        """Use CLIP for embedding-based scene matching."""
        # Build text embeddings for script scenes
        script_texts = [self._script_scene_to_text(s) for s in script_scenes]
        script_embeddings = self.clip_encoder.encode_text(script_texts)

        # Build text embeddings for video scenes
        video_texts = [self._video_scene_to_text(v) for v in video_scenes]
        video_embeddings = self.clip_encoder.encode_text(video_texts)

        # Greedy matching: assign each script scene to best video scene
        aligned: List[Optional[AlignedScriptScene]] = [None] * len(script_scenes)
        used_video = set()

        for i, script_emb in enumerate(script_embeddings):
            best_score = -1.0
            best_video_idx = None

            for j, video_emb in enumerate(video_embeddings):
                if j in used_video:
                    continue
                score = float(script_emb @ video_emb.T)
                if score > best_score:
                    best_score = score
                    best_video_idx = j

            if best_video_idx is not None:
                vs = video_scenes[best_video_idx]
                aligned[i] = AlignedScriptScene(
                    script_scene=script_scenes[i],
                    scene_id=vs["scene_id"],
                    movie_id=vs.get("movie_id", "unknown"),
                    start_seconds=vs.get("start_seconds", 0.0),
                    end_seconds=vs.get("end_seconds", 0.0),
                    start_score=best_score,
                    alignment_method="text_matched",
                    confidence=best_score,
                )
                used_video.add(best_video_idx)

        return aligned

    def _align_with_llm(
        self,
        script_scenes: List[ScriptScene],
        video_scenes: List[Dict[str, Any]],
    ) -> List[Optional[AlignedScriptScene]]:
        """Use LLM for fuzzy script-video scene matching."""
        client = self._get_client()

        aligned: List[Optional[AlignedScriptScene]] = [None] * len(script_scenes)

        for i, script_scene in enumerate(script_scenes):
            script_desc = self._script_scene_to_text(script_scene)

            # Build prompt for LLM matching
            scene_list = "\n".join(
                f"{j}. [{v.get('start_seconds', 0):.0f}s-{v.get('end_seconds', 0):.0f}s] "
                f"{v.get('description', '')[:100]}"
                for j, v in enumerate(video_scenes)
            )

            prompt = f"""Match this screenplay scene to one of the video scenes below.

Screenplay: "{script_desc}"

Video scenes:
{scene_list}

Output ONLY the number (0-{len(video_scenes) - 1}) of the best match, or -1 if no good match.
"""

            try:
                response = client.generate_content(
                    model=None,
                    contents=[{"role": "user", "content": prompt}],
                    max_tokens=16,
                    temperature=0.0,
                )
                match = int(response.strip())
                if 0 <= match < len(video_scenes):
                    vs = video_scenes[match]
                    aligned[i] = AlignedScriptScene(
                        script_scene=script_scene,
                        scene_id=vs["scene_id"],
                        movie_id=vs.get("movie_id", "unknown"),
                        start_seconds=vs.get("start_seconds", 0.0),
                        end_seconds=vs.get("end_seconds", 0.0),
                        start_score=0.8,
                        alignment_method="llm_matched",
                        confidence=0.8,
                    )
            except Exception as e:
                logger.warning(f"LLM alignment failed for scene {i}: {e}")

        return aligned

    def _fill_duration_gaps(
        self,
        aligned: List[Optional[AlignedScriptScene]],
        video_scenes: List[Dict[str, Any]],
        num_script_scenes: int,
    ) -> List[AlignedScriptScene]:
        """
        Fill in unaligned scenes using duration-based estimation.

        When a script scene has no video match, estimate its position
        based on relative scene duration.
        """
        # Collect aligned scenes
        aligned_scenes = [(i, a) for i, a in enumerate(aligned) if a is not None]

        if not aligned_scenes:
            # No alignment at all — estimate evenly
            total_duration = (
                max((v.get("end_seconds", 0) for v in video_scenes), default=0)
                - min((v.get("start_seconds", 0) for v in video_scenes), default=0)
            )
            avg_duration = total_duration / max(num_script_scenes, 1)

            for i in range(num_script_scenes):
                start = i * avg_duration
                aligned[i] = AlignedScriptScene(
                    script_scene=ScriptScene(
                        heading=f"SCENE_{i}",
                        location="",
                        setting="",
                        time_of_day="",
                        scene_type="UNKNOWN",
                    ),
                    scene_id="unknown",
                    movie_id="",
                    start_seconds=start,
                    end_seconds=start + avg_duration,
                    start_score=0.0,
                    alignment_method="duration_avg",
                    confidence=0.3,
                )
            return aligned

        # Fill gaps between aligned scenes
        for i in range(num_script_scenes):
            if aligned[i] is not None:
                continue

            # Find nearest aligned neighbors
            prev_idx = None
            next_idx = None
            for j, (idx, _) in enumerate(aligned_scenes):
                if idx < i:
                    prev_idx = j
                if idx > i and next_idx is None:
                    next_idx = j

            if prev_idx is not None and next_idx is not None:
                prev = aligned_scenes[prev_idx][1]
                next_ = aligned_scenes[next_idx][1]
                gap = next_.start_seconds - prev.end_seconds
                gap_count = next_.scene_id.count("_")  # rough estimate
                offset = (i - aligned_scenes[prev_idx][0]) / max(gap_count, 1)
                start = prev.end_seconds + offset * gap
                end = start + (prev.end_seconds - prev.start_seconds)
            elif prev_idx is not None:
                prev = aligned_scenes[prev_idx][1]
                start = prev.end_seconds
                end = start + (prev.end_seconds - prev.start_seconds)
            elif next_idx is not None:
                next_ = aligned_scenes[next_idx][1]
                end = next_.start_seconds
                start = end - (next_.end_seconds - next_.start_seconds)
            else:
                start, end = 0.0, 30.0

            aligned[i] = AlignedScriptScene(
                script_scene=ScriptScene(
                    heading=f"SCENE_{i}",
                    location="",
                    setting="",
                    time_of_day="",
                    scene_type="UNKNOWN",
                ),
                scene_id="unknown",
                movie_id="",
                start_seconds=start,
                end_seconds=end,
                start_score=0.0,
                alignment_method="duration_filled",
                confidence=0.4,
            )

        return aligned

    def _refine_with_dialogue(
        self,
        aligned: List[AlignedScriptScene],
        whisper_transcript: Dict[str, Any],
    ) -> List[AlignedScriptScene]:
        """
        Refine alignment using dialogue matching between script and Whisper.

        If a script scene's dialogue matches Whisper transcript,
        snap the alignment to the matched segment.
        """
        chunks = whisper_transcript.get("chunks", [])
        if not chunks:
            return aligned

        for a in aligned:
            if a is None:
                continue
            script_dialogue = " ".join(d["text"] for d in a.script_scene.dialogue_lines)
            if not script_dialogue:
                continue

            # Find best-matching Whisper chunk
            best_match = None
            best_score = 0.0
            for chunk in chunks:
                # Simple word overlap
                overlap = len(set(script_dialogue.lower().split()) & set(chunk["text"].lower().split()))
                score = overlap / max(len(set(script_dialogue.lower().split())), 1)
                if score > best_score:
                    best_score = score
                    best_match = chunk

            if best_match and best_score > 0.3:
                a.start_seconds = best_match["start_seconds"]
                a.end_seconds = best_match["end_seconds"]
                a.alignment_method = "dialogue_matched"
                a.confidence = max(a.confidence, best_score)

        return aligned

    def _assign_narrative_arcs(
        self,
        aligned: List[AlignedScriptScene],
    ) -> List[AlignedScriptScene]:
        """
        Assign narrative arc labels based on scene position.

        Arc distribution (approximate):
          - 0-10%: exposition
          - 10-50%: rising_action
          - 50-60%: climax
          - 60-80%: falling_action
          - 80-100%: resolution
        """
        n = len(aligned)
        if n == 0:
            return aligned

        for i, a in enumerate(aligned):
            if a is None:
                continue
            ratio = i / n
            if ratio < 0.1:
                a.narrative_arc = "exposition"
            elif ratio < 0.5:
                a.narrative_arc = "rising_action"
            elif ratio < 0.6:
                a.narrative_arc = "climax"
            elif ratio < 0.8:
                a.narrative_arc = "falling_action"
            else:
                a.narrative_arc = "resolution"

        return aligned

    # ── Text Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _script_scene_to_text(scene: ScriptScene) -> str:
        chars = ", ".join(scene.characters[:5])
        return f"{scene.heading} | {scene.action_description} | Characters: {chars}"

    @staticmethod
    def _video_scene_to_text(scene: Dict[str, Any]) -> str:
        desc = scene.get("description", "") or scene.get("vlm_description", "")
        chars = scene.get("characters", [])
        chars_str = ", ".join(chars[:5]) if chars else ""
        return f"{desc} | Characters: {chars_str}"

    def __repr__(self) -> str:
        return f"ScriptSceneAligner(clip={'yes' if self.clip_encoder else 'no'})"
