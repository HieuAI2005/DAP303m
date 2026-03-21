"""
Semantic Scene Segmenter — LLM-Based Boundary Detection

Uses Subtitle Transcripts + Gemini Flash to group short
optical shots into meaningful "Semantic Scenes" (based on SceneRAG Level 3 logic).
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict
import sys
import tempfile

from preprocess_data.config import PreprocessConfig as Cfg
from preprocess_data.extractors._scene_gemini_client import SceneGeminiClient

if str(Cfg.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(Cfg.SRC_DIR))

logger = logging.getLogger(__name__)

# System prompt for high-reasoning boundary detection
SCENE_BOUNDARY_SYSTEM_PROMPT = """You are an expert cinematic analyst specializing in narrative and temporal analysis of films.
Your task is to identify semantic scene boundaries in movie transcripts by applying MACRO-TO-MICRO reasoning. 

A "semantic scene boundary" is a point in time where the STORY meaningfully changes due to:
- Location change (exterior/interior, different place, different country)
- Significant time jump (hours, days, or years later)
- Narrative focus shift (from character A to B, from plot thread X to Y)
- Tonal shift (tragic to comedic, action to dialogue-heavy romantic scene)

Do NOT mark boundaries for:
- Brief cuts between characters in the same conversation
- Short reaction shots mid-scene
- Flashforwards/cuts that return to the same scene instantly

--- GLOBAL NARRATIVE THINKING ---
You will be provided with the Global Context of the movie (Plot, Synopsis, Script snippets).
1. First, read the transcript chunk and map it to the Global Context. Where are we in the story? What Act or Sequence does this belong to?
2. Use this macro-level understanding to chunk the transcript appropriately. Scenes should be as long as necessary. Do not split a continuous conversational sequence just because it is long. 
3. Evaluate candidates: would an audience perceive this as a new scene starting?
4. Output the final list of boundary timestamps with dual-tier reasoning (Macro context + Micro reason)."""

# The main boundary detection prompt (uses str.replace, not str.format)
SCENE_BOUNDARY_PROMPT = """Below is the Global Narrative Context, a list of PHYSICAL SCENE ANCHORS (determined by visual analysis), and a chronological transcript segment.

Your task: Identify which of these PHYSICAL SCENE ANCHORS (or other timestamps in the transcript) correspond to a meaningful NEW NARRATIVE SCENE.

=== GLOBAL CONTEXT ===
{global_context}
======================

=== PHYSICAL SCENE ANCHORS (Tool-detected) ===
{physical_anchors}
==============================================

=== TRANSCRIPT START ===
{transcript}
=== TRANSCRIPT END ===

Now analyze macro-to-micro, then output EXACTLY this JSON and nothing else:
{
  "scenes": [
    {
      "timestamp_sec": 120.5,
      "sequence_context": "Act 1: Character A explains the mission to Character B.",
      "reason": "Narrative shift: Tone becomes serious, mission briefing begins."
    }
  ]
}

Note: If a physical anchor coincides with a narrative shift, prioritize that exact timestamp.
If no clear boundaries exist in this segment, return: {"scenes": []}
"""


class SemanticSceneSegmenter:
    """Group basic optical shots into semantic scenes using LLM + Subtitles."""

    def __init__(self):
        self._llm = None
        # State for tracking LLM reasoning
        self.boundary_reasons = {}
        self.aligned_reasons = {}
        # Heuristics
        self.min_scene_duration = 30.0
        self.max_scene_duration = 300.0
        # Script alignment state (populated by segment_movie)
        self._aligned_script_scenes = []
        self._script_aligner = None
        self._cast_alias_map: Dict[str, str] = {}
        self._cast_characters: List[str] = []

    def _get_llm(self):
        if self._llm is None:
            self._llm = SceneGeminiClient(
                api_keys=Cfg.get_scene_gemini_api_keys(),
                model=Cfg.SCENE_GEMINI_MODEL,
                max_calls_per_hour=Cfg.SCENE_GEMINI_MAX_CALLS_PER_HOUR,
                timeout_ms=Cfg.SCENE_GEMINI_TIMEOUT_MS,
            )
        return self._llm

    def process_movie(self, movie_id: str) -> bool:
        """Read annotation (shots) & SRT, segment via LLM, and overwrite annotation with semantic scenes."""
        ann_path = Cfg.get_annotation_dir() / f"{movie_id}.json"
        srt_path = Cfg.get_subtitle_dir() / f"{movie_id}.srt"

        # Try nested movie_id subfolder structure (e.g. data/movie_output/<movie_id>/annotation/)
        if not ann_path.exists() and Cfg.OUTPUT_DIR is not None:
            ann_path = Cfg.OUTPUT_DIR / movie_id / "annotation" / f"{movie_id}.json"
        if not srt_path.exists() and Cfg.OUTPUT_DIR is not None:
            srt_path = Cfg.OUTPUT_DIR / movie_id / "subtitle" / f"{movie_id}.srt"

        if not ann_path.exists():
            logger.error(f"  ❌ Annotation not found for {movie_id}")
            return False

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)
        except Exception as e:
            logger.error(f"  ❌ Failed to read annotation {movie_id}: {e}")
            return False

        raw_shots = ann_data.get("raw_shots", [])
        if not raw_shots:
            logger.warning(
                "  No raw optical shots in annotation. Ensure annotator is updated."
            )
            return False

        duration = float(ann_data.get("duration", 0.0))

        # 1.5 Fetch Global Narrative Context (Metadata + Script)
        # Ensure metadata and strings are actively crawled/refreshed
        import sys

        if str(Cfg.SRC_DIR) not in sys.path:
            sys.path.insert(0, str(Cfg.SRC_DIR))
        from preprocess_data.extractors.metadata_extractor import MetadataCrawler

        crawler = MetadataCrawler()
        crawler.crawl(movie_id, force=False)
        self._prime_cast_alias_map(movie_id)

        global_context = self._fetch_global_context(movie_id)

        # Segment!
        scenes = self.segment_movie(
            movie_id,
            raw_shots,
            ann_data.get("scene", []),
            srt_path,
            duration,
            global_context,
        )

        # Update and save
        ann_data["scene"] = scenes
        ann_data["story"] = scenes  # Add story for compatibility with user example
        ann_data["total_scenes"] = len(scenes)
        ann_data["semantic_segmentation"] = True

        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(ann_data, f, ensure_ascii=False, indent=2)

        mapping_path = self.export_script_scene_mapping(movie_id, scenes)
        if mapping_path:
            logger.info(f"  📎 Script scene mapping saved → {mapping_path.name}")

        return True

    def backfill_script_mapping(self, movie_id: str) -> List[Dict]:
        """
        Re-run deterministic screenplay enrichment after semantic segmentation exists.

        This does not call Gemini again. It only refreshes:
        - script_aligned cache
        - annotation scene screenplay fields
        - semantic->script mapping artifact
        - auto_graph screenplay fields if auto_graph already exists
        """
        ann_path = Cfg.get_annotation_dir() / f"{movie_id}.json"
        if not ann_path.exists():
            logger.warning(f"  No annotation found for screenplay backfill: {movie_id}")
            return []

        ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
        scenes = list(ann_data.get("scene", []))
        if not scenes:
            return []

        from preprocess_data.extractors.script_aligner import ScriptAligner

        script_aligner = ScriptAligner()
        aligned_script_scenes = script_aligner.align(movie_id, force=True)
        self._aligned_script_scenes = aligned_script_scenes
        self._script_aligner = script_aligner
        self._prime_cast_alias_map(movie_id)

        if aligned_script_scenes:
            scenes = self._enrich_scenes_with_screenplay(scenes, aligned_script_scenes)

        ann_data["scene"] = scenes
        ann_data["story"] = scenes
        ann_path.write_text(
            json.dumps(ann_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.export_script_scene_mapping(movie_id, scenes)
        self._update_auto_graph_script_fields(movie_id, scenes)
        return scenes

    def segment_movie(
        self,
        movie_id: str,
        shots: List[Dict],
        physical_scenes: List[Dict],
        srt_path: Path,
        duration: float,
        global_context: str = "",
    ) -> List[Dict]:
        """
        Group optical shots into semantic scenes with Physical Anchor guidance.
        """
        logger.info(
            f"  🎬 Segmenting with {len(shots)} shots and {len(physical_scenes)} physical anchors..."
        )

        # 1. Load Subtitles
        # Check encoding
        subs = []
        try:
            from movierag.indexing.dialogue_indexer import DialogueIndexer

            indexer = DialogueIndexer()  # For parsing SRT
            # srt_path might need special reading if it's not standard utf-8
            subs = self._load_subtitles(srt_path)
        except Exception as e:
            logger.warning(f"  Subtitle load failed: {e}")

        if not subs:
            logger.warning(
                "  No subtitles loaded. Falling back to physical scene grouping."
            )
            return physical_scenes

        # 2. Extract Physical Anchors as text
        anchors_txt = "\n".join(
            [
                f"- {s['start_seconds']:.1f}s: Physical cut/Visual shift"
                for s in physical_scenes
            ]
        )
        self.physical_anchors_txt = anchors_txt
        self._all_physical_scenes = physical_scenes  # Store for V3 filtering

        # 3. Batch LLM segmentation
        # (Rest of logic follows, but prompt is updated via SCENE_BOUNDARY_PROMPT)
        if not shots:
            return []

        logger.info(
            f"  🎬 Semantic Segmentation: {movie_id} ({len(shots)} optical shots)"
        )

        # State for reasoning
        self.boundary_reasons = {}
        self.aligned_reasons = {}

        # 1. Align screenplay to video timeline (cached, fast on second run)
        from preprocess_data.extractors.script_aligner import ScriptAligner

        script_aligner = ScriptAligner()
        self._aligned_script_scenes = script_aligner.align(movie_id)
        self._script_aligner = script_aligner

        if self._aligned_script_scenes:
            arc_summary = script_aligner.summarize_arc(self._aligned_script_scenes)
            global_context = f"{global_context}\n\n{arc_summary}".strip()
            logger.info(
                f"  📜 Script narrative arc injected ({len(self._aligned_script_scenes)} scenes aligned)."
            )

        # 2. Parse SRT
        subs = self._parse_srt(srt_path)

        # 3. Detect boundaries via LLM
        if not subs:
            logger.warning(
                "  No subtitles found! Using heuristic grouping without LLM."
            )
            boundaries = []
        else:
            boundaries = self._detect_boundaries_llm(subs, global_context)

        # 4. Refine and Align boundaries with Optical Shots
        shot_boundaries = [shots[0]["start_sec"]] + [s["end_sec"] for s in shots]
        refined_boundaries = self._align_and_refine_boundaries(
            boundaries, shot_boundaries, duration
        )

        # 5. Group shots
        scenes = self._group_shots(shots, refined_boundaries, duration)
        logger.info(
            f"  ✅ Semantic Segmentation: grouped {len(shots)} shots into {len(scenes)} scenes."
        )

        # 6. Enrich each scene with screenplay type classification
        if self._aligned_script_scenes:
            scenes = self._enrich_scenes_with_screenplay(
                scenes, self._aligned_script_scenes
            )
            logger.info(f"  🏷️  Scene type classification complete.")

        # 7. Add granular subtitles to each scene (story format)
        if subs:
            scenes = self._add_subtitles_to_scenes(scenes, subs)
            logger.info(f"  💬 Subtitles merged into story format.")

        return scenes

    def export_script_scene_mapping(self, movie_id: str, scenes: List[Dict]) -> Path | None:
        """Persist explicit semantic-scene -> screenplay-scene mapping for inspection."""
        if not scenes:
            return None

        out_path = (
            Cfg.get_semantic_script_mapping_dir()
            / f"{movie_id}_semantic_script_mapping.json"
        )
        payload = []
        for scene in scenes:
            payload.append(
                {
                    "scene_id": scene.get("id", ""),
                    "shot": scene.get("shot", []),
                    "frame": scene.get("frame", []),
                    "start_seconds": scene.get("start_seconds", 0.0),
                    "end_seconds": scene.get("end_seconds", 0.0),
                    "start_time": scene.get("start_time"),
                    "end_time": scene.get("end_time"),
                    "script_time_of_day": scene.get("script_time_of_day", ""),
                    "script_location": scene.get("script_location", ""),
                    "script_characters": scene.get("script_characters", []),
                    "script_headings": scene.get("script_headings", []),
                    "script_primary_heading": scene.get("script_primary_heading", ""),
                    "dominant_script_scene_ref": scene.get(
                        "dominant_script_scene_ref"
                    ),
                    "dominant_script_overlap_sec": scene.get(
                        "dominant_script_overlap_sec", 0.0
                    ),
                    "script_scene_refs": scene.get("script_scene_refs", []),
                }
            )

        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path

    def _load_subtitles(self, srt_path: Path) -> List[Dict]:
        """Wrapper for _parse_srt"""
        return self._parse_srt(srt_path)

    def _parse_srt(self, srt_path: Path) -> List[Dict]:
        if not srt_path or not srt_path.exists():
            return []

        # Try multiple encodings — real SRTs often have Windows-1252 smart quotes
        for enc in ("utf-8-sig", "cp1252", "utf-8"):
            try:
                text = srt_path.read_text(encoding=enc).strip()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            text = srt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []

        blocks = text.split("\n\n")
        subs = []
        for block in blocks:
            lines = block.split("\n")
            if len(lines) >= 3:
                time_line = lines[1]

                # Fix: In case text has newlines internally, safely rejoin
                content = " ".join([line.strip() for line in lines[2:] if line.strip()])

                if " --> " in time_line:
                    start_str, end_str = time_line.split(" --> ")
                    start_sec = self._parse_srt_time(start_str)
                    end_sec = self._parse_srt_time(end_str)
                    subs.append({"start": start_sec, "end": end_sec, "text": content})
        return subs

    def _parse_srt_time(self, time_str: str) -> float:
        """Convert HH:MM:SS,mmm to seconds"""
        time_str = time_str.strip().replace(",", ".")
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return 0.0

    def _resolve_meta_path(self, movie_id: str) -> Path | None:
        search_dirs = [Cfg.get_meta_dir(), *getattr(Cfg, "META_SEARCH_DIRS", [])]
        seen = set()

        for meta_dir in search_dirs:
            candidate_dir = Path(meta_dir)
            candidate_key = str(candidate_dir.resolve()) if candidate_dir.exists() else str(candidate_dir)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)

            candidate = candidate_dir / f"{movie_id}.json"
            if candidate.exists():
                return candidate

        return None

    @staticmethod
    def _clean_character_display(value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s*\(.*?\)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" -")

    @classmethod
    def _normalize_character_key(cls, value: str) -> str:
        text = cls._clean_character_display(value).upper()
        text = text.replace("CONT'D", " ")
        text = re.sub(r"[^A-Z0-9\s]", " ", text)
        text = re.sub(r"\bO S\b|\bV O\b|\bOS\b|\bVO\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _build_character_aliases(cls, character_name: str) -> List[str]:
        normalized = cls._normalize_character_key(character_name)
        if not normalized:
            return []
        tokens = [token for token in normalized.split() if token]
        aliases = {normalized}
        if tokens:
            aliases.add(tokens[0])
            aliases.add(tokens[-1])
            if len(tokens) >= 2:
                aliases.add(" ".join(tokens[:2]))
        return [alias for alias in aliases if len(alias) >= 2]

    def _prime_cast_alias_map(self, movie_id: str) -> None:
        alias_map: Dict[str, str] = {}
        cast_characters: List[str] = []
        meta_path = self._resolve_meta_path(movie_id)
        if meta_path and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            for cast_entry in meta.get("cast", []) or []:
                character_name = self._clean_character_display(
                    cast_entry.get("character", "")
                )
                if not character_name or character_name.lower() == "unknown":
                    continue
                if character_name not in cast_characters:
                    cast_characters.append(character_name)
                for alias in self._build_character_aliases(character_name):
                    alias_map.setdefault(alias, character_name)
        self._cast_alias_map = alias_map
        self._cast_characters = cast_characters

    def _canonicalize_script_characters(self, names: List[str]) -> List[str]:
        generic_tokens = {
            "CLASS",
            "FEMALE",
            "MALE",
            "MAN",
            "WOMAN",
            "BOY",
            "GIRL",
            "VOICE",
            "NARRATOR",
            "VISITOR",
            "STRANGER",
            "TEACHER",
            "SECRETARY",
            "MOTHER",
            "FATHER",
        }
        resolved: List[str] = []
        for raw_name in names or []:
            cleaned = self._clean_character_display(str(raw_name or ""))
            normalized = self._normalize_character_key(cleaned)
            if not normalized or normalized in {"UNKNOWN", "N A"}:
                continue
            candidate = self._cast_alias_map.get(normalized)
            if not candidate:
                tokens = [token for token in normalized.split() if token]
                if not tokens or all(token in generic_tokens for token in tokens):
                    continue
                if len(tokens) == 1 and len(tokens[0]) < 2:
                    continue
                candidate = cleaned.title() if cleaned.isupper() else cleaned
            if candidate and candidate not in resolved:
                resolved.append(candidate)
        return resolved

    def _fetch_global_context(self, movie_id: str) -> str:
        """Fetch metadata and script content to build a global narrative context string."""
        context = []

        # 1. Fetch JSON Metadata
        meta_path = self._resolve_meta_path(movie_id)
        if meta_path and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)

                title = meta_data.get("title", "")
                if title:
                    context.append(f"Title: {title}")

                storyline = meta_data.get("storyline", "")
                if storyline:
                    context.append(f"Storyline: {storyline}")

                plot = meta_data.get("plot", "")
                if plot:
                    context.append(f"Plot: {plot}")

            except Exception as e:
                logger.warning(f"Failed to read metadata for {movie_id}: {e}")

        # 2. Script Narrative Arc (via ScriptAligner — lightweight heading-only summary)
        # The full per-chunk screenplay context is injected later in _detect_boundaries_llm()
        script_path = Cfg.get_script_dir() / f"{movie_id}.script"
        if script_path.exists():
            context.append(
                "\n[Script is available and will be injected as local context per analysis chunk]"
            )

        return "\n\n".join(context) if context else "No Global Context Available."

    def _detect_boundaries_llm(
        self, subs: List[Dict], global_context: str = ""
    ) -> List[float]:
        llm = self._get_llm()
        refined_boundaries = []
        chunk_errors: List[str] = []

        # ── Screenplay-aware adaptive chunking ───────────────────────────────
        # Gemini token limits are intentionally left open, so chunking can carry
        # larger transcript/script windows without prematurely dropping context.
        CHUNK_SIZE = 1200.0  # target chunk size in seconds
        OVERLAP_SEC = 180.0  # overlap from previous chunk

        aligned_scenes = getattr(self, "_aligned_script_scenes", [])
        script_aligner = getattr(self, "_script_aligner", None)

        # Build candidate chunk break points: prefer screenplay scene starts
        script_break_times = sorted(
            {sc.start_sec for sc in aligned_scenes if sc.start_sec > 0}
        )

        # Build sub-lists of subs grouped into adaptive chunks
        chunks: List[List[Dict]] = []
        prev_chunk_tail: List[Dict] = []  # overlap subs from prev chunk
        chunk_start_time = subs[0]["start"] if subs else 0.0
        current_chunk: List[Dict] = []

        for sub in subs:
            current_chunk.append(sub)
            elapsed = sub["end"] - chunk_start_time

            if elapsed < CHUNK_SIZE:
                continue

            # Prefer to break at a screenplay scene transition near this point
            # Look for a screenplay break within ±90s of the natural cut point
            best_break = sub["end"]  # fallback: break right here
            cut_candidates = [
                t for t in script_break_times if sub["end"] - 90 <= t <= sub["end"] + 90
            ]
            if cut_candidates:
                # Choose screenplay break closest to where we naturally want to cut
                best_break = min(cut_candidates, key=lambda t: abs(t - sub["end"]))
                # Extend or trim the chunk to reach that screenplay break
                # (will be handled naturally on next iterations)
                if best_break > sub["end"] + 30:
                    # The screenplay break is still 30+ seconds away — keep going
                    continue

            # Close this chunk
            if current_chunk:
                chunks.append(prev_chunk_tail + current_chunk)
                # Save tail as overlap for next chunk (last OVERLAP_SEC of subs)
                prev_chunk_tail = [
                    s
                    for s in current_chunk
                    if s["start"] >= current_chunk[-1]["end"] - OVERLAP_SEC
                ]
                current_chunk = []
                chunk_start_time = best_break

        if current_chunk:
            chunks.append(prev_chunk_tail + current_chunk)

        if not chunks:
            chunks = [subs]  # fallback: single chunk

        for i, chunk in enumerate(chunks):
            chunk_start = chunk[0]["start"]
            chunk_end = chunk[-1]["end"]
            transcript_text = "\n".join(
                [f"[{sub['start']:.1f}] {sub['text']}" for sub in chunk]
            )

            # Inject local screenplay context for this specific time window
            local_script_ctx = ""
            if aligned_scenes and script_aligner:
                local_script_ctx = script_aligner.get_script_context(
                    aligned_scenes, chunk_start, chunk_end, max_chars=60000
                )

            # Filter physical anchors to only those in the current chunk (V3 optimization)
            filtered_anchors = self._get_anchors_for_chunk(
                chunk_start, chunk_end, overlap=30.0
            )

            # Build enriched global context (macro arc + local screenplay for this chunk)
            chunk_context = global_context
            if local_script_ctx:
                chunk_context = f"{global_context}\n\n{local_script_ctx}"

            prompt = SCENE_BOUNDARY_PROMPT.replace("{global_context}", chunk_context)
            prompt = prompt.replace(
                "{physical_anchors}",
                filtered_anchors or "None in this time window.",
            )
            prompt = prompt.replace("{transcript}", transcript_text)

            try:
                import re as _re
                import time as _time

                response_text = llm.generate_text(
                    prompt,
                    system_prompt=SCENE_BOUNDARY_SYSTEM_PROMPT,
                    thinking_level="HIGH",
                    temperature=0.4,
                    max_output_tokens=Cfg.SCENE_GEMINI_MAX_OUTPUT_TOKENS or None,
                )

                # DEBUG PROBE
                debug_log_path = (
                    Cfg.OUTPUT_DIR / "llm_debug_dump.txt"
                    if Cfg.OUTPUT_DIR
                    else Path(tempfile.gettempdir()) / "llm_debug_dump.txt"
                )
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- LLM RESPONSE ---\n{response_text}\n--- END ---\n")

                if not response_text or not response_text.strip():
                    logger.warning(
                        f"    Chunk {i + 1}/{len(chunks)}: LLM returned empty response. Skipping."
                    )
                    continue

                text = response_text

                # Extract JSON block more robustly
                # Note: (?R) is not supported by built-in 're'. Greedy brace matching is effective for LLM output.
                json_match = _re.search(r"(\{.*\})", text, _re.DOTALL)

                if json_match:
                    text = json_match.group(0)

                # Clean up common malformed JSON issues (trailing commas, etc.)
                text = _re.sub(r",\s*\}", "}", text)
                text = _re.sub(r",\s*\]", "]", text)

                try:
                    data = json.loads(text.strip())
                except json.JSONDecodeError as je:
                    logger.warning(
                        f"    Chunk {i + 1}/{len(chunks)}: JSON decode failed ({je}). Trying to repair..."
                    )
                    # Extreme repair: try to find the last valid closure
                    last_brace = text.rfind("}")
                    if last_brace != -1:
                        try:
                            data = json.loads(text[: last_brace + 1])
                        except:
                            raise je
                    else:
                        raise je

                scenes = data.get("scenes", [])

                b_sec = []
                for s in scenes:
                    b = s.get("timestamp_sec")
                    seq_ctx = s.get("sequence_context", "")
                    raw_reason = s.get("reason", "No reason provided")
                    reason = f"[{seq_ctx}] {raw_reason}" if seq_ctx else raw_reason
                    if b is not None:
                        val = float(b)
                        b_sec.append(val)
                        self.boundary_reasons[val] = reason

                refined_boundaries.extend(b_sec)
                script_note = "(+screenplay)" if local_script_ctx else ""
                logger.info(
                    f"    Chunk {i + 1}/{len(chunks)} {script_note}: Found {len(b_sec)} boundaries."
                )
            except Exception as e:
                chunk_errors.append(str(e))
                logger.error(
                    f"    Chunk {i + 1}/{len(chunks)} boundary detection failed: {e}"
                )

        if not refined_boundaries and chunk_errors:
            lowered_errors = " | ".join(chunk_errors).lower()
            hard_fail_markers = (
                "429",
                "resource_exhausted",
                "quota",
                "no active gemini scene keys are available",
                "all gemini scene keys are disabled",
            )
            if any(marker in lowered_errors for marker in hard_fail_markers):
                raise RuntimeError(f"scene_keys_unavailable: {chunk_errors[0]}")

        return sorted(list(set(refined_boundaries)))

    def _get_anchors_for_chunk(
        self, start: float, end: float, overlap: float = 30.0
    ) -> str:
        """Filter physical scenes to a specific time window (V3 optimization)."""
        if not hasattr(self, "_all_physical_scenes"):
            return ""

        filtered = []
        # Include anchors slightly outside the window for context
        window_start = max(0, start - overlap)
        window_end = end + overlap

        for s in self._all_physical_scenes:
            ts = s.get("start_seconds", 0.0)
            if window_start <= ts <= window_end:
                filtered.append(f"- {ts:.1f}s: Physical cut/Visual shift")

        if not filtered:
            return ""
        return "\n".join(filtered)

    def _enrich_scenes_with_screenplay(
        self, scenes: List[Dict], aligned_script_scenes: list
    ) -> List[Dict]:
        """
        Enrich each grouped semantic scene with screenplay-derived type tags.
        Adds: environment, time_of_day, scene_type, script_characters, script_location.
        """
        for scene in scenes:
            start_sec = scene.get("start_seconds", 0.0)
            end_sec = scene.get("end_seconds", start_sec + 30)

            overlapping = self._get_overlapping_script_scenes(
                aligned_script_scenes, start_sec, end_sec
            )

            scene_tags = self._classify_scene_from_script(overlapping)
            scene.update(scene_tags)
            script_refs = []
            for script_scene in overlapping:
                ref = self._script_scene_to_ref(script_scene)
                ref["characters"] = self._canonicalize_script_characters(
                    ref.get("characters", [])
                )
                script_refs.append(ref)
            scene["script_scene_refs"] = script_refs
            scene["script_scene_count"] = len(scene["script_scene_refs"])
            scene["script_headings"] = [
                ref["heading"] for ref in scene["script_scene_refs"]
            ]
            dominant = self._select_dominant_script_scene(
                script_refs, start_sec, end_sec
            )
            scene["dominant_script_scene_ref"] = dominant
            scene["dominant_script_overlap_sec"] = round(
                float(dominant.get("overlap_seconds", 0.0)) if dominant else 0.0, 2
            )
            scene["script_primary_heading"] = dominant.get("heading", "") if dominant else (
                scene["script_scene_refs"][0]["heading"] if scene["script_scene_refs"] else ""
            )
            scene["script_characters"] = self._canonicalize_script_characters(
                scene.get("script_characters", [])
            )

        return scenes

    @staticmethod
    def _script_scene_to_ref(script_scene) -> Dict:
        return {
            "script_scene_uid": getattr(script_scene, "scene_uid", ""),
            "scene_num": getattr(script_scene, "scene_num", None),
            "heading": getattr(script_scene, "heading", ""),
            "location": getattr(script_scene, "location", ""),
            "time_of_day": getattr(script_scene, "time_of_day", ""),
            "characters": list(getattr(script_scene, "characters", []) or []),
            "start_sec": round(float(getattr(script_scene, "start_sec", 0.0)), 2),
            "end_sec": round(float(getattr(script_scene, "end_sec", 0.0)), 2),
            "anchor_quality": getattr(script_scene, "anchor_quality", ""),
            "confidence_score": round(
                float(getattr(script_scene, "confidence_score", 0.0) or 0.0), 3
            ),
            "anchor_start_sec": getattr(script_scene, "anchor_start_sec", None),
            "anchor_end_sec": getattr(script_scene, "anchor_end_sec", None),
            "linear_start_sec": round(
                float(getattr(script_scene, "linear_start_sec", 0.0)), 2
            ),
            "linear_end_sec": round(
                float(getattr(script_scene, "linear_end_sec", 0.0)), 2
            ),
        }

    @staticmethod
    def _get_overlapping_script_scenes(
        aligned_script_scenes: list, start_sec: float, end_sec: float
    ) -> List:
        overlapping = [
            sc
            for sc in aligned_script_scenes
            if sc.end_sec > start_sec and sc.start_sec < end_sec
        ]
        if overlapping:
            return sorted(overlapping, key=lambda scene: (scene.start_sec, scene.end_sec))

        nearest = min(
            aligned_script_scenes,
            key=lambda s: abs(s.start_sec - start_sec),
            default=None,
        )
        return [nearest] if nearest else []

    @staticmethod
    def _select_dominant_script_scene(
        script_refs: List[Dict], start_sec: float, end_sec: float
    ) -> Dict | None:
        if not script_refs:
            return None

        quality_rank = {"full": 2, "partial": 1, "linear": 0}

        def sort_key(ref: Dict):
            overlap = max(
                0.0,
                min(end_sec, float(ref.get("end_sec", 0.0)))
                - max(start_sec, float(ref.get("start_sec", 0.0))),
            )
            return (
                overlap,
                quality_rank.get(str(ref.get("anchor_quality", "linear")), 0),
                -float(ref.get("start_sec", 0.0)),
            )

        dominant = dict(max(script_refs, key=sort_key))
        dominant["overlap_seconds"] = round(
            max(
                0.0,
                min(end_sec, float(dominant.get("end_sec", 0.0)))
                - max(start_sec, float(dominant.get("start_sec", 0.0))),
            ),
            2,
        )
        return dominant

    def _classify_scene_from_script(self, script_scenes: list) -> Dict:
        """
        Classify a semantic scene using the aligned screenplay ScriptScene objects.
        Returns a dict of tags to add to the scene dict.
        """
        if not script_scenes:
            return {
                "scene_type": "unknown",
                "environment": "unknown",
                "time_of_day": "unknown",
            }

        # Aggregate across all overlapping script scenes
        headings = [sc.heading for sc in script_scenes]
        heading_upper = " ".join(headings).upper()

        # Filter out screenplay stage directions that get parsed as character names
        def _is_real_character(name: str) -> bool:
            n = name.strip().upper()
            # Skip pure stage directions and parentheticals
            if any(
                sd in n
                for sd in {
                    "(MORE)",
                    "(BEAT)",
                    "(O.S.)",
                    "(V.O.)",
                    "(CONT'",
                    "END TITLE",
                    "TITLE CARD",
                    " CARD:",
                    "END TITL",
                }
            ):
                return False
            # Skip things that start with a digit (scene numbers, shot numbers)
            if n and n[0].isdigit():
                return False
            return True

        import re as _re_cls

        raw_chars = {
            c for sc in script_scenes for c in sc.characters if _is_real_character(c)
        }
        # Strip parenthetical suffixes for cleaner display: "HARVEY MILK (CONT'D)" -> "HARVEY MILK"
        all_chars = sorted(
            {_re_cls.sub(r"\s*\(.*?\)", "", c).strip() for c in raw_chars if c.strip()}
        )
        all_chars = self._canonicalize_script_characters(all_chars)

        # action_lines may be empty for cache-loaded scenes. Also search heading text.
        all_action = " ".join(" ".join(sc.action_lines) for sc in script_scenes).upper()
        all_action = all_action + " " + heading_upper  # Enrich with heading keywords
        total_dialogue = sum(len(sc.dialogue_lines) for sc in script_scenes)
        total_action = sum(len(sc.action_lines) for sc in script_scenes)
        locations = [sc.location for sc in script_scenes]
        times = [sc.time_of_day for sc in script_scenes]

        # ── Environment ─────────────────────────────────────────────────────
        if "INT." in heading_upper and "EXT." not in heading_upper:
            environment = "indoor"
        elif "EXT." in heading_upper and "INT." not in heading_upper:
            environment = "outdoor"
        else:
            environment = "mixed"

        # ── Time of day ──────────────────────────────────────────────────────
        tod_priority = ["NIGHT", "DAWN", "DUSK", "DAY", "MORNING", "EVENING"]
        combined_tod = " ".join(times).upper()
        time_of_day = "unknown"
        for t in tod_priority:
            if t in combined_tod:
                time_of_day = t.lower()
                break

        # ── Scene type ───────────────────────────────────────────────────────
        ACTION_KEYWORDS = {
            "GUN",
            "SHOOT",
            "FIRE",
            "FIGHT",
            "CHASE",
            "PUNCH",
            "HIT",
            "EXPLOSION",
            "RUNS",
            "CRASH",
            "STAB",
            "BLOOD",
            "BATTLE",
            "SLAM",
            "THROWS",
            "ATTACK",
            "SCREAMS",
            "ESCAPES",
        }
        ROMANCE_KEYWORDS = {
            "KISS",
            "EMBRACE",
            "HOLD",
            "LOVE",
            "BED",
            "TOUCH",
            "ROMANTIC",
            "SENSUAL",
            "CARESS",
            "INTIMATE",
            "TOGETHER",
        }
        COMEDY_KEYWORDS = {
            "LAUGH",
            "JOKE",
            "FUNNY",
            "COMIC",
            "CHUCKLE",
            "SMIRKS",
            "GRINS",
            "WINK",
            "CLOWN",
            "SILLY",
        }
        TENSION_KEYWORDS = {
            "TENSE",
            "NERVOUS",
            "SWEAT",
            "FEAR",
            "STARE",
            "HESITATE",
            "CONFRONTS",
            "GLARES",
            "THREATEN",
            "INTERROGAT",
            "ARGUE",
        }
        MONTAGE_KEYWORDS = {"MONTAGE", "SERIES", "QUICK SHOTS", "INTERCUT"}

        action_words = set(all_action.split())

        def hits(keywords):
            return bool(action_words & keywords)

        # Priority-based scene type detection
        if hits(MONTAGE_KEYWORDS) or "MONTAGE" in heading_upper:
            scene_type = "montage"
        elif hits(ACTION_KEYWORDS):
            scene_type = "action"
        elif hits(ROMANCE_KEYWORDS):
            scene_type = "romance"
        elif hits(TENSION_KEYWORDS):
            scene_type = "tension"
        elif hits(COMEDY_KEYWORDS):
            scene_type = "comedy"
        elif total_dialogue > total_action * 2:
            scene_type = "dialogue"
        elif total_action > total_dialogue * 2:
            scene_type = "action-visual"  # action with little dialogue
        else:
            scene_type = "mixed"

        # ── Character count classification ───────────────────────────────────
        n_chars = len(all_chars)
        if n_chars == 0:
            char_type = "no_characters"  # e.g. B-roll, action
        elif n_chars == 1:
            char_type = "solo"
        elif n_chars == 2:
            char_type = "duo"
        elif n_chars <= 4:
            char_type = "small_group"
        else:
            char_type = "ensemble"

        return {
            "environment": environment,
            "time_of_day": time_of_day,
            "script_time_of_day": time_of_day,
            "scene_type": scene_type,
            "character_type": char_type,
            "script_characters": all_chars[:8],  # Top 8 to avoid bloat
            "script_location": locations[0] if locations else "",
        }

    def _update_auto_graph_script_fields(self, movie_id: str, scenes: List[Dict]) -> None:
        graph_path = (
            Cfg.get_scene_graph_dir() / movie_id / f"{movie_id}_auto_graph.json"
        )
        if not graph_path.exists():
            return

        scene_map = {scene.get("id"): scene for scene in scenes if scene.get("id")}
        try:
            graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"  Failed to read auto graph for screenplay backfill: {exc}")
            return

        changed = False
        for clip in graph_data.get("clips", []):
            scene = scene_map.get(clip.get("scene_id"))
            if not scene:
                continue
            clip["scene_type"] = scene.get("scene_type", "")
            clip["environment"] = scene.get("environment", "")
            clip["script_time_of_day"] = scene.get("script_time_of_day", "")
            clip["character_type"] = scene.get("character_type", "")
            clip["script_location"] = scene.get("script_location", "")
            clip["script_characters"] = scene.get("script_characters", [])
            clip["script_scene_refs"] = scene.get("script_scene_refs", [])
            clip["script_scene_count"] = scene.get("script_scene_count", 0)
            clip["script_primary_heading"] = scene.get("script_primary_heading", "")
            clip["script_headings"] = scene.get("script_headings", [])
            clip["dominant_script_scene_ref"] = scene.get(
                "dominant_script_scene_ref"
            )
            clip["dominant_script_overlap_sec"] = scene.get(
                "dominant_script_overlap_sec", 0.0
            )
            changed = True

        if changed:
            graph_path.write_text(
                json.dumps(graph_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _call_high_reasoning_llm(self, llm, prompt: str) -> str:
        """Call gpt-oss-120b directly with reasoning_effort:high + system prompt for best boundary detection."""
        # Try the high-reasoning model first via direct Groq call
        if hasattr(llm, "_groq_client") and llm._groq_client:
            try:
                import time

                logger.info("  🧠 Calling gpt-oss-120b [reasoning_effort=high]...")
                completion = llm._groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": SCENE_BOUNDARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1.0,
                    max_completion_tokens=Cfg.SCENE_REASONING_MAX_COMPLETION_TOKENS or None,
                    top_p=1,
                    stream=True,
                    extra_body={"reasoning_effort": "high"},
                )
                full_text = ""
                for chunk_data in completion:
                    full_text = full_text + (chunk_data.choices[0].delta.content or "")
                logger.info(f"  ✅ gpt-oss-120b responded ({len(full_text)} chars)")

                # DEBUG PROBE
                debug_log_path = (
                    Cfg.OUTPUT_DIR / "llm_debug_dump.txt"
                    if Cfg.OUTPUT_DIR
                    else Path(tempfile.gettempdir()) / "llm_debug_dump.txt"
                )
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n\n--- GPT-OSS-120B RAW RESPONSE ---\n{full_text}\n--- END ---\n"
                    )

                if not full_text.strip():
                    raise RuntimeError("Received empty response string")

                return full_text
            except Exception as e:
                logger.warning(
                    f"  gpt-oss-120b failed ({e}), falling back to default LLM..."
                )

        # Fallback: use standard generate_text
        res_text = llm.generate_text(
            prompt,
            system_prompt=SCENE_BOUNDARY_SYSTEM_PROMPT,
            temperature=0.1,
            max_completion_tokens=Cfg.SCENE_REASONING_MAX_COMPLETION_TOKENS or None,
        )

        # DEBUG PROBE
        debug_log_path = (
            Cfg.OUTPUT_DIR / "llm_debug_dump.txt"
            if Cfg.OUTPUT_DIR
            else Path(tempfile.gettempdir()) / "llm_debug_dump.txt"
        )
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- FALLBACK RAW RESPONSE ---\n{res_text}\n--- END ---\n")

        return res_text

    def _align_and_refine_boundaries(
        self, llm_boundaries: List[float], shot_boundaries: List[float], duration: float
    ) -> List[float]:
        """Align LLM boundaries to optical cuts and apply min duration heuristic (NO max limit)."""
        # 1. Align LLM boundaries to nearest optical cut (within 15s)
        aligned = []
        for lb in llm_boundaries:
            if lb <= 0 or lb >= duration:
                continue
            closest_shot_b = min(shot_boundaries, key=lambda x: abs(x - lb))
            reason = self.boundary_reasons.get(lb, "LLM Semantic Boundary")

            if abs(closest_shot_b - lb) <= 15.0:
                aligned.append(closest_shot_b)
                self.aligned_reasons[closest_shot_b] = reason
            else:
                aligned.append(lb)
                self.aligned_reasons[lb] = reason

        aligned = sorted(list(set(aligned)))

        # 2. Heuristic Refinement (Min limit ONLY)
        refined = [0.0]

        for b in aligned:
            # Min Duration Rule (Avoid micro-flashes)
            if b - refined[-1] < self.min_scene_duration:
                continue
            refined.append(b)

        refined.append(duration)
        refined = sorted(list(set(refined)))

        # Strip 0 and duration to return just the internal boundaries
        return [b for b in refined if 0 < b < duration]

    def _group_shots(
        self, shots: List[Dict], boundaries: List[float], duration: float
    ) -> List[Dict]:
        scenes = []
        if not shots:
            return []

        current_scene_shots = []
        current_scene_reason = "Opening scene"
        boundary_idx = 0

        # Sentinel boundary
        if boundary_idx < len(boundaries):
            next_boundary = boundaries[boundary_idx]
        else:
            next_boundary = duration + 1.0

        for shot in shots:
            shot_midpoint = (shot["start_sec"] + shot["end_sec"]) / 2.0

            if shot_midpoint > next_boundary and current_scene_shots:
                # Close current scene with the reason that started it
                scenes.append(
                    self._make_scene_dict(
                        scenes, current_scene_shots, current_scene_reason
                    )
                )
                current_scene_shots = [shot]

                # Fetch reason for the next scene that just started
                current_scene_reason = self.aligned_reasons.get(
                    next_boundary, "Unknown boundary"
                )

                # Advance boundary
                while (
                    boundary_idx < len(boundaries)
                    and shot_midpoint > boundaries[boundary_idx]
                ):
                    boundary_idx += 1
                if boundary_idx < len(boundaries):
                    next_boundary = boundaries[boundary_idx]
                else:
                    next_boundary = duration + 1.0
            else:
                current_scene_shots.append(shot)

        if current_scene_shots:
            scenes.append(
                self._make_scene_dict(scenes, current_scene_shots, current_scene_reason)
            )

        # Fallback if somehow no scenes formed (e.g., only 1 scene)
        if not scenes:
            scenes.append(self._make_scene_dict(scenes, shots, "Opening scene"))

        return scenes

    def _make_scene_dict(
        self, existing_scenes: List[Dict], shot_group: List[Dict], reason: str
    ) -> Dict:
        """Create a single Scene dict from a list of shots."""
        idx = len(existing_scenes)
        first, last = shot_group[0], shot_group[-1]

        # calculate start and end times
        start_frame = first.get("start_frame", 0)
        end_frame = last.get("end_frame", 0)
        start_sec = first.get("start_sec", 0.0)
        end_sec = last.get("end_sec", 0.0)

        def _fmt_time(seconds: float) -> str:
            total = max(0, int(seconds))
            h, rem = divmod(total, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        return {
            "id": f"scene_{idx}",
            "description": reason,  # Story format
            "reason": reason,  # MovieNet format
            "shot": [first["shot_idx"], last["shot_idx"]],
            "frame": [start_frame, end_frame],
            "time": [start_sec, end_sec],
            "start_time": _fmt_time(start_sec),
            "end_time": _fmt_time(end_sec),
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration": [start_sec, end_sec],  # Story format
            "subtitle": [],  # Populated later
        }

    def _add_subtitles_to_scenes(
        self, scenes: List[Dict], subs: List[Dict]
    ) -> List[Dict]:
        """Merge subs into the scene dictionary structure."""
        for scene in scenes:
            start = scene["start_seconds"]
            end = scene["end_seconds"]

            scene_subs = []
            for s in subs:
                # Overlap check
                if s["start"] < end and s["end"] > start:
                    scene_subs.append(
                        {"duration": [s["start"], s["end"]], "sentences": [s["text"]]}
                    )
            scene["subtitle"] = scene_subs
        return scenes
