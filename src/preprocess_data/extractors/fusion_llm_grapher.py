"""
Fusion Grapher (LLM).

Fuses the three isolated data streams into a final JSON structure:
1. Temporal Visuals (from VLM)
2. Exact Subtitles (Speech to Text)
3. Deterministic Identities (from CV/DBSCAN + One-Shot Mapper)

Outputs the final clip_graph.json suitable for the Vector Index.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

from preprocess_data.config import PreprocessConfig as Cfg
from movierag.generation.universal_client import (
    LLMRateLimitError,
    UniversalLLMClient,
    is_rate_limit_error,
)

logger = logging.getLogger(__name__)

FUSION_PROMPT = """You are assembling a final Clip Graph for a movie scene.
You have three inputs for this continuous scene:
1. Visual Description (From VLM): Describes the setting and anonymous physical actions.
2. Subtitles (Dialogue): Describes what the characters are saying and when.
3. Identity Map (From CV): Shows how generic Face Clusters translate to TMDB Cast Names.

Your task is to merge the Visual Description and Subtitles, explicitly identifying who is speaking or acting based on the 'Identity Map'. Write a comprehensive Scene Description.

INPUTS:
======
Visuals:
{visuals}

Dialogue:
{dialogue}

Known Identity Map:
{identities}
======

Return EXACTLY a JSON dictionary holding the fused output:
{{
  "situation": "Short 3-5 word situation label (e.g., 'Coffee shop argument')",
  "description": "Comprehensive narrative description of the scene combining visuals and dialogue.",
  "characters": [{{"name": "Actual Character Name"}}],
  "attributes": ["tense", "romantic", "dark", "fast-paced"],
  "interactions": ["talking", "fighting", "running"],
  "scene_label": "The setting/location",
  "start_shot": {start_shot},
  "end_shot": {end_shot}
}}
Only return valid JSON."""


def _fmt_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class FusionLLMGrapher:
    def __init__(self):
        self.llm = UniversalLLMClient()

    def process_movie(self, movie_id: str, force: bool = False) -> List[Dict]:
        """Merge all semantic modalities into a final clip_graph."""
        logger.info(f"\n[6b/8] Multi-Modal Fusion Graphing for {movie_id}...")
        
        # 1. Load Data Paths
        kf_dir = Cfg.get_shot_keyf_dir() / movie_id
        ann_path = Cfg.get_annotation_dir() / f"{movie_id}.json"
        vlm_path = kf_dir / "vlm_temporal_descriptions.json"
        mapped_path = kf_dir / "mapped_identities.json"
        srt_path = Cfg.get_subtitle_dir() / f"{movie_id}.srt"
        out_path = self._graph_output_path(movie_id)
        existing_data = self._load_existing_graph(out_path) if out_path.exists() else {}
        clips_by_scene = {}
        if existing_data and not force:
            for clip in existing_data.get("clips", []):
                scene_id = clip.get("scene_id")
                if scene_id:
                    clips_by_scene[scene_id] = clip
            if existing_data.get("status") == "complete" and clips_by_scene:
                logger.info("  ⏩ Fusion graph already exists. Use force=True to overwrite.")
                return list(clips_by_scene.values())
            if clips_by_scene:
                logger.info(
                    "  ↻ Resuming fusion graph with %s cached clips.",
                    len(clips_by_scene),
                )
        
        if not vlm_path.exists():
            logger.error(f"  ❌ VLM descriptions not found. Run step 6a first.")
            return []
            
        with open(vlm_path, "r", encoding="utf-8") as f:
            vlm_data = json.load(f)
        
        meta = self._load_movie_meta(movie_id)
        cast_map = self._build_cast_map(meta)
        alias_map = self._build_character_alias_map(cast_map)
        identity_mapping: Dict[str, str] = {}
        identities_text = self._build_identity_context(identity_mapping, cast_map, [])
        if mapped_path.exists():
            with open(mapped_path, "r", encoding="utf-8") as f:
                ident_data = json.load(f)
                identity_mapping = ident_data.get("mappings", {})
                
        # Load Subtitles (Helper function)
        subtitles = self._load_subtitles(srt_path)
        
        # Load Semantic Scenes
        with open(ann_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)
            
        scenes = ann_data.get("scene", [])
        
        vlm_scenes = vlm_data.get("scenes", {})
        
        if not scenes or not vlm_scenes:
            logger.warning("  Missing scenes or VLM data.")
            return []
            
        clips = list(clips_by_scene.values())
        logger.info(f"  Fusing {len(scenes)} Semantic Scenes...")
        
        for i, scene in enumerate(scenes):
            scene_id = scene.get("id")
            if not scene_id or scene_id not in vlm_scenes:
                continue
            if scene_id in clips_by_scene and not force:
                continue

            shot_range = scene.get("shot", [i, i])
            if len(shot_range) < 2:
                shot_range = [i, i]

            start_sec = float(
                scene.get("start_seconds", scene["frame"][0] / ann_data.get("fps", 24))
            )
            end_sec = float(
                scene.get("end_seconds", scene["frame"][1] / ann_data.get("fps", 24))
            )
            
            # Get specific dialogue overlapping this scene
            scene_dialogue = self._filter_subtitles(subtitles, start_sec, end_sec)
            dialogue_text = "\n".join(scene_dialogue) if scene_dialogue else "No dialogue detected."
            
            # --- DEFENSIVE TOKEN MANAGEMENT (Fix Groq 413 Errors) ---
            MAX_DIALOGUE_CHARS = 15000  # ~3500 tokens
            if len(dialogue_text) > MAX_DIALOGUE_CHARS:
                logger.warning(f"  ⚠️ Scene {scene_id} dialogue heavily exceeds {MAX_DIALOGUE_CHARS} chars! Truncating to avoid 413 TPM Crash.")
                dialogue_text = dialogue_text[:MAX_DIALOGUE_CHARS] + "\n...[TRUNCATED DUE TO TPM LIMITS]"
                
            visuals_dict = vlm_scenes[scene_id]
            visuals_text = json.dumps(visuals_dict, indent=2)
            identities_text = self._build_identity_context(
                identity_mapping,
                cast_map,
                scene.get("script_characters", []),
            )
            
            prompt = FUSION_PROMPT.format(
                visuals=visuals_text,
                dialogue=dialogue_text,
                identities=identities_text,
                start_shot=int(shot_range[0]),
                end_shot=int(shot_range[1]),
            )
            
            try:
                response_text = self.llm.generate_text(
                    prompt=prompt,
                    system_prompt="You are a strict JSON returning agent. Return ONLY valid JSON.",
                    temperature=0.1,
                    max_completion_tokens=Cfg.FUSION_LLM_MAX_COMPLETION_TOKENS or None,
                )
                
                # Parse JSON
                import re
                json_str = response_text
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                
                clip = json.loads(json_str)
                # Ensure structure
                clip["clip_id"] = f"{movie_id}_clip_{i:04d}"
                clip["auto_generated"] = True
                clip["scene_id"] = scene_id
                clip["start_shot"] = int(shot_range[0])
                clip["end_shot"] = int(shot_range[1])
                clip["start_seconds"] = round(start_sec, 3)
                clip["end_seconds"] = round(end_sec, 3)
                clip["start_time"] = _fmt_hms(start_sec)
                clip["end_time"] = _fmt_hms(end_sec)
                clip["annotation_frame"] = scene.get("frame", [])
                clip["scene_type"] = scene.get("scene_type", "")
                clip["environment"] = scene.get("environment", "")
                clip["script_time_of_day"] = scene.get(
                    "script_time_of_day", scene.get("time_of_day", "")
                )
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
                clip = self._postprocess_clip(
                    clip,
                    scene,
                    cast_map=cast_map,
                    alias_map=alias_map,
                )
                clips.append(clip)
                clips_by_scene[scene_id] = clip
                self._save_scene_graph(
                    movie_id,
                    list(clips_by_scene.values()),
                    ann_data.get("movie_id", movie_id),
                    status="partial",
                )
                
            except Exception as e:
                if is_rate_limit_error(e):
                    self._save_scene_graph(
                        movie_id,
                        list(clips_by_scene.values()),
                        ann_data.get("movie_id", movie_id),
                        status="partial",
                        error_message=str(e),
                    )
                    raise LLMRateLimitError(
                        f"Fusion graphing hit rate limit for {movie_id}: {e}"
                    ) from e
                logger.error(f"  ❌ Fusion LLM failed for scene {scene_id}: {e}")
                
        # Save output
        if clips:
            out_path = self._save_scene_graph(
                movie_id,
                sorted(
                    clips,
                    key=lambda item: (
                        float(item.get("start_seconds", 0.0)),
                        item.get("clip_id", ""),
                    ),
                ),
                ann_data.get("movie_id", movie_id),
                status="complete",
            )
            logger.info(f"  ✅ Saved {len(clips)} fused clips to {out_path.name}")
            
        return clips

    @staticmethod
    def _graph_output_path(movie_id: str) -> Path:
        out_dir = Cfg.get_scene_graph_dir() / movie_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{movie_id}_auto_graph.json"

    @staticmethod
    def _load_existing_graph(out_path: Path) -> Dict:
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_scene_graph(
        self,
        movie_id: str,
        clips: List[Dict],
        title: str,
        status: str = "complete",
        error_message: str = "",
    ) -> Path:
        out_path = self._graph_output_path(movie_id)

        data = {
            "movie_id": movie_id,
            "title": title,
            "total_clips": len(clips),
            "generated_by": "FusionGraphLLM",
            "status": status,
            "last_error": error_message,
            "clips": clips,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return out_path
        
    def _load_subtitles(self, srt_path: Path) -> List[Dict]:
        if not srt_path.exists():
            return []
        import re
        text = ""
        for encoding in ("utf-8-sig", "cp1252", "utf-8"):
            try:
                text = srt_path.read_text(encoding=encoding).strip()
                break
            except UnicodeDecodeError:
                continue
        if not text:
            text = srt_path.read_text(encoding="utf-8", errors="replace").strip()
        lines = re.split(r"\n{2,}", text)
        subs = []
        for block in lines:
            parts = block.split('\n')
            if len(parts) >= 3:
                times = parts[1].split(' --> ')
                if len(times) == 2:
                    try:
                        starts = self._time_to_sec(times[0])
                        ends = self._time_to_sec(times[1])
                        text = " ".join([l.strip() for l in parts[2:] if l.strip()])
                        subs.append({"start": starts, "end": ends, "text": text})
                    except:
                        pass
        return subs
        
    def _time_to_sec(self, time_str: str) -> float:
        parts = time_str.replace(',', '.').split(':')
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
        
    def _filter_subtitles(self, subs: List[Dict], start_sec: float, end_sec: float) -> List[str]:
        filtered = []
        for s in subs:
            # Overlap check
            if s["start"] < end_sec and s["end"] > start_sec:
                filtered.append(f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}")
        return filtered

    @staticmethod
    def _clean_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @classmethod
    def _normalize_name(cls, value: Any) -> str:
        text = cls._clean_text(value)
        text = re.sub(r"\s*\(.*?\)", "", text)
        text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
        text = re.sub(r"\bO S\b|\bV O\b|\bOS\b|\bVO\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    @staticmethod
    def _load_movie_meta(movie_id: str) -> Dict[str, Any]:
        meta_path = Cfg.get_meta_dir() / f"{movie_id}.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @classmethod
    def _build_cast_map(cls, meta: Dict[str, Any]) -> Dict[str, str]:
        cast_map: Dict[str, str] = {}
        for cast_entry in meta.get("cast", []) or []:
            actor = cls._clean_text(cast_entry.get("name"))
            character = cls._clean_text(cast_entry.get("character"))
            if not actor or not character or character.lower() == "unknown":
                continue
            cast_map[actor] = character
        return cast_map

    @classmethod
    def _build_character_alias_map(cls, cast_map: Dict[str, str]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        for character in cast_map.values():
            normalized = cls._normalize_name(character)
            if not normalized:
                continue
            tokens = [token for token in normalized.split() if token]
            aliases = {normalized}
            if tokens:
                aliases.add(tokens[0])
                aliases.add(tokens[-1])
                if len(tokens) >= 2:
                    aliases.add(" ".join(tokens[:2]))
            for alias in aliases:
                if len(alias) >= 2:
                    alias_map.setdefault(alias, character)
        return alias_map

    @classmethod
    def _canonicalize_characters(
        cls, values: List[Any], alias_map: Dict[str, str]
    ) -> List[str]:
        generic = {
            "unknown",
            "man",
            "woman",
            "boy",
            "girl",
            "voice",
            "narrator",
            "teacher",
            "visitor",
            "stranger",
            "class",
        }
        resolved: List[str] = []
        for raw in values or []:
            if isinstance(raw, dict):
                raw = raw.get("name", "") or raw.get("character", "")
            cleaned = cls._clean_text(raw)
            normalized = cls._normalize_name(cleaned)
            if not normalized:
                continue
            candidate = alias_map.get(normalized)
            if not candidate:
                tokens = [token for token in normalized.split() if token]
                if not tokens or all(token in generic for token in tokens):
                    continue
                if len(tokens) == 1 and len(tokens[0]) < 2:
                    continue
                candidate = cleaned.title() if cleaned.isupper() else cleaned
            if candidate and candidate not in resolved:
                resolved.append(candidate)
        return resolved

    @classmethod
    def _build_cast_in_scene(
        cls, characters: List[str], cast_map: Dict[str, str]
    ) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for actor, character in cast_map.items():
            if character in characters:
                results.append({"actor": actor, "character": character})
        return results

    @classmethod
    def _build_identity_context(
        cls,
        identity_mapping: Dict[str, str],
        cast_map: Dict[str, str],
        script_characters: List[str],
    ) -> str:
        sections: List[str] = []
        if identity_mapping:
            sections.append(
                "Mapped face clusters:\n" + json.dumps(identity_mapping, indent=2, ensure_ascii=False)
            )
        cast_lines = [f"- {actor} as {character}" for actor, character in list(cast_map.items())[:15]]
        if cast_lines:
            sections.append("Cast roster:\n" + "\n".join(cast_lines))
        cleaned_script_chars = [cls._clean_text(name) for name in script_characters or [] if cls._clean_text(name)]
        if cleaned_script_chars:
            sections.append("Script character hints:\n- " + "\n- ".join(cleaned_script_chars[:8]))
        if not sections:
            return "No identities mapped."
        return "\n\n".join(sections)

    @classmethod
    def _postprocess_clip(
        cls,
        clip: Dict[str, Any],
        scene: Dict[str, Any],
        cast_map: Dict[str, str],
        alias_map: Dict[str, str],
    ) -> Dict[str, Any]:
        llm_characters = cls._canonicalize_characters(clip.get("characters", []), alias_map)
        script_characters = cls._canonicalize_characters(
            scene.get("script_characters", []),
            alias_map,
        )
        if not script_characters:
            for ref in scene.get("script_scene_refs", []) or []:
                script_characters.extend(
                    cls._canonicalize_characters(ref.get("characters", []), alias_map)
                )
        characters = script_characters or llm_characters
        if not characters and cast_map:
            heading_text = cls._normalize_name(scene.get("script_primary_heading", ""))
            for character in cast_map.values():
                norm_character = cls._normalize_name(character)
                if norm_character and norm_character in heading_text:
                    characters.append(character)
        deduped: List[str] = []
        for name in characters:
            if name and name not in deduped:
                deduped.append(name)
        clip["characters"] = deduped[:8]
        clip["cast_in_scene"] = cls._build_cast_in_scene(clip["characters"], cast_map)
        return clip
