"""
5-Layer Temporal Chunk Builder

Builds temporal chunks by merging 5 data layers:
  ① Timestamps (from annotation or keyframe_index.json)
  ② Semantics (from MovieGraphs clips or auto-generated)
  ③ Dialogue (from SRT subtitles)
  ④ Metadata (from meta JSON)
  ⑤ Keyframes (from shot_keyf/ with precise timestamps)

Supports TWO modes:
  A) Annotated: pre-existing annotation + MovieGraphs data
  B) Ingest: NEW video with no prior data — uses scene detection + STT

Adapted from: scripts/build_temporal_chunks.py
"""

import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from preprocess_data.config import PreprocessConfig as Cfg
from preprocess_data.indexing._screenplay_evidence import build_screenplay_payload
from .subtitle_parser import SubtitleParser

logger = logging.getLogger(__name__)

VISUAL_OBJECT_KEYWORDS = (
    "award",
    "bag",
    "bed",
    "bench",
    "blanket",
    "book",
    "briefcase",
    "bus",
    "cake",
    "car",
    "chair",
    "church",
    "coffee",
    "costume",
    "candle",
    "certificate",
    "classroom",
    "coat",
    "couch",
    "curtain",
    "desk",
    "door",
    "flowers",
    "food",
    "frame",
    "glass",
    "glasses",
    "hallway",
    "hospital",
    "kitchen",
    "lamp",
    "living room",
    "pancakes",
    "plant",
    "portrait",
    "prop room",
    "restaurant",
    "school",
    "shawl",
    "sofa",
    "soldiers",
    "stage",
    "stairs",
    "staircase",
    "street",
    "table",
    "television",
    "tent",
    "telescope",
    "toy",
    "train",
    "trophy",
    "uniform",
    "vase",
    "watch",
    "window",
    "wine",
)


class ChunkBuilder:
    """Build 5-layer temporal chunks for movies."""

    def __init__(self):
        self._keyframe_index_cache: Dict[str, List[Dict]] = {}
        self._vlm_scene_cache: Dict[str, Dict[str, Dict]] = {}
        self._script_scene_vlm_cache: Dict[str, Dict[str, Dict]] = {}
        self._script_scene_payload_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.max_chunk_duration_sec = float(Cfg.CHUNK_MAX_DURATION_SEC)
        self.min_script_chunk_overlap_sec = float(Cfg.CHUNK_MIN_SCRIPT_OVERLAP_SEC)
        self.max_keyframes_per_chunk = max(1, int(Cfg.CHUNK_MAX_KEYFRAMES))

    @staticmethod
    def _scene_idx_from_value(value, fallback: int = 0) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"(\d+)$", value)
            if match:
                return int(match.group(1))
        return fallback

    @staticmethod
    def _img_idx_from_name(path_str: str) -> int:
        name = Path(path_str).name
        match = re.search(r"-(\d+)\.jpg$", name, re.IGNORECASE)
        if match:
            return max(int(match.group(1)) - 1, 0)

        match = re.search(r"_img_(\d+)\.jpg$", name, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return 0

    @classmethod
    def _normalize_keyframe_entry(
        cls, keyframe: Dict, raw_shots: Optional[List[Dict]] = None
    ) -> Optional[Dict]:
        path = keyframe.get("path") or keyframe.get("frame_path")
        if not path:
            return None

        scene_idx = keyframe.get("scene_idx")
        if scene_idx is None or isinstance(scene_idx, str):
            scene_idx = cls._scene_idx_from_value(
                keyframe.get("scene_id"), keyframe.get("shot_id", 0)
            )

        shot_id = int(keyframe.get("shot_id", keyframe.get("shot_idx", 0)))
        img_idx = keyframe.get("img_idx")
        if img_idx is None:
            img_idx = cls._img_idx_from_name(str(path))

        timestamp_sec = keyframe.get("timestamp_sec", keyframe.get("timestamp"))
        if raw_shots and 0 <= shot_id < len(raw_shots):
            shot = raw_shots[shot_id]
            start_sec = float(shot.get("start_sec", 0.0))
            end_sec = float(shot.get("end_sec", start_sec))
            duration = max(end_sec - start_sec, 0.0)
            if duration > 0:
                # Match the original thumbnail timing used by save_images/keyframe linking.
                timestamp_sec = start_sec + duration * ((int(img_idx) + 0.5) / 3.0)

        if timestamp_sec is None:
            timestamp_sec = 0.0

        return {
            "path": str(path),
            "timestamp_sec": float(timestamp_sec),
            "scene_idx": int(scene_idx),
            "shot_id": shot_id,
            "img_idx": int(img_idx),
            "source": keyframe.get("source", "manifest"),
        }

    @staticmethod
    def _load_raw_shots(movie_id: str) -> List[Dict]:
        ann_path = Cfg.get_annotation_dir() / f"{movie_id}.json"
        if not ann_path.exists():
            return []
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return data.get("raw_shots", [])

    # ═══════════════════════════════════════════════════════════
    # Layer ⑤: Keyframe time index (with precision + fallback)
    # ═══════════════════════════════════════════════════════════

    def _load_keyframe_index(self, movie_id: str) -> List[Dict]:
        """Load keyframes with timestamps (from index JSON or heuristic)."""
        if movie_id in self._keyframe_index_cache:
            return self._keyframe_index_cache[movie_id]

        # Find keyframe directory
        keyf_dir = None
        for d in Cfg.get_keyframe_search_dirs():
            candidate = d / movie_id
            if not candidate.exists():
                continue

            has_manifest = any(
                (candidate / manifest_name).exists()
                for manifest_name in (
                    "vector_clean_index.json",
                    "keyframe_index.json",
                    "vlm_quality_index.json",
                )
            )
            has_images = any(candidate.glob("*.jpg")) or any(
                any(nested.glob("*.jpg"))
                for nested in (
                    candidate / "vector_clean",
                    candidate / "vlm_quality",
                    candidate / "faces",
                )
                if nested.exists()
            )

            if has_manifest or has_images:
                keyf_dir = candidate
                break

        if not keyf_dir:
            self._keyframe_index_cache[movie_id] = []
            return []

        # Strategy A: normalized manifest from the new two-layer pipeline.
        raw_shots = self._load_raw_shots(movie_id)

        for manifest_name in (
            "vector_clean_index.json",
            "keyframe_index.json",
            "vlm_quality_index.json",
        ):
            idx_path = keyf_dir / manifest_name
            if not idx_path.exists():
                continue

            try:
                data = json.loads(idx_path.read_text(encoding="utf-8"))
                result = []
                for keyframe in data.get("keyframes", []):
                    normalized = self._normalize_keyframe_entry(
                        keyframe, raw_shots=raw_shots
                    )
                    if normalized:
                        normalized["source"] = manifest_name
                        result.append(normalized)
                self._keyframe_index_cache[movie_id] = result
                return result
            except Exception as e:
                logger.warning(f"Failed to read keyframe manifest {idx_path.name}: {e}")

        # Strategy B: Heuristic (shot_num × 3s)
        result = []
        legacy_images = list(keyf_dir.glob("shot_*_img_*.jpg"))
        if not legacy_images:
            legacy_images = list((keyf_dir / "vector_clean").glob("*.jpg"))

        for img_path in sorted(legacy_images):
            match = re.match(r"shot_(\d+)_img_(\d+)\.jpg", img_path.name)
            if match:
                shot_num = int(match.group(1))
                img_idx = int(match.group(2))
            else:
                shot_match = re.search(r"shot_(\d+)-(\d+)\.jpg", img_path.name)
                if not shot_match:
                    continue
                shot_num = int(shot_match.group(1)) - 1
                img_idx = max(int(shot_match.group(2)) - 1, 0)

            result.append(
                {
                    "path": str(img_path),
                    "timestamp_sec": shot_num * Cfg.KEYFRAME_INTERVAL_SEC,
                    "scene_idx": shot_num,
                    "shot_id": shot_num,
                    "img_idx": img_idx,
                    "source": "heuristic_3s",
                }
            )

        self._keyframe_index_cache[movie_id] = result
        return result

    def _find_keyframes_by_time(
        self, movie_id: str, start_sec: float, end_sec: float
    ) -> List[str]:
        """Find keyframes in [start_sec, end_sec] time range."""
        kf_index = self._load_keyframe_index(movie_id)
        if not kf_index or end_sec <= start_sec:
            return []

        # Group by shot_id to avoid collapsing an entire semantic scene into one frame.
        groups: Dict[int, List[Dict]] = {}
        for kf in kf_index:
            if start_sec <= kf["timestamp_sec"] <= end_sec:
                shot_id = int(kf.get("shot_id", kf.get("scene_idx", 0)))
                groups.setdefault(shot_id, []).append(kf)

        IMG_PRIORITY = {1: 0, 0: 1, 2: 2}
        selected: List[Dict[str, Any]] = []
        for shot_id in sorted(groups.keys()):
            best = sorted(
                groups[shot_id],
                key=lambda x: (
                    IMG_PRIORITY.get(int(x.get("img_idx", 0)), 99),
                    abs(float(x.get("timestamp_sec", 0.0)) - ((start_sec + end_sec) / 2.0)),
                ),
            )
            if best:
                selected.append(best[0])

        if len(selected) > self.max_keyframes_per_chunk:
            step = len(selected) / float(self.max_keyframes_per_chunk)
            selected = [
                selected[min(int(i * step), len(selected) - 1)]
                for i in range(self.max_keyframes_per_chunk)
            ]

        paths = []
        for item in selected:
            resolved_path = Cfg.resolve_keyframe_path(movie_id, item["path"])
            if resolved_path not in paths:
                paths.append(resolved_path)
        return paths

    def _load_vlm_scenes(self, movie_id: str) -> Dict[str, Dict]:
        if movie_id in self._vlm_scene_cache:
            return self._vlm_scene_cache[movie_id]

        path = Cfg.get_shot_keyf_dir() / movie_id / "vlm_temporal_descriptions.json"
        if not path.exists():
            self._vlm_scene_cache[movie_id] = {}
            return {}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scenes = payload.get("scenes", {}) if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.warning("Failed to load VLM scene map for %s: %s", movie_id, exc)
            scenes = {}

        self._vlm_scene_cache[movie_id] = scenes
        return scenes

    def _load_script_scene_vlm(self, movie_id: str) -> Dict[str, Dict]:
        if movie_id in self._script_scene_vlm_cache:
            return self._script_scene_vlm_cache[movie_id]

        path = Cfg.get_shot_keyf_dir() / movie_id / "vlm_script_scene_descriptions.json"
        if not path.exists():
            self._script_scene_vlm_cache[movie_id] = {}
            return {}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scenes = payload.get("scenes", {}) if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.warning(
                "Failed to load script-scene VLM map for %s: %s", movie_id, exc
            )
            scenes = {}

        self._script_scene_vlm_cache[movie_id] = scenes
        return scenes

    def _load_script_scene_payloads(self, movie_id: str) -> Dict[str, Dict[str, Any]]:
        if movie_id in self._script_scene_payload_cache:
            return self._script_scene_payload_cache[movie_id]

        payloads: Dict[str, Dict[str, Any]] = {}
        try:
            from preprocess_data.extractors.script_aligner import ScriptAligner

            aligned_scenes = ScriptAligner().align(movie_id, force=False)
            for scene in aligned_scenes:
                scene_uid = str(getattr(scene, "scene_uid", "") or "").strip()
                if not scene_uid:
                    continue
                payloads[scene_uid] = build_screenplay_payload(scene)
        except Exception as exc:
            logger.warning(
                "Failed to load screenplay payloads for %s: %s", movie_id, exc
            )

        self._script_scene_payload_cache[movie_id] = payloads
        return payloads

    def _attach_screenplay_payload(
        self,
        movie_id: str,
        clip_payload: Dict[str, Any],
        script_ref: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        merged = dict(clip_payload)
        scene_uid = ""
        if script_ref:
            scene_uid = str(script_ref.get("script_scene_uid", "") or "").strip()
        if not scene_uid:
            scene_uid = str(merged.get("script_scene_uid", "") or "").strip()

        screenplay_payload = {}
        if scene_uid:
            screenplay_payload = self._load_script_scene_payloads(movie_id).get(
                scene_uid, {}
            )

        if screenplay_payload:
            merged.update(screenplay_payload)

        merged.setdefault(
            "alignment_confidence",
            round(float((script_ref or {}).get("confidence_score", 0.0) or 0.0), 3),
        )

        if (
            not self._clean_text(merged.get("dialogue_excerpt", ""))
            and self._clean_text(merged.get("screenplay_dialogue_excerpt", ""))
        ):
            merged["dialogue_excerpt"] = self._clean_text(
                merged.get("screenplay_dialogue_excerpt", "")
            )

        return merged

    @staticmethod
    def _clean_text(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text

    @classmethod
    def _truncate_text(cls, value: str, max_chars: int) -> str:
        text = cls._clean_text(value)
        if len(text) <= max_chars:
            return text
        clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
        if not clipped:
            clipped = text[:max_chars].strip()
        return clipped.rstrip(".") + "..."

    @classmethod
    def _sentence_excerpt(
        cls, value: str, max_sentences: int = 2, max_chars: int = 320
    ) -> str:
        text = cls._clean_text(value)
        if not text:
            return ""
        parts: List[str] = []
        total = 0
        for sentence in re.split(r"(?<=[\.\!\?])\s+", text):
            cleaned = cls._clean_text(sentence)
            if not cleaned:
                continue
            if total + len(cleaned) + 1 > max_chars:
                break
            parts.append(cleaned)
            total += len(cleaned) + 1
            if len(parts) >= max_sentences:
                break
        if parts:
            return " ".join(parts)
        return cls._truncate_text(text, max_chars)

    @classmethod
    def _extract_visual_objects(cls, *texts: str) -> List[str]:
        combined = " ".join(cls._clean_text(text).lower() for text in texts if text)
        if not combined:
            return []
        results: List[str] = []
        for keyword in VISUAL_OBJECT_KEYWORDS:
            if keyword in combined and keyword not in results:
                results.append(keyword)
            if len(results) >= max(1, int(Cfg.VISUAL_FALLBACK_MAX_OBJECTS)):
                break
        return results

    @classmethod
    def _humanize_location(cls, location: str, time_of_day: str = "") -> str:
        text = cls._clean_text(location)
        if not text:
            return ""
        text = re.sub(r"\bINT\./EXT\.|\bINT/EXT\.|\bEXT/INT\.", "interior/exterior", text, flags=re.IGNORECASE)
        text = re.sub(r"\bINT\.", "interior", text, flags=re.IGNORECASE)
        text = re.sub(r"\bEXT\.", "exterior", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" -")
        tod = cls._clean_text(time_of_day).lower()
        if tod and tod not in text.lower():
            return f"{text} during {tod}"
        return text

    @classmethod
    def _merge_visual_payload(
        cls, primary: Dict[str, Any], secondary: Dict[str, Any]
    ) -> Dict[str, Any]:
        primary = primary or {}
        secondary = secondary or {}
        merged = {
            "setting": cls._clean_text(primary.get("setting") or secondary.get("setting", "")),
            "actions": cls._clean_text(primary.get("actions") or secondary.get("actions", "")),
            "visual_focus": cls._clean_text(
                primary.get("visual_focus") or secondary.get("visual_focus", "")
            ),
        }
        objects: List[str] = []
        for source in (primary.get("visual_objects", []), secondary.get("visual_objects", [])):
            for item in source or []:
                cleaned = cls._clean_text(item).lower()
                if cleaned and cleaned not in objects:
                    objects.append(cleaned)
        if objects:
            merged["visual_objects"] = objects[: max(1, int(Cfg.VISUAL_FALLBACK_MAX_OBJECTS))]
        return {key: value for key, value in merged.items() if value}

    @classmethod
    def _build_visual_fallback(
        cls,
        clip: Dict[str, Any],
        script_ref: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        heading = cls._clean_text(
            (script_ref or {}).get("heading")
            or clip.get("script_primary_heading", "")
            or clip.get("scene_label", "")
        )
        location = cls._clean_text(
            (script_ref or {}).get("location") or clip.get("script_location", "")
        )
        time_of_day = cls._clean_text(
            (script_ref or {}).get("time_of_day") or clip.get("script_time_of_day", "")
        )
        screenplay_action = cls._clean_text(clip.get("screenplay_action_excerpt", ""))
        screenplay_context = cls._clean_text(clip.get("screenplay_context_excerpt", ""))
        semantic_description = cls._clean_text(clip.get("description", ""))
        scene_label = cls._clean_text(clip.get("scene_label", ""))
        characters = [
            cls._clean_text(name)
            for name in (clip.get("characters", []) or [])
            if cls._clean_text(name)
        ]

        humanized_location = cls._humanize_location(location or heading, time_of_day)
        setting = ""
        if humanized_location:
            setting = f"Scene set in {humanized_location}."
        if screenplay_action:
            action_setting = cls._sentence_excerpt(screenplay_action, max_sentences=2, max_chars=260)
            if action_setting:
                setting = " ".join(part for part in (setting, action_setting) if part).strip()
        if not setting and semantic_description:
            setting = cls._sentence_excerpt(semantic_description, max_sentences=2, max_chars=260)

        actions = cls._sentence_excerpt(
            screenplay_action or semantic_description or screenplay_context,
            max_sentences=3,
            max_chars=340,
        )
        visual_focus = heading or scene_label
        if not visual_focus and characters:
            visual_focus = ", ".join(characters[:2])

        visual_objects = cls._extract_visual_objects(
            heading,
            location,
            screenplay_action,
            screenplay_context,
            semantic_description,
        )

        payload = {
            "setting": setting,
            "actions": actions,
            "visual_focus": visual_focus,
            "visual_objects": visual_objects,
        }
        return {key: value for key, value in payload.items() if value}

    @classmethod
    def _normalize_name(cls, value: str) -> str:
        text = cls._clean_text(value)
        text = re.sub(r"\(.*?\)", "", text)
        text = text.replace("'s", " ")
        text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    @classmethod
    def _clean_dialogue_lines(cls, dialogues: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen_counts: Dict[str, int] = {}
        for raw in dialogues or []:
            text = cls._clean_text(raw)
            if not text:
                continue

            alpha_chars = sum(ch.isalpha() for ch in text)
            if alpha_chars < 2:
                continue

            if alpha_chars / max(len(text), 1) < 0.45:
                continue

            norm = cls._normalize_name(text)
            if not norm:
                continue

            seen_counts[norm] = seen_counts.get(norm, 0) + 1
            if len(norm.split()) <= 4 and seen_counts[norm] > 2:
                continue

            cleaned.append(text)
        return cleaned

    @staticmethod
    def _dialogue_excerpt(dialogues: List[str], max_chars: int = 420) -> str:
        excerpt_parts: List[str] = []
        total = 0
        for line in dialogues:
            if total + len(line) + 1 > max_chars:
                break
            excerpt_parts.append(line)
            total += len(line) + 1
        return " ".join(excerpt_parts).strip()

    @classmethod
    def _dialogue_quality(cls, text: str) -> float:
        cleaned = cls._clean_text(text)
        if not cleaned:
            return 0.0

        tokens = [token for token in re.findall(r"\w+", cleaned.lower()) if token]
        if not tokens:
            return 0.0

        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        alpha_ratio = sum(ch.isalpha() for ch in cleaned) / max(len(cleaned), 1)
        repeated_thanks = cleaned.lower().count("thank you")
        noise_markers = (
            "transciprokey",
            "advanced effects",
            "top 10",
            "subtitle",
            "credits",
            "the end",
        )
        noise_hits = sum(1 for marker in noise_markers if marker in cleaned.lower())

        score = 0.50
        score += min(0.25, unique_ratio * 0.35)
        score += min(0.20, alpha_ratio * 0.20)
        if len(tokens) >= 12:
            score += 0.10
        if repeated_thanks >= 3:
            score -= 0.25
        if noise_hits:
            score -= 0.20 * noise_hits
        return max(0.0, min(1.0, score))

    @classmethod
    def _select_primary_dialogue(
        cls,
        clip: Dict[str, Any],
        subtitle_excerpt: str,
        max_chars: int = 900,
    ) -> tuple[str, str]:
        subtitle_excerpt = cls._clean_text(subtitle_excerpt)
        screenplay_excerpt = cls._clean_text(
            clip.get("screenplay_dialogue_excerpt", "")
        )
        if screenplay_excerpt:
            screenplay_excerpt = screenplay_excerpt[:max_chars]
        if subtitle_excerpt:
            subtitle_excerpt = subtitle_excerpt[:max_chars]

        confidence = float(
            clip.get(
                "alignment_confidence",
                ((clip.get("dominant_script_scene_ref") or {}).get("confidence_score", 0.0) or 0.0),
            )
            or 0.0
        )
        subtitle_quality = cls._dialogue_quality(subtitle_excerpt)

        if screenplay_excerpt and not subtitle_excerpt:
            return screenplay_excerpt, "screenplay"
        if subtitle_excerpt and not screenplay_excerpt:
            return subtitle_excerpt, "subtitle"
        if not screenplay_excerpt and not subtitle_excerpt:
            return "", "none"

        if clip.get("chunk_source") == "script_scene":
            if confidence >= 0.45:
                return screenplay_excerpt, "screenplay"
            if subtitle_quality < 0.55:
                return screenplay_excerpt, "screenplay"
            if len(screenplay_excerpt) >= int(len(subtitle_excerpt) * 1.35):
                return screenplay_excerpt, "screenplay"

        if confidence >= 0.70 and screenplay_excerpt:
            return screenplay_excerpt, "screenplay"

        return subtitle_excerpt, "subtitle"

    @classmethod
    def _resolve_dialogue_full_text(
        cls,
        clip: Dict[str, Any],
        dialogues: Optional[List[str]],
    ) -> str:
        subtitle_lines = cls._clean_dialogue_lines(dialogues or [])
        subtitle_full = cls._clean_text(" ".join(subtitle_lines))
        subtitle_excerpt = cls._clean_text(clip.get("subtitle_dialogue_excerpt", ""))
        screenplay_turns = [
            cleaned
            for cleaned in (
                cls._clean_text(turn)
                for turn in (clip.get("screenplay_dialogue_turns", []) or [])
            )
            if cleaned
        ]
        screenplay_full = cls._clean_text(" ".join(screenplay_turns))
        if not screenplay_full:
            screenplay_full = cls._clean_text(
                clip.get("screenplay_dialogue_excerpt", "")
            )
        primary = cls._clean_text(clip.get("dialogue_excerpt", ""))
        source = str(clip.get("dialogue_source", "subtitle") or "subtitle").strip().lower()

        if source == "screenplay":
            return screenplay_full or primary or subtitle_full or subtitle_excerpt
        if source == "subtitle":
            return subtitle_full or subtitle_excerpt or primary or screenplay_full
        return primary or screenplay_full or subtitle_full or subtitle_excerpt

    def _resolve_vlm_payload(
        self,
        movie_id: str,
        clip: Dict[str, Any],
        script_ref: Optional[Dict[str, Any]],
        fallback_vlm: Dict[str, Any],
    ) -> tuple[Dict[str, Any], str]:
        screenplay_fallback = self._build_visual_fallback(clip, script_ref)
        scene_uid = ""
        if script_ref:
            scene_uid = str(script_ref.get("script_scene_uid", "") or "").strip()
        if not scene_uid:
            scene_uid = str(clip.get("script_scene_uid", "") or "").strip()

        if scene_uid:
            script_vlm = self._load_script_scene_vlm(movie_id).get(scene_uid)
            if isinstance(script_vlm, dict) and script_vlm:
                merged = self._merge_visual_payload(
                    script_vlm,
                    screenplay_fallback,
                )
                return merged, "script_scene_vlm"

        if fallback_vlm:
            if script_ref or clip.get("chunk_source") == "script_scene":
                merged = self._merge_visual_payload(
                    screenplay_fallback,
                    fallback_vlm,
                )
                source = (
                    "screenplay_visual_fallback+semantic_scene_vlm"
                    if screenplay_fallback
                    else "semantic_scene_vlm"
                )
            else:
                merged = self._merge_visual_payload(
                    fallback_vlm,
                    screenplay_fallback,
                )
                source = (
                    "semantic_scene_vlm+screenplay"
                    if screenplay_fallback
                    else "semantic_scene_vlm"
                )
            return merged, source
        if screenplay_fallback:
            return screenplay_fallback, "screenplay_visual_fallback"
        return {}, "none"

    @classmethod
    def _build_cast_catalog(cls, cast_map: Dict[str, str]) -> List[Dict[str, Any]]:
        descriptor_terms = {
            "mother",
            "father",
            "mom",
            "dad",
            "man",
            "woman",
            "boy",
            "girl",
            "voice",
            "teacher",
            "doctor",
            "nurse",
            "waiter",
            "waitress",
            "visitor",
            "stranger",
            "class",
            "narrator",
            "secretary",
        }

        catalog: List[Dict[str, Any]] = []
        for actor, character in (cast_map or {}).items():
            canonical = cls._clean_text(re.sub(r"\(.*?\)", "", str(character or "")))
            norm = cls._normalize_name(canonical)
            if not norm:
                continue
            tokens = [token for token in norm.split() if token]
            has_descriptor = any(token in descriptor_terms for token in tokens)
            aliases = {norm}
            if tokens and not has_descriptor:
                aliases.add(tokens[0])
                if len(tokens) > 1:
                    aliases.add(tokens[-1])
            catalog.append(
                {
                    "actor": actor,
                    "character": canonical,
                    "aliases": aliases,
                }
            )
        return catalog

    @classmethod
    def _map_names_to_cast(
        cls, names: List[str], cast_map: Dict[str, str]
    ) -> List[str]:
        catalog = cls._build_cast_catalog(cast_map)
        generic_terms = {
            "female",
            "male",
            "teacher",
            "man",
            "woman",
            "boy",
            "girl",
            "voice",
            "narrator",
            "class",
            "unknown",
        }
        resolved: List[str] = []
        for raw_name in names or []:
            if isinstance(raw_name, dict):
                raw_name = raw_name.get("name", "") or raw_name.get("character", "")
            norm = cls._normalize_name(raw_name)
            if not norm or norm == "unknown":
                continue
            match = next(
                (
                    item["character"]
                    for item in catalog
                    if norm in item["aliases"]
                ),
                None,
            )
            if match:
                candidate = match
            else:
                tokens = [token for token in norm.split() if token]
                if not tokens or all(token in generic_terms for token in tokens):
                    continue
                if len(tokens) == 1 and len(tokens[0]) < 2:
                    continue
                candidate = cls._clean_text(raw_name)
            if candidate and candidate not in resolved:
                resolved.append(candidate)
        return resolved

    @classmethod
    def _curate_characters(
        cls,
        clip_characters: List[str],
        script_characters: List[str],
        cast_map: Dict[str, str],
    ) -> List[str]:
        curated = cls._map_names_to_cast(script_characters, cast_map)
        if curated:
            return curated[:6]

        clip_matches = cls._map_names_to_cast(clip_characters, cast_map)
        if clip_matches:
            return clip_matches[:6]

        fallback: List[str] = []
        for name in clip_characters or []:
            cleaned = cls._clean_text(name)
            if not cleaned or cleaned.lower() == "unknown":
                continue
            if cleaned not in fallback:
                fallback.append(cleaned)
        return fallback[:6]

    @classmethod
    def _build_cast_in_scene(
        cls, characters: List[str], cast_map: Dict[str, str]
    ) -> List[Dict[str, str]]:
        catalog = cls._build_cast_catalog(cast_map)
        result: List[Dict[str, str]] = []
        for name in characters or []:
            norm = cls._normalize_name(name)
            match = next(
                (item for item in catalog if norm in item["aliases"]),
                None,
            )
            if match:
                result.append(
                    {"actor": match["actor"], "character": match["character"]}
                )
        return result

    @classmethod
    def _script_overlap_refs(
        cls, clip: Dict, start_sec: float, end_sec: float
    ) -> List[Dict]:
        refs: List[Dict] = []
        for ref in clip.get("script_scene_refs", []) or []:
            overlap_start = max(start_sec, float(ref.get("start_sec", start_sec)))
            overlap_end = min(end_sec, float(ref.get("end_sec", end_sec)))
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap <= 0:
                continue
            ref_copy = dict(ref)
            ref_copy["overlap_seconds"] = round(overlap, 2)
            ref_copy["segment_start_seconds"] = round(overlap_start, 2)
            ref_copy["segment_end_seconds"] = round(overlap_end, 2)
            refs.append(ref_copy)
        return sorted(
            refs,
            key=lambda item: (
                float(item.get("segment_start_seconds", 0.0)),
                float(item.get("segment_end_seconds", 0.0)),
            ),
        )

    def _should_expand_script_chunks(
        self, start_sec: float, end_sec: float, script_refs: List[Dict]
    ) -> bool:
        duration = max(0.0, end_sec - start_sec)
        indexable = [
            ref
            for ref in script_refs
            if float(ref.get("overlap_seconds", 0.0)) >= self.min_script_chunk_overlap_sec
            or str(ref.get("anchor_quality", "linear")) in {"full", "partial"}
        ]
        if len(indexable) >= 2:
            return True
        return duration > self.max_chunk_duration_sec and len(script_refs) >= 2

    @classmethod
    def _compose_chunk_description(
        cls,
        clip: Dict,
        script_ref: Optional[Dict],
        vlm_scene: Dict,
        dialogue_excerpt: str,
        characters: List[str],
    ) -> str:
        parts: List[str] = []
        heading = ""
        if script_ref:
            heading = cls._clean_text(script_ref.get("heading", ""))
        if heading:
            parts.append(f"{heading}.")

        screenplay_action = cls._truncate_text(
            clip.get("screenplay_action_excerpt", ""), 360
        )
        screenplay_dialogue = cls._truncate_text(
            clip.get("screenplay_dialogue_excerpt", ""), 420
        )
        screenplay_context = cls._truncate_text(
            clip.get("screenplay_context_excerpt", ""), 360
        )
        setting = cls._truncate_text(vlm_scene.get("setting", ""), 260)
        actions = cls._truncate_text(vlm_scene.get("actions", ""), 320)
        visual_focus = cls._truncate_text(vlm_scene.get("visual_focus", ""), 160)
        clip_description = cls._truncate_text(clip.get("description", ""), 320)
        visual_objects = [
            cls._clean_text(item)
            for item in (vlm_scene.get("visual_objects", []) or [])
            if cls._clean_text(item)
        ]
        dialogue_excerpt = cls._truncate_text(dialogue_excerpt, 420)
        dominant_script_uid = (
            (clip.get("dominant_script_scene_ref") or {}).get("script_scene_uid", "")
        )
        include_parent_vlm = True
        if (
            script_ref
            and int(clip.get("script_scene_count", 0) or 0) > 3
            and dominant_script_uid
            and dominant_script_uid != script_ref.get("script_scene_uid", "")
        ):
            include_parent_vlm = False

        if setting and include_parent_vlm:
            parts.append(f"Setting: {setting.rstrip('.')}.")
        if actions and include_parent_vlm:
            parts.append(f"Visual action: {actions.rstrip('.')}.")
        if visual_focus and include_parent_vlm:
            parts.append(f"Visual focus: {visual_focus.rstrip('.')}.")
        if visual_objects:
            parts.append(
                f"Visible objects: {', '.join(visual_objects[:6])}."
            )
        elif clip_description and not script_ref:
            parts.append(clip_description.rstrip(".") + ".")

        if screenplay_action:
            parts.append(f"Screenplay action: {screenplay_action.rstrip('.')}.")
        elif screenplay_context:
            parts.append(f"Screenplay context: {screenplay_context.rstrip('.')}.")

        if characters:
            parts.append(f"Characters in focus: {', '.join(characters)}.")

        dialogue_norm = cls._clean_text(dialogue_excerpt).lower()
        screenplay_dialogue_norm = cls._clean_text(screenplay_dialogue).lower()
        if screenplay_dialogue:
            parts.append(f"Screenplay dialogue: {screenplay_dialogue.rstrip('.')}.")

        if dialogue_excerpt and dialogue_norm != screenplay_dialogue_norm:
            parts.append(f"Dialogue focus: {dialogue_excerpt.rstrip('.')}.")

        return " ".join(parts).strip()

    @staticmethod
    def _allow_vlm_for_script_ref(clip: Dict, script_ref: Dict) -> bool:
        if not script_ref:
            return True

        dominant_script_uid = (
            (clip.get("dominant_script_scene_ref") or {}).get("script_scene_uid", "")
        )
        anchor_quality = str(script_ref.get("anchor_quality", "linear"))
        confidence = float(script_ref.get("confidence_score", 0.0) or 0.0)

        if dominant_script_uid and dominant_script_uid == script_ref.get("script_scene_uid", ""):
            return True
        if anchor_quality in {"full", "partial"} and confidence >= 0.45:
            return True
        if confidence >= 0.75:
            return True
        return False

    def _expand_clip_into_segments(
        self,
        movie_id: str,
        clip: Dict,
        start_sec: float,
        end_sec: float,
        cast_map: Dict[str, str],
        vlm_scene: Dict,
        srt_entries: List[Dict],
    ) -> List[Dict[str, Any]]:
        script_refs = self._script_overlap_refs(clip, start_sec, end_sec)
        if not self._should_expand_script_chunks(start_sec, end_sec, script_refs):
            return []

        segments: List[Dict[str, Any]] = []
        for ref in script_refs:
            seg_start = float(ref.get("segment_start_seconds", start_sec))
            seg_end = float(ref.get("segment_end_seconds", end_sec))
            overlap = max(0.0, seg_end - seg_start)
            if (
                overlap < self.min_script_chunk_overlap_sec
                and str(ref.get("anchor_quality", "linear")) == "linear"
            ):
                continue

            raw_dialogues = SubtitleParser.align(srt_entries, seg_start, seg_end)
            dialogues = self._clean_dialogue_lines(raw_dialogues)
            dialogue_excerpt = self._dialogue_excerpt(dialogues)
            characters = self._curate_characters(
                clip.get("characters", []),
                ref.get("characters", []),
                cast_map,
            )
            include_vlm = self._allow_vlm_for_script_ref(clip, ref)
            segment_clip = {
                **clip,
                "chunk_source": "script_scene",
                "parent_scene_id": clip.get("scene_id", ""),
                "parent_clip_id": clip.get("clip_id", ""),
                "script_scene_uid": ref.get("script_scene_uid", ""),
                "script_scene_refs": [ref],
                "script_scene_count": 1,
                "script_headings": [ref.get("heading", "")] if ref.get("heading") else [],
                "script_primary_heading": ref.get("heading", ""),
                "dominant_script_scene_ref": ref,
                "dominant_script_overlap_sec": ref.get("overlap_seconds", overlap),
                "script_location": ref.get("location", clip.get("script_location", "")),
                "script_time_of_day": ref.get(
                    "time_of_day", clip.get("script_time_of_day", "")
                ),
                "script_characters": list(ref.get("characters", []) or []),
                "scene_label": ref.get("heading", clip.get("scene_label", "")),
                "situation": clip.get("situation", ref.get("heading", "")),
                "characters": characters,
                "subtitle_dialogue_excerpt": dialogue_excerpt,
                "description": self._compose_chunk_description(
                    clip=clip,
                    script_ref=ref,
                    vlm_scene=vlm_scene if include_vlm else {},
                    dialogue_excerpt=dialogue_excerpt,
                    characters=characters,
                ),
            }
            segment_clip = self._attach_screenplay_payload(
                movie_id, segment_clip, script_ref=ref
            )
            resolved_vlm, vision_source = self._resolve_vlm_payload(
                movie_id,
                segment_clip,
                ref,
                vlm_scene if include_vlm else {},
            )
            primary_dialogue, dialogue_source = self._select_primary_dialogue(
                segment_clip,
                segment_clip.get("subtitle_dialogue_excerpt", dialogue_excerpt),
            )
            segment_clip.update(
                {
                    "dialogue_excerpt": primary_dialogue,
                    "dialogue_source": dialogue_source,
                    "vision_setting": self._clean_text(resolved_vlm.get("setting", "")),
                    "vision_actions": self._clean_text(resolved_vlm.get("actions", "")),
                    "vision_objects": list(resolved_vlm.get("visual_objects", []) or []),
                    "visual_focus": self._clean_text(resolved_vlm.get("visual_focus", "")),
                    "vision_source": vision_source,
                }
            )
            segment_clip["description"] = self._compose_chunk_description(
                clip=segment_clip,
                script_ref=ref,
                vlm_scene=resolved_vlm,
                dialogue_excerpt=primary_dialogue,
                characters=characters,
            )
            segments.append(
                {
                    "start_seconds": seg_start,
                    "end_seconds": seg_end,
                    "clip": segment_clip,
                    "dialogues": dialogues,
                }
            )
        return segments

    def _enrich_clip_for_chunk(
        self, movie_id: str, clip: Dict, cast_map: Dict[str, str], vlm_scene: Dict
    ) -> Dict[str, Any]:
        dominant = clip.get("dominant_script_scene_ref") or {}
        characters = self._curate_characters(
            clip.get("characters", []),
            clip.get("script_characters", []),
            cast_map,
        )
        subtitle_dialogue_excerpt = self._dialogue_excerpt(
            self._clean_dialogue_lines(clip.get("dialogue", []))
        )
        enriched = {
            **clip,
            "chunk_source": clip.get("chunk_source", "semantic_scene"),
            "parent_scene_id": clip.get("scene_id", ""),
            "parent_clip_id": clip.get("clip_id", ""),
            "script_scene_uid": dominant.get("script_scene_uid", ""),
            "characters": characters,
            "subtitle_dialogue_excerpt": subtitle_dialogue_excerpt,
        }
        enriched = self._attach_screenplay_payload(
            movie_id, enriched, script_ref=dominant or None
        )
        resolved_vlm, vision_source = self._resolve_vlm_payload(
            movie_id,
            enriched,
            dominant or None,
            vlm_scene,
        )
        primary_dialogue, dialogue_source = self._select_primary_dialogue(
            enriched,
            subtitle_dialogue_excerpt,
        )
        enriched.update(
            {
                "dialogue_excerpt": primary_dialogue,
                "dialogue_source": dialogue_source,
                "vision_setting": self._clean_text(resolved_vlm.get("setting", "")),
                "vision_actions": self._clean_text(resolved_vlm.get("actions", "")),
                "vision_objects": list(resolved_vlm.get("visual_objects", []) or []),
                "visual_focus": self._clean_text(resolved_vlm.get("visual_focus", "")),
                "vision_source": vision_source,
            }
        )
        enriched["description"] = self._compose_chunk_description(
            clip=enriched,
            script_ref=dominant or None,
            vlm_scene=resolved_vlm,
            dialogue_excerpt=primary_dialogue,
            characters=characters,
        )
        return enriched

    # ═══════════════════════════════════════════════════════════
    # Build chunks for a movie
    # ═══════════════════════════════════════════════════════════

    def build_for_movie(
        self,
        movie_id: str,
        unified_data: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Build all temporal chunks for a single movie.

        Works in TWO modes:
          A) If annotation + MovieGraphs data exists → clip-centric with scene anchoring
          B) If only keyframe_index.json exists (new video) → scene-centric from keyframe index
        """
        logger.info(f"\n{'=' * 50}\n  Building chunks: {movie_id}\n{'=' * 50}")

        # Load all layers
        scenes = self._load_annotation_scenes(movie_id)
        clips = self._load_clips(movie_id, unified_data or {})
        meta = self._load_meta(movie_id)
        srt_entries = SubtitleParser.load_for_movie(movie_id)

        title = meta.get("title", movie_id)
        genres = meta.get("genres", [])
        cast_map = {
            c.get("name", ""): c.get("character", "") for c in meta.get("cast", [])
        }

        chunks = []

        if scenes and clips:
            # ── Mode A: Clip-centric with annotation scene timestamps ──
            chunks = self._build_clip_centric(
                movie_id, clips, scenes, srt_entries, title, genres, cast_map
            )
        elif scenes:
            # ── Mode A (scene-only): No MovieGraphs but has annotations ──
            chunks = self._build_scene_only(
                movie_id, scenes, srt_entries, title, genres
            )
        else:
            # ── Mode B: NEW VIDEO — no annotation, use keyframe_index.json ──
            kf_index = self._load_keyframe_index(movie_id)
            if kf_index:
                chunks = self._build_from_keyframe_index(
                    movie_id, kf_index, srt_entries, title, genres
                )
            else:
                logger.warning(f"  {movie_id}: No data sources found. Skipping.")

        # Stats
        logger.info(
            f"  [RESULT] {movie_id}: {len(chunks)} chunks, "
            f"{sum(1 for c in chunks if c['keyframe_paths'])} with keyframes, "
            f"{sum(1 for c in chunks if c['dialogue'])} with dialogue"
        )
        return chunks

    # ── Mode A: Clip-centric (annotated dataset) ──

    def _build_clip_centric(
        self, movie_id, clips, scenes, srt, title, genres, cast_map
    ):
        chunks = []
        vlm_scene_map = self._load_vlm_scenes(movie_id)
        chunk_idx = 0

        for clip in clips:
            clip_start = clip.get("start_seconds")
            clip_end = clip.get("end_seconds")
            if (
                clip_start is not None
                and clip_end is not None
                and float(clip_end) > float(clip_start)
            ):
                start_sec, end_sec = float(clip_start), float(clip_end)
                ts_source = clip.get("timestamp_source", "clip_explicit")
            else:
                matched = self._match_clip_to_scene(clip, scenes)
                if matched:
                    start_sec, end_sec = matched["start_seconds"], matched["end_seconds"]
                    ts_source = "annotation_frame"
                else:
                    start_sec, end_sec = 0, 0
                    ts_source = "none"

            scene_id = clip.get("scene_id", "")
            vlm_scene = vlm_scene_map.get(scene_id, {})
            script_segments = self._expand_clip_into_segments(
                movie_id=movie_id,
                clip=clip,
                start_sec=start_sec,
                end_sec=end_sec,
                cast_map=cast_map,
                vlm_scene=vlm_scene,
                srt_entries=srt,
            )

            if script_segments:
                for segment in script_segments:
                    seg_start = float(segment.get("start_seconds", start_sec))
                    seg_end = float(segment.get("end_seconds", end_sec))
                    keyframes = self._find_keyframes_by_time(movie_id, seg_start, seg_end)
                    if not keyframes:
                        keyframes = self._find_keyframes_by_time(movie_id, start_sec, end_sec)

                    chunks.append(
                        self._make_chunk(
                            movie_id,
                            chunk_idx,
                            title,
                            genres,
                            seg_start,
                            seg_end,
                            "script_window",
                            clip=segment.get("clip", clip),
                            dialogues=segment.get("dialogues", []),
                            keyframes=keyframes,
                            cast_map=cast_map,
                        )
                    )
                    chunk_idx += 1
                continue

            dialogues = SubtitleParser.align(srt, start_sec, end_sec) if start_sec or end_sec else []
            keyframes = self._find_keyframes_by_time(movie_id, start_sec, end_sec)
            enriched_clip = self._enrich_clip_for_chunk(
                movie_id, clip, cast_map, vlm_scene
            )

            chunks.append(
                self._make_chunk(
                    movie_id,
                    chunk_idx,
                    title,
                    genres,
                    start_sec,
                    end_sec,
                    ts_source,
                    clip=enriched_clip,
                    dialogues=dialogues,
                    keyframes=keyframes,
                    cast_map=cast_map,
                )
            )
            chunk_idx += 1
        return chunks

    # ── Mode A (scene-only) ──

    def _build_scene_only(self, movie_id, scenes, srt, title, genres):
        chunks = []
        for i, scene in enumerate(scenes):
            s, e = scene["start_seconds"], scene["end_seconds"]
            dialogues = SubtitleParser.align(srt, s, e)
            keyframes = self._find_keyframes_by_time(movie_id, s, e)

            chunks.append(
                self._make_chunk(
                    movie_id,
                    i,
                    title,
                    genres,
                    s,
                    e,
                    "annotation_frame",
                    clip={
                        "scene_type": scene.get("scene_type", ""),
                        "environment": scene.get("environment", ""),
                        "script_time_of_day": scene.get("script_time_of_day", ""),
                        "character_type": scene.get("character_type", ""),
                        "script_location": scene.get("script_location", ""),
                        "script_characters": scene.get("script_characters", []),
                        "script_scene_refs": scene.get("script_scene_refs", []),
                        "script_scene_count": scene.get("script_scene_count", 0),
                        "script_primary_heading": scene.get(
                            "script_primary_heading", ""
                        ),
                        "script_headings": scene.get("script_headings", []),
                        "dominant_script_scene_ref": scene.get(
                            "dominant_script_scene_ref"
                        ),
                        "dominant_script_overlap_sec": scene.get(
                            "dominant_script_overlap_sec", 0.0
                        ),
                    },
                    dialogues=dialogues,
                    keyframes=keyframes,
                    scene_label=scene.get("place_tag", ""),
                )
            )
        return chunks

    # ── Mode B: NEW VIDEO — from keyframe_index.json scenes ──

    def _build_from_keyframe_index(self, movie_id, kf_index, srt, title, genres):
        """Build chunks from keyframe_index.json for new videos with no annotation."""
        # Group keyframes by scene_idx
        scene_groups: Dict[int, List[Dict]] = {}
        for kf in kf_index:
            si = kf["scene_idx"]
            scene_groups.setdefault(si, []).append(kf)

        chunks = []
        for i, si in enumerate(sorted(scene_groups.keys())):
            kfs = scene_groups[si]
            start_sec = min(kf["timestamp_sec"] for kf in kfs)
            end_sec = max(kf["timestamp_sec"] for kf in kfs)
            # Expand range for single-keyframe scenes
            if end_sec == start_sec:
                end_sec = start_sec + Cfg.KEYFRAME_INTERVAL_SEC

            dialogues = SubtitleParser.align(srt, start_sec, end_sec)
            paths = [kf["path"] for kf in sorted(kfs, key=lambda x: x["img_idx"])]

            chunks.append(
                self._make_chunk(
                    movie_id,
                    i,
                    title,
                    genres,
                    start_sec,
                    end_sec,
                    "keyframe_index",
                    dialogues=dialogues,
                    keyframes=paths,
                )
            )
        return chunks

    # ── Chunk factory ──

    @staticmethod
    def _make_chunk(
        movie_id,
        idx,
        title,
        genres,
        start_sec,
        end_sec,
        ts_source,
        clip=None,
        dialogues=None,
        keyframes=None,
        cast_map=None,
        scene_label="",
    ) -> Dict:
        dialogues = ChunkBuilder._clean_dialogue_lines(dialogues or [])
        keyframes = keyframes or []
        clip = clip or {}
        cast_map = cast_map or {}

        def _fmt(s):
            h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"

        characters = ChunkBuilder._curate_characters(
            clip.get("characters", []),
            clip.get("script_characters", []),
            cast_map,
        )
        subtitle_dialogue_excerpt = ChunkBuilder._clean_text(
            clip.get("subtitle_dialogue_excerpt", ChunkBuilder._dialogue_excerpt(dialogues))
        )
        dialogue_text = ChunkBuilder._clean_text(
            clip.get("dialogue_excerpt", "")
        )[:900]
        if not dialogue_text:
            dialogue_text = ChunkBuilder._dialogue_excerpt(dialogues, max_chars=900)
        if not dialogue_text:
            dialogue_text = ChunkBuilder._clean_text(
                clip.get("screenplay_dialogue_excerpt", "")
            )[:900]

        return {
            "chunk_id": f"{movie_id}_chunk_{idx:04d}",
            "movie_id": movie_id,
            "title": title,
            "genres": genres,
            # Temporal
            "start_time": _fmt(start_sec),
            "end_time": _fmt(end_sec),
            "start_seconds": round(start_sec, 2),
            "end_seconds": round(end_sec, 2),
            "duration_seconds": round(end_sec - start_sec, 2),
            "timestamp_source": ts_source,
            # Semantic
            "clip_id": clip.get("clip_id", ""),
            "description": clip.get("description", ""),
            "situation": clip.get("situation", ""),
            "scene_label": clip.get("scene_label", scene_label),
            "chunk_source": clip.get("chunk_source", "semantic_scene"),
            "parent_scene_id": clip.get("parent_scene_id", clip.get("scene_id", "")),
            "parent_clip_id": clip.get("parent_clip_id", clip.get("clip_id", "")),
            "script_scene_uid": clip.get("script_scene_uid", ""),
            "characters": characters,
            "character_ids": clip.get("character_ids", []),
            "attributes": clip.get("attributes", []),
            "interactions": clip.get("interactions", []),
            "scene_type": clip.get("scene_type", ""),
            "environment": clip.get("environment", ""),
            "script_time_of_day": clip.get("script_time_of_day", ""),
            "character_type": clip.get("character_type", ""),
            "script_location": clip.get("script_location", ""),
            "script_characters": clip.get("script_characters", []),
            "script_scene_refs": clip.get("script_scene_refs", []),
            "script_scene_count": clip.get("script_scene_count", 0),
            "script_primary_heading": clip.get("script_primary_heading", ""),
            "script_headings": clip.get("script_headings", []),
            "dominant_script_scene_ref": clip.get("dominant_script_scene_ref"),
            "dominant_script_overlap_sec": clip.get(
                "dominant_script_overlap_sec", 0.0
            ),
            "alignment_confidence": clip.get(
                "alignment_confidence",
                float(
                    (clip.get("dominant_script_scene_ref") or {}).get(
                        "confidence_score", 0.0
                    )
                    or 0.0
                ),
            ),
            "dialogue_excerpt": clip.get(
                "dialogue_excerpt", ChunkBuilder._dialogue_excerpt(dialogues)
            ),
            "subtitle_dialogue_excerpt": subtitle_dialogue_excerpt,
            "dialogue_source": clip.get("dialogue_source", "subtitle"),
            "screenplay_action_excerpt": clip.get("screenplay_action_excerpt", ""),
            "screenplay_dialogue_turns": clip.get("screenplay_dialogue_turns", []),
            "screenplay_dialogue_excerpt": clip.get(
                "screenplay_dialogue_excerpt", ""
            ),
            "screenplay_context_excerpt": clip.get("screenplay_context_excerpt", ""),
            "screenplay_evidence": clip.get("screenplay_evidence", ""),
            "vision_setting": clip.get("vision_setting", ""),
            "vision_actions": clip.get("vision_actions", ""),
            "vision_objects": clip.get("vision_objects", []),
            "visual_focus": clip.get("visual_focus", ""),
            "vision_source": clip.get("vision_source", "none"),
            # Dialogue
            "dialogue": dialogues,
            "dialogue_text": dialogue_text,
            "dialogue_full_text": ChunkBuilder._resolve_dialogue_full_text(
                clip, dialogues
            ),
            # Shot range
            "shot_start": clip.get("start_shot", 0),
            "shot_end": clip.get("end_shot", 0),
            # Keyframes
            "keyframe_paths": keyframes,
            "num_keyframes": len(keyframes),
            # Cast mapping
            "cast_in_scene": ChunkBuilder._build_cast_in_scene(characters, cast_map),
        }

    # ── Helpers ──

    @staticmethod
    def _load_annotation_scenes(movie_id: str) -> List[Dict]:
        ann_path = Cfg.get_annotation_dir() / f"{movie_id}.json"
        if not ann_path.exists():
            return []
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        # Get FPS from keyframe_index or default
        fps = float(data.get("fps", 24.0))
        for d in Cfg.get_keyframe_search_dirs():
            for manifest_name in (
                "vector_clean_index.json",
                "keyframe_index.json",
                "vlm_quality_index.json",
            ):
                idx_path = d / movie_id / manifest_name
                if not idx_path.exists():
                    continue
                try:
                    idx = json.loads(idx_path.read_text(encoding="utf-8"))
                    fps = float(idx.get("video_fps", fps))
                except Exception:
                    pass
                break

        scenes = []
        for i, s in enumerate(data.get("scene", [])):
            fr = s.get("frame", [0, 0])
            sh = s.get("shot", [0, 0])
            if len(fr) < 2 or len(sh) < 2:
                continue
            ss, es = fr[0] / fps, fr[1] / fps
            scenes.append(
                {
                    "scene_idx": i,
                    "shot_start": sh[0],
                    "shot_end": sh[1],
                    "start_seconds": round(ss, 2),
                    "end_seconds": round(es, 2),
                    "duration_seconds": round(es - ss, 2),
                    "place_tag": s.get("place_tag"),
                    "action_tag": s.get("action_tag"),
                }
            )
        return scenes

    @staticmethod
    def _load_clips(movie_id: str, unified_data: Dict) -> List[Dict]:
        def get_char_attr(ch, attr):
            if isinstance(ch, dict):
                return ch.get(attr, "")
            elif isinstance(ch, str) and attr == "name":
                return ch
            return ""

        def normalize_clips(raw_clips: List[Dict]) -> List[Dict]:
            return [
                {
                    "clip_id": c.get("clip_id", ""),
                    "start_shot": c.get("start_shot", 0),
                    "end_shot": c.get("end_shot", 0),
                    "start_seconds": c.get("start_seconds"),
                    "end_seconds": c.get("end_seconds"),
                    "start_time": c.get("start_time"),
                    "end_time": c.get("end_time"),
                    "scene_id": c.get("scene_id", ""),
                    "annotation_frame": c.get("annotation_frame", []),
                    "scene_type": c.get("scene_type", ""),
                    "environment": c.get("environment", ""),
                    "script_time_of_day": c.get("script_time_of_day", ""),
                    "character_type": c.get("character_type", ""),
                    "script_location": c.get("script_location", ""),
                    "script_characters": c.get("script_characters", []),
                    "script_scene_refs": c.get("script_scene_refs", []),
                    "script_scene_count": c.get("script_scene_count", 0),
                    "script_primary_heading": c.get("script_primary_heading", ""),
                    "script_headings": c.get("script_headings", []),
                    "dominant_script_scene_ref": c.get("dominant_script_scene_ref"),
                    "dominant_script_overlap_sec": c.get(
                        "dominant_script_overlap_sec", 0.0
                    ),
                    "description": c.get("description", ""),
                    "situation": c.get("situation", ""),
                    "scene_label": c.get("scene_label", ""),
                    "characters": [
                        get_char_attr(ch, "name") for ch in c.get("characters", [])
                    ],
                    "character_ids": [
                        get_char_attr(ch, "id") for ch in c.get("characters", [])
                    ],
                    "attributes": list(set(c.get("attributes", []))),
                    "interactions": c.get("interactions", []),
                }
                for c in raw_clips or []
            ]

        movie = unified_data.get("movies", {}).get(movie_id)
        if movie:
            normalized = normalize_clips(movie.get("clips", []))
            if normalized:
                return normalized

        graph_path = Cfg.get_scene_graph_dir() / movie_id / f"{movie_id}_auto_graph.json"
        if graph_path.exists():
            try:
                graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
                normalized = normalize_clips(graph_data.get("clips", []))
                if normalized:
                    return normalized
            except Exception as exc:
                logger.warning("Failed to read auto graph clips for %s: %s", movie_id, exc)

        return []

    @staticmethod
    def _load_meta(movie_id: str) -> Dict:
        for d in Cfg.META_SEARCH_DIRS:
            p = d / f"{movie_id}.json"
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {}

    @staticmethod
    def _match_clip_to_scene(clip, scenes):
        clip_start, clip_end = clip.get("start_shot", -1), clip.get("end_shot", -1)
        
        # 1. Strict overlap match (for MovieGraphs clips)
        best, best_overlap = None, 0
        for scene in scenes:
            overlap = max(
                0,
                min(clip_end, scene["shot_end"]) - max(clip_start, scene["shot_start"]),
            )
            if overlap > best_overlap:
                best_overlap, best = overlap, scene
        
        # 2. Contains match
        if not best:
            for scene in scenes:
                if scene["shot_start"] <= clip_start < scene["shot_end"]:
                    return scene
                    
        # 3. Direct scene_idx match (for Auto-generated clips where start_shot == scene_idx)
        if not best:
            for scene in scenes:
                if scene.get("scene_idx") == clip_start:
                    return scene
        
        return best

    # ═══════════════════════════════════════════════════════════
    # Batch build + save
    # ═══════════════════════════════════════════════════════════

    def build_all(self, movie_ids: List[str] = None) -> Dict:
        """Build chunks for all movies and save to disk."""
        # Load unified dataset
        unified_data = {}
        if Cfg.UNIFIED_DATASET_JSON.exists():
            unified_data = json.loads(
                Cfg.UNIFIED_DATASET_JSON.read_text(encoding="utf-8")
            )

        if movie_ids is None:
            movie_ids = Cfg.get_all_movie_ids()

        Cfg.get_temporal_chunks_dir().mkdir(parents=True, exist_ok=True)
        all_chunks = []
        stats = {"movies": 0, "chunks": 0}

        # Collect all chunks first to get movie_chunks for logging
        movie_chunks = {}
        for mid in movie_ids:
            chunks = self.build_for_movie(mid, unified_data)
            if chunks:
                movie_chunks[mid] = chunks

        for i, (mid, chunks) in enumerate(movie_chunks.items(), 1):
            logger.info(f"    [{i}/{len(movie_chunks)}] {mid}: {len(chunks)} chunks")
            if mid:
                out = Cfg.get_temporal_chunks_dir() / f"{mid}_chunks.json"
                out.write_text(
                    json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                all_chunks.extend(chunks)
                stats["movies"] += 1
                stats["chunks"] += len(chunks)

        # Save merged
        merged = {
            "metadata": {
                "total_movies": stats["movies"],
                "total_chunks": stats["chunks"],
            },
            "chunks": all_chunks,
        }
        merged_path = Cfg.get_temporal_chunks_dir() / "all_chunks.json"
        merged_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(
            f"\n  ✅ Built {stats['chunks']} chunks for {stats['movies']} movies → {merged_path}"
        )
        return merged
