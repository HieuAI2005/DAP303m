"""
Build script-aware sub-scenes from semantic scenes + screenplay alignment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from preprocess_data.config import PreprocessConfig as Cfg
from preprocess_data.extractors.script_aligner import ScriptAligner, ScriptScene
from preprocess_data.indexing.subtitle_parser import SubtitleParser
from preprocess_data.indexing._screenplay_evidence import build_screenplay_payload

logger = logging.getLogger(__name__)


class ScriptSubsceneBuilder:
    """Derive indexable screenplay sub-scenes under each semantic scene."""

    def __init__(self):
        self._aligner = ScriptAligner()

    def build_for_movie(self, movie_id: str) -> List[Dict]:
        annotation_scenes = self._load_annotation_scenes(movie_id)
        if not annotation_scenes:
            self._save(movie_id, [])
            return []

        aligned_script_scenes = self._aligner.align(movie_id, force=False)
        if not aligned_script_scenes:
            self._save(movie_id, [])
            return []
        self._ensure_script_scene_uids(movie_id, aligned_script_scenes)

        clip_by_scene = self._load_clip_by_scene(movie_id)
        chunks = self._load_chunks(movie_id)
        subtitles = SubtitleParser.load_for_movie(movie_id)

        subscenes: List[Dict] = []
        for idx, scene in enumerate(annotation_scenes):
            scene_id = scene.get("id") or f"scene_{idx}"
            scene_start = float(scene.get("start_seconds", 0.0))
            scene_end = float(scene.get("end_seconds", scene_start))
            scene_duration = max(scene_end - scene_start, 1.0)

            parent_clip = clip_by_scene.get(scene_id, {})
            overlaps = self._get_overlapping_script_scenes(
                aligned_script_scenes, scene_start, scene_end
            )
            for script_scene in overlaps:
                overlap_start = max(scene_start, float(script_scene.start_sec))
                overlap_end = min(scene_end, float(script_scene.end_sec))
                overlap_seconds = max(0.0, overlap_end - overlap_start)
                parent_chunk = self._find_parent_chunk(
                    chunks=chunks,
                    parent_clip_id=parent_clip.get("clip_id", ""),
                    script_scene_uid=getattr(script_scene, "scene_uid", ""),
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                )
                index_eligible = (
                    overlap_seconds >= 5.0
                    or script_scene.anchor_quality in {"full", "partial"}
                )

                script_duration = max(
                    float(script_scene.end_sec) - float(script_scene.start_sec), 1.0
                )
                dialogue_excerpt = self._build_dialogue_excerpt(
                    subtitles, overlap_start, overlap_end
                )
                screenplay_payload = build_screenplay_payload(script_scene)

                subscene_id = self._make_subscene_id(
                    movie_id=movie_id,
                    parent_scene_id=scene_id,
                    script_scene=script_scene,
                    overlap_start=overlap_start,
                )

                subscene = {
                    "subscene_id": subscene_id,
                    "movie_id": movie_id,
                    "script_scene_uid": getattr(script_scene, "scene_uid", ""),
                    "parent_scene_id": scene_id,
                    "parent_chunk_id": parent_chunk.get("chunk_id", ""),
                    "parent_clip_id": parent_clip.get("clip_id", ""),
                    "start_seconds": round(overlap_start, 2),
                    "end_seconds": round(overlap_end, 2),
                    "start_time": self._fmt_hms(overlap_start),
                    "end_time": self._fmt_hms(overlap_end),
                    "script_heading": script_scene.heading,
                    "script_location": script_scene.location,
                    "script_time_of_day": script_scene.time_of_day,
                    "script_characters": list(script_scene.characters),
                    "anchor_quality": script_scene.anchor_quality,
                    "confidence_score": round(
                        float(script_scene.confidence_score or 0.0), 3
                    ),
                    "alignment_confidence": round(
                        float(script_scene.confidence_score or 0.0), 3
                    ),
                    "overlap_seconds": round(overlap_seconds, 2),
                    "overlap_ratio_semantic": round(overlap_seconds / scene_duration, 3),
                    "overlap_ratio_script": round(overlap_seconds / script_duration, 3),
                    "semantic_description": parent_clip.get(
                        "description", scene.get("description", scene.get("reason", ""))
                    ),
                    "semantic_scene_label": parent_clip.get(
                        "scene_label", scene.get("place_tag", "")
                    ),
                    "dialogue_excerpt": dialogue_excerpt,
                    "index_eligible": index_eligible,
                    "indexable": False,
                    "is_canonical_subscene": False,
                }
                subscene.update(screenplay_payload)
                subscenes.append(subscene)

        self._assign_canonical_indexability(subscenes)
        subscenes.sort(
            key=lambda item: (
                float(item.get("start_seconds", 0.0)),
                item.get("parent_scene_id", ""),
                item.get("script_heading", ""),
            )
        )
        self._save(movie_id, subscenes)
        logger.info(
            "  Built %s script sub-scenes for %s", len(subscenes), movie_id
        )
        return subscenes

    def _load_annotation_scenes(self, movie_id: str) -> List[Dict]:
        path = Cfg.get_annotation_dir() / f"{movie_id}.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("scene", [])
        except Exception as exc:
            logger.warning(f"  Failed to read annotation scenes for {movie_id}: {exc}")
            return []

    def _load_clip_by_scene(self, movie_id: str) -> Dict[str, Dict]:
        path = Cfg.get_scene_graph_dir() / movie_id / f"{movie_id}_auto_graph.json"
        if not path.exists():
            return {}
        try:
            clips = json.loads(path.read_text(encoding="utf-8")).get("clips", [])
        except Exception as exc:
            logger.warning(f"  Failed to read auto graph for {movie_id}: {exc}")
            return {}
        return {clip.get("scene_id", ""): clip for clip in clips if clip.get("scene_id")}

    def _load_chunks(self, movie_id: str) -> List[Dict]:
        path = Cfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"  Failed to read chunks for {movie_id}: {exc}")
            return []

    @staticmethod
    def _find_parent_chunk(
        chunks: List[Dict],
        parent_clip_id: str,
        script_scene_uid: str,
        overlap_start: float,
        overlap_end: float,
    ) -> Dict:
        candidates = [
            chunk
            for chunk in chunks
            if chunk.get("clip_id", "") == parent_clip_id
        ]
        if not candidates:
            return {}

        exact_script_match = [
            chunk
            for chunk in candidates
            if chunk.get("script_scene_uid", "") == script_scene_uid
        ]
        if exact_script_match:
            candidates = exact_script_match

        def sort_key(chunk: Dict):
            start = float(chunk.get("start_seconds", 0.0))
            end = float(chunk.get("end_seconds", start))
            overlap = max(0.0, min(end, overlap_end) - max(start, overlap_start))
            return (
                overlap,
                -abs(start - overlap_start),
                -abs(end - overlap_end),
            )

        return max(candidates, key=sort_key, default={})

    @staticmethod
    def _get_overlapping_script_scenes(
        aligned_script_scenes: List[ScriptScene], start_sec: float, end_sec: float
    ) -> List[ScriptScene]:
        overlaps = [
            scene
            for scene in aligned_script_scenes
            if scene.end_sec > start_sec and scene.start_sec < end_sec
        ]
        if overlaps:
            return sorted(overlaps, key=lambda scene: (scene.start_sec, scene.end_sec))

        nearest = min(
            aligned_script_scenes,
            key=lambda scene: abs(scene.start_sec - start_sec),
            default=None,
        )
        return [nearest] if nearest else []

    @staticmethod
    def _build_dialogue_excerpt(
        subtitles: List[Dict], overlap_start: float, overlap_end: float
    ) -> str:
        if overlap_end <= overlap_start:
            return ""
        dialogue_lines = SubtitleParser.align(subtitles, overlap_start, overlap_end)
        if not dialogue_lines:
            return ""
        excerpt = " ".join(dialogue_lines).strip()
        return excerpt[:900]

    @staticmethod
    def _clean_text(text: str) -> str:
        import re

        return re.sub(r"\s+", " ", str(text or "")).strip()

    @classmethod
    def _build_action_excerpt(cls, script_scene: ScriptScene, max_chars: int = 420) -> str:
        parts: List[str] = []
        total = 0
        for line in getattr(script_scene, "action_lines", []) or []:
            cleaned = cls._clean_text(line)
            if not cleaned:
                continue
            if total + len(cleaned) + 1 > max_chars:
                break
            parts.append(cleaned)
            total += len(cleaned) + 1
        return " ".join(parts)

    @classmethod
    def _build_dialogue_turns(
        cls, script_scene: ScriptScene, max_turns: int = 8, max_chars: int = 520
    ) -> List[str]:
        turns: List[str] = []
        total = 0
        for turn in getattr(script_scene, "dialogue_lines", []) or []:
            speaker = cls._clean_text(turn.get("char", ""))
            text = cls._clean_text(turn.get("text", ""))
            if not text:
                continue
            formatted = f"{speaker}: {text}" if speaker else text
            if total + len(formatted) + 1 > max_chars:
                break
            turns.append(formatted)
            total += len(formatted) + 1
            if len(turns) >= max_turns:
                break
        return turns

    @classmethod
    def _build_screenplay_dialogue_excerpt(
        cls, dialogue_turns: List[str], max_chars: int = 420
    ) -> str:
        excerpt_parts: List[str] = []
        total = 0
        for turn in dialogue_turns:
            if total + len(turn) + 1 > max_chars:
                break
            excerpt_parts.append(turn)
            total += len(turn) + 1
        return " ".join(excerpt_parts)

    @classmethod
    def _build_screenplay_evidence(
        cls,
        script_scene: ScriptScene,
        action_excerpt: str,
        dialogue_turns: List[str],
    ) -> str:
        parts = [f"Heading: {cls._clean_text(script_scene.heading)}"]
        if action_excerpt:
            parts.append(f"Action: {action_excerpt}")
        if dialogue_turns:
            parts.append(f"Dialogue turns: {' | '.join(dialogue_turns)}")
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _ensure_script_scene_uids(
        movie_id: str, aligned_script_scenes: List[ScriptScene]
    ) -> None:
        for idx, script_scene in enumerate(aligned_script_scenes):
            if getattr(script_scene, "scene_uid", ""):
                continue
            script_scene.scene_uid = f"{movie_id}_script_scene_{idx:04d}"

    @staticmethod
    def _assign_canonical_indexability(subscenes: List[Dict]) -> None:
        quality_rank = {"full": 2, "partial": 1, "linear": 0}
        groups: Dict[str, List[Dict]] = {}
        for subscene in subscenes:
            scene_uid = str(subscene.get("script_scene_uid", "")).strip()
            if not scene_uid:
                scene_uid = subscene.get("subscene_id", "")
                subscene["script_scene_uid"] = scene_uid
            groups.setdefault(scene_uid, []).append(subscene)

        for group in groups.values():
            eligible = [item for item in group if item.get("index_eligible")]
            target_pool = eligible or group
            canonical = max(
                target_pool,
                key=lambda item: (
                    float(item.get("overlap_seconds", 0.0)),
                    quality_rank.get(str(item.get("anchor_quality", "linear")), 0),
                    float(
                        item.get(
                            "alignment_confidence",
                            item.get("confidence_score", 0.0),
                        )
                        or 0.0
                    ),
                    float(item.get("overlap_ratio_script", 0.0)),
                    -float(item.get("start_seconds", 0.0)),
                ),
            )
            split_count = len(group)
            for item in group:
                item["script_scene_split_count"] = split_count
                item["is_canonical_subscene"] = item is canonical
                item["indexable"] = bool(item is canonical and item.get("index_eligible"))

    @staticmethod
    def _script_scene_token(script_scene: ScriptScene) -> str:
        scene_uid = str(getattr(script_scene, "scene_uid", "") or "").strip()
        if scene_uid:
            return scene_uid.rsplit("_", 1)[-1]
        return "sxxx"

    @staticmethod
    def _make_subscene_id(
        movie_id: str,
        parent_scene_id: str,
        script_scene: ScriptScene,
        overlap_start: float,
    ) -> str:
        scene_num = script_scene.scene_num
        if isinstance(scene_num, int):
            scene_label = f"s{scene_num:03d}"
        else:
            scene_label = ScriptSubsceneBuilder._script_scene_token(script_scene)
        return (
            f"{movie_id}__{parent_scene_id}__{scene_label}__"
            f"{int(max(overlap_start, 0.0) * 100):08d}"
        )

    @staticmethod
    def _fmt_hms(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _save(self, movie_id: str, subscenes: List[Dict]) -> Path:
        out_path = Cfg.get_script_subscenes_dir() / f"{movie_id}_script_subscenes.json"
        out_path.write_text(
            json.dumps(subscenes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path
