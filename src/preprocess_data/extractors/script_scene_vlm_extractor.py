"""
Script-scene visual extractor.

Builds visual descriptions at screenplay-aligned scene granularity instead of
reusing one semantic-scene VLM summary for all child chunks.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from preprocess_data.config import PreprocessConfig as Cfg
from preprocess_data.extractors._keyframe_manifest import (
    load_keyframe_entries,
    normalize_keyframe_entry,
)
from preprocess_data.extractors.script_aligner import ScriptAligner
from preprocess_data.indexing._screenplay_evidence import build_screenplay_payload
from movierag.generation.universal_client import (
    LLMRateLimitError,
    UniversalLLMClient,
    is_rate_limit_error,
)

logger = logging.getLogger(__name__)


SCRIPT_SCENE_VLM_PROMPT = """You are analyzing chronologically ordered movie frames from one screenplay-aligned scene.
Use the screenplay heading and excerpt only as a locator hint. If the hint conflicts with the frames, trust the frames.
Describe only what is visually supported.

Return EXACTLY valid JSON with this schema:
{
  "setting": "Detailed visual setting, lighting, atmosphere, and spatial layout",
  "actions": "Chronological physical actions and interactions visible in the frames",
  "visual_objects": ["object 1", "object 2"],
  "visual_focus": "Short phrase naming the dominant visual moment"
}
Return ONLY JSON.
"""


class ScriptSceneVLMExtractor:
    def __init__(self):
        self.vlm = UniversalLLMClient()
        self._aligner = ScriptAligner()

    def process_movie(self, movie_id: str, force: bool = False) -> bool:
        logger.info("\n[6aa/8] Script-scene VLM extraction for %s...", movie_id)

        movie_dir = Cfg.get_shot_keyf_dir() / movie_id
        out_path = movie_dir / "vlm_script_scene_descriptions.json"
        existing_output = self._load_existing_output(out_path) if out_path.exists() else {}
        results = {}
        if existing_output and not force:
            if existing_output.get("status") == "complete":
                logger.info("  ⏩ Script-scene VLM already exists. Use force=True to overwrite.")
                return True
            results = dict(existing_output.get("scenes", {}))

        aligned_scenes = self._aligner.align(movie_id, force=False)
        if not aligned_scenes:
            logger.warning("  No aligned script scenes found for %s.", movie_id)
            self._save_results(out_path, movie_id, 0, {}, "complete")
            return False

        index_path, keyframes = self._load_all_keyframes(movie_dir)
        if index_path is None or not keyframes:
            logger.warning("  No keyframe manifest found for script-scene VLM on %s.", movie_id)
            self._save_results(out_path, movie_id, len(aligned_scenes), {}, "complete")
            return False

        logger.info(
            "  Processing %s aligned script scenes with %s keyframes",
            len(aligned_scenes),
            len(keyframes),
        )

        for idx, scene in enumerate(aligned_scenes):
            scene_uid = str(getattr(scene, "scene_uid", "") or "").strip()
            if not scene_uid:
                continue
            if scene_uid in results and not force:
                continue

            selected = self._select_keyframes_for_scene(movie_id, keyframes, scene.start_sec, scene.end_sec)
            if len(selected) < max(1, Cfg.SCRIPT_SCENE_VLM_MIN_FRAMES):
                continue

            prompt = self._build_prompt(scene)
            images_base64 = self._encode_images(selected)
            if not images_base64:
                continue

            logger.info(
                "    -> Script scene %s/%s: %s (%s frames)",
                idx + 1,
                len(aligned_scenes),
                scene_uid,
                len(images_base64),
            )

            try:
                response_text = self.vlm.generate_multi_vision(
                    prompt=prompt,
                    images_base64=images_base64,
                    temperature=0.1,
                    max_tokens=Cfg.VLM_MAX_COMPLETION_TOKENS or None,
                    max_completion_tokens=Cfg.VLM_MAX_COMPLETION_TOKENS or None,
                )
                visual_data = self._parse_json(response_text)
                visual_data.update(
                    {
                        "script_scene_uid": scene_uid,
                        "heading": scene.heading,
                        "location": scene.location,
                        "time_of_day": scene.time_of_day,
                        "frame_count": len(images_base64),
                        "start_seconds": round(float(scene.start_sec or 0.0), 2),
                        "end_seconds": round(float(scene.end_sec or 0.0), 2),
                    }
                )
                results[scene_uid] = visual_data
                self._save_results(
                    out_path,
                    movie_id=movie_id,
                    total_scenes=len(aligned_scenes),
                    results=results,
                    status="partial",
                )
            except Exception as exc:
                if is_rate_limit_error(exc):
                    self._save_results(
                        out_path,
                        movie_id=movie_id,
                        total_scenes=len(aligned_scenes),
                        results=results,
                        status="partial",
                        error_message=str(exc),
                    )
                    raise LLMRateLimitError(
                        f"Script-scene VLM hit rate limit for {movie_id}: {exc}"
                    ) from exc
                logger.error("      ❌ Script-scene VLM failed for %s: %s", scene_uid, exc)

        self._save_results(
            out_path,
            movie_id=movie_id,
            total_scenes=len(aligned_scenes),
            results=results,
            status="complete",
        )
        logger.info("  ✅ Saved script-scene VLM descriptions to %s", out_path.name)
        return True

    @staticmethod
    def _load_existing_output(out_path: Path) -> Dict[str, Any]:
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _save_results(
        out_path: Path,
        movie_id: str,
        total_scenes: int,
        results: Dict[str, Any],
        status: str,
        error_message: str = "",
    ) -> None:
        output = {
            "movie_id": movie_id,
            "total_scenes": total_scenes,
            "completed_scenes": len(results),
            "status": status,
            "last_error": error_message,
            "scenes": results,
        }
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _parse_json(response_text: str) -> Dict[str, Any]:
        if not response_text:
            return {"setting": "", "actions": "", "visual_objects": [], "visual_focus": ""}

        json_str = response_text
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                json_str = match.group(0)

        payload = json.loads(json_str)
        if not isinstance(payload, dict):
            return {"setting": "", "actions": "", "visual_objects": [], "visual_focus": ""}
        payload.setdefault("setting", "")
        payload.setdefault("actions", "")
        payload.setdefault("visual_objects", [])
        payload.setdefault("visual_focus", "")
        return payload

    @staticmethod
    def _build_prompt(scene) -> str:
        screenplay_payload = build_screenplay_payload(scene)
        context_excerpt = screenplay_payload.get("screenplay_context_excerpt", "")
        characters = ", ".join(getattr(scene, "characters", []) or [])
        lines = [
            SCRIPT_SCENE_VLM_PROMPT,
            "",
            "Screenplay hint:",
            f"Heading: {getattr(scene, 'heading', '')}",
            f"Location: {getattr(scene, 'location', '')}",
            f"Time of day: {getattr(scene, 'time_of_day', '')}",
        ]
        if characters:
            lines.append(f"Characters listed in script: {characters}")
        if context_excerpt:
            lines.append(f"Screenplay excerpt: {context_excerpt}")
        return "\n".join(lines).strip()

    @staticmethod
    def _encode_images(paths: List[str]) -> List[str]:
        encoded: List[str] = []
        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                continue
            try:
                encoded.append(
                    f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"
                )
            except Exception:
                continue
        return encoded

    @staticmethod
    def _load_all_keyframes(movie_dir: Path) -> tuple[Path | None, List[Dict[str, Any]]]:
        primary_path, primary_entries = load_keyframe_entries(
            movie_dir,
            preferred_names=["vector_clean_index.json", "vlm_quality_index.json", "keyframe_index.json"],
        )
        merged: List[Dict[str, Any]] = []
        seen_paths = set()

        for name in ("vector_clean_index.json", "vlm_quality_index.json", "keyframe_index.json"):
            manifest_path = movie_dir / name
            if not manifest_path.exists():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for raw in payload.get("keyframes", []) or []:
                entry = normalize_keyframe_entry(raw)
                key = (
                    str(entry.get("path", "") or ""),
                    round(float(entry.get("timestamp_sec", 0.0) or 0.0), 3),
                )
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                merged.append(entry)

        if not merged and primary_entries:
            merged = list(primary_entries)

        merged = sorted(
            merged,
            key=lambda item: (
                float(item.get("timestamp_sec", 0.0)),
                int(item.get("shot_id", 0) or 0),
                int(item.get("img_idx", 0) or 0),
            ),
        )
        return primary_path, merged

    @staticmethod
    def _select_keyframes_for_scene(
        movie_id: str,
        keyframes: List[Dict[str, Any]],
        start_sec: float,
        end_sec: float,
    ) -> List[str]:
        selected = [
            entry
            for entry in keyframes
            if float(entry.get("timestamp_sec", 0.0)) >= float(start_sec)
            and float(entry.get("timestamp_sec", 0.0)) <= float(end_sec)
        ]
        if not selected and keyframes:
            center = (float(start_sec) + float(end_sec)) / 2.0
            nearby = sorted(
                keyframes,
                key=lambda item: abs(float(item.get("timestamp_sec", 0.0)) - center),
            )
            selected = [
                item
                for item in nearby[: max(3, int(Cfg.SCRIPT_SCENE_VLM_MIN_FRAMES or 1))]
                if abs(float(item.get("timestamp_sec", 0.0)) - center) <= 12.0
            ]
        selected = sorted(
            selected,
            key=lambda item: (
                float(item.get("timestamp_sec", 0.0)),
                -float((item.get("quality") or {}).get("composite", 0.0) or 0.0),
            ),
        )
        if not selected:
            return []

        max_images = max(1, int(Cfg.SCRIPT_SCENE_VLM_MAX_IMAGES or 1))
        if len(selected) > max_images:
            step = len(selected) / float(max_images)
            selected = [selected[min(int(i * step), len(selected) - 1)] for i in range(max_images)]

        paths: List[str] = []
        for item in selected:
            resolved = Cfg.resolve_keyframe_path(movie_id, str(item.get("path", "")))
            if resolved and resolved not in paths:
                paths.append(resolved)
        return paths
