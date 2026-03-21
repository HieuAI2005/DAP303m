"""
Batch ingest runner with resume support and rate-limit aware checkpointing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PreprocessConfig as Cfg
from .pipeline import PipelineRunner

logger = logging.getLogger(__name__)


class BatchIngestRunner:
    """Run the preprocess pipeline across many movies with persistent state."""

    MOVIE_ID_PATTERN = re.compile(r"^tt\d+$")

    def __init__(
        self,
        output_dir: str | Path,
        movie_ids: Optional[List[str]] = None,
        force: bool = False,
        limit: Optional[int] = None,
        stop_on_rate_limit: bool = True,
        cleanup_transient_artifacts: bool | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        Cfg.set_output_dir(self.output_dir)
        Cfg.ensure_dirs()

        self.force = force
        self.stop_on_rate_limit = stop_on_rate_limit
        self.cleanup_transient_artifacts = (
            cleanup_transient_artifacts
            if cleanup_transient_artifacts is not None
            else os.getenv("MOVIERAG_CLEANUP_TRANSIENT_ARTIFACTS", "1").lower()
            in {"1", "true", "yes", "on"}
        )
        discovered_ids = movie_ids or self.discover_movie_ids()
        self.movie_ids = discovered_ids[:limit] if limit else discovered_ids
        self.manifest_path = Cfg.get_batch_state_dir() / "batch_manifest.json"

    def discover_movie_ids(self) -> List[str]:
        movie_ids = set()
        for video_path in Cfg.RAW_VIDEOS_DIR.glob("*.*"):
            if not video_path.is_file():
                continue
            if not self.MOVIE_ID_PATTERN.match(video_path.stem):
                continue
            movie_ids.add(video_path.stem)
        return sorted(movie_ids)

    def run(self) -> Dict[str, Any]:
        manifest = self._load_manifest()
        manifest["output_root"] = str(self.output_dir)
        manifest["movie_queue"] = list(self.movie_ids)
        manifest["runtime_config"] = self._capture_runtime_config()
        manifest["last_run_started_at"] = self._now_iso()
        manifest["stopped_due_to_rate_limit"] = False
        self._refresh_summary(manifest)
        self._save_manifest(manifest)

        for index, movie_id in enumerate(self.movie_ids, start=1):
            entry = manifest.setdefault("movies", {}).setdefault(movie_id, {})
            video_path = Cfg.get_video_path(movie_id)
            if not video_path:
                entry.update(
                    {
                        "status": "skipped",
                        "last_error": "video_not_found",
                        "video_path": "",
                        "finished_at": self._now_iso(),
                    }
                )
                self._refresh_summary(manifest)
                self._save_manifest(manifest)
                continue

            current_artifacts = self._collect_artifacts(movie_id)
            if not self.force and self._artifacts_look_complete(current_artifacts):
                entry.update(
                    {
                        "movie_id": movie_id,
                        "status": "completed",
                        "video_path": str(video_path),
                        "source_srt": str(self._resolve_source_srt(movie_id) or ""),
                        "finished_at": self._now_iso(),
                        "last_error": "",
                        "failed_step": "",
                        "artifacts": current_artifacts,
                    }
                )
                logger.info(
                    "[%s/%s] Skipping already-complete movie %s",
                    index,
                    len(self.movie_ids),
                    movie_id,
                )
                self._refresh_summary(manifest)
                self._save_manifest(manifest)
                self._cleanup_movie_workspace(movie_id)
                continue

            if entry.get("status") == "completed" and not self.force:
                logger.info("[%s/%s] Skipping completed movie %s", index, len(self.movie_ids), movie_id)
                self._cleanup_movie_workspace(movie_id)
                continue

            source_srt = self._resolve_source_srt(movie_id)
            entry.update(
                {
                    "movie_id": movie_id,
                    "status": "running",
                    "queue_index": index,
                    "video_path": str(video_path),
                    "source_srt": str(source_srt) if source_srt else "",
                    "attempt_count": int(entry.get("attempt_count", 0)) + 1,
                    "started_at": self._now_iso(),
                    "finished_at": "",
                    "last_error": "",
                    "failed_step": "",
                    "completed_steps": [],
                }
            )
            self._refresh_summary(manifest)
            self._save_manifest(manifest)

            logger.info("[%s/%s] Running pipeline for %s", index, len(self.movie_ids), movie_id)
            runner = PipelineRunner(
                movie_id=movie_id,
                video_path=video_path,
                srt_path=source_srt,
                force=self.force,
            )
            success = runner.run_all()

            entry.update(
                {
                    "status": "completed" if success else "failed",
                    "finished_at": self._now_iso(),
                    "completed_steps": list(runner.completed_steps),
                    "failed_step": runner.failed_step,
                    "last_error": runner.last_error,
                    "artifacts": self._collect_artifacts(movie_id),
                }
            )
            if success and not self._artifacts_look_complete(entry["artifacts"]):
                entry["status"] = "incomplete"
                entry["last_error"] = entry.get("last_error") or "critical_artifacts_missing"

            if not success and runner.is_rate_limited():
                entry["status"] = "rate_limited"
                manifest["stopped_due_to_rate_limit"] = True
                manifest["rate_limited_movie_id"] = movie_id
                self._refresh_summary(manifest)
                self._save_manifest(manifest)
                if self.stop_on_rate_limit:
                    logger.warning(
                        "Stopping batch run after rate limit on %s. State saved at %s",
                        movie_id,
                        self.manifest_path,
                    )
                    break

            self._refresh_summary(manifest)
            self._save_manifest(manifest)
            cleanup_info = self._cleanup_movie_workspace(movie_id)
            if cleanup_info:
                entry["transient_cleanup"] = cleanup_info
                self._save_manifest(manifest)

        manifest["last_run_finished_at"] = self._now_iso()
        self._refresh_summary(manifest)
        self._save_manifest(manifest)
        return manifest

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load batch manifest, recreating: %s", exc)
        return {
            "output_root": str(self.output_dir),
            "movies": {},
            "movie_queue": [],
            "runtime_config": {},
            "summary": {},
            "stopped_due_to_rate_limit": False,
        }

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _refresh_summary(self, manifest: Dict[str, Any]) -> None:
        movies = manifest.get("movies", {}).values()
        summary = {
            "total_in_queue": len(manifest.get("movie_queue", [])),
            "completed": 0,
            "failed": 0,
            "rate_limited": 0,
            "running": 0,
            "skipped": 0,
            "pending": 0,
        }
        for movie in movies:
            status = movie.get("status", "pending")
            if status not in summary:
                summary[status] = 0
            summary[status] += 1

        summary["pending"] = max(
            0,
            len(manifest.get("movie_queue", []))
            - sum(
                summary.get(key, 0)
                for key in ("completed", "failed", "rate_limited", "running", "skipped")
            ),
        )
        manifest["summary"] = summary

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    @staticmethod
    def _resolve_source_srt(movie_id: str) -> Optional[Path]:
        candidates = [
            Cfg.MOVIENET_SUBSET_DIR / "subtitle" / f"{movie_id}.srt",
            Cfg.GLOBAL_DATA_DIR / "temp_pipeline" / "subtitle" / f"{movie_id}.srt",
            Cfg.GLOBAL_DATA_DIR / "pipeline_full_test" / "subtitle" / f"{movie_id}.srt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _collect_artifacts(self, movie_id: str) -> Dict[str, Any]:
        artifact_paths = {
            "annotation": Cfg.get_annotation_dir() / f"{movie_id}.json",
            "subtitle": Cfg.get_subtitle_dir() / f"{movie_id}.srt",
            "scene_graph": Cfg.get_scene_graph_dir() / movie_id / f"{movie_id}_auto_graph.json",
            "chunks": Cfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json",
            "script_subscenes": Cfg.get_script_subscenes_dir() / f"{movie_id}_script_subscenes.json",
            "kg_graph": Cfg.get_index_dir() / f"{movie_id}_kg.graphml",
            "visual_index": Cfg.get_index_dir() / "visual_index.faiss",
            "visual_metadata": Cfg.get_index_dir() / "visual_index_metadata.json",
            "visual_scene_index": Cfg.get_index_dir() / "visual_index_scenes.faiss",
            "visual_scene_metadata": Cfg.get_index_dir() / "visual_index_scenes_meta.json",
            "script_scene_index": Cfg.get_index_dir() / "script_scene_index.faiss",
            "script_scene_metadata": Cfg.get_index_dir() / "script_scene_index_metadata.json",
        }
        result: Dict[str, Any] = {}
        for key, path in artifact_paths.items():
            result[key] = {
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
            }
        return result

    def _cleanup_movie_workspace(self, movie_id: str) -> Dict[str, Any]:
        if not self.cleanup_transient_artifacts:
            return {}

        movie_root = self.output_dir / movie_id
        removed: list[str] = []
        freed_bytes = 0

        for folder_name in ("clips", "shot_images"):
            folder = movie_root / folder_name
            if not folder.exists():
                continue
            freed_bytes += sum(
                path.stat().st_size for path in folder.rglob("*") if path.is_file()
            )
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(folder_name)

        if movie_root.exists():
            try:
                if not any(movie_root.iterdir()):
                    movie_root.rmdir()
            except OSError:
                pass

        if removed:
            logger.info(
                "Cleaned transient artifacts for %s: %s (freed %.2f GB)",
                movie_id,
                ", ".join(removed),
                freed_bytes / (1024 ** 3),
            )
        return {
            "removed": removed,
            "freed_bytes": int(freed_bytes),
            "cleaned_at": self._now_iso(),
        } if removed else {}

    @staticmethod
    def _artifacts_look_complete(artifacts: Dict[str, Any]) -> bool:
        required_keys = (
            "annotation",
            "chunks",
            "scene_graph",
            "kg_graph",
            "visual_index",
            "visual_metadata",
            "script_scene_index",
            "script_scene_metadata",
        )
        for key in required_keys:
            info = artifacts.get(key, {})
            if not info.get("exists") or int(info.get("size", 0)) <= 0:
                return False
        return True

    @staticmethod
    def _capture_runtime_config() -> Dict[str, Any]:
        keys = [
            "MOVIERAG_OUTPUT_DIR",
            "MOVIERAG_CLIP_MODEL",
            "MOVIERAG_CLIP_BATCH_SIZE",
            "MOVIERAG_CLIP_DEVICE",
            "MOVIERAG_CLIP_LOCAL_ONLY",
            "MOVIERAG_LLM_MODEL",
            "MOVIERAG_LLM_PRIMARY_MODEL",
            "MOVIERAG_LLM_FALLBACK_MODELS",
            "MOVIERAG_RUNTIME_LLM_MODEL",
            "MOVIERAG_LLM_MAX_RETRIES",
            "MOVIERAG_LLM_RETRY_BASE_SEC",
            "MOVIERAG_VISUAL_SEARCH_STRATEGY",
            "MOVIERAG_VISUAL_SCORE_THRESHOLD",
            "MOVIERAG_ALLOW_GEMINI_VISION",
            "GEMINI_SCENE_MAX_CALLS_PER_HOUR",
            "MOVIERAG_CLEANUP_TRANSIENT_ARTIFACTS",
        ]
        captured = {key: os.getenv(key, "") for key in keys if os.getenv(key)}
        captured["scene_gemini_model"] = Cfg.SCENE_GEMINI_MODEL
        captured["scene_gemini_key_count"] = len(Cfg.get_scene_gemini_api_keys())
        captured["neo4j_uri"] = Cfg.get_neo4j_config()["uri"]
        return captured
