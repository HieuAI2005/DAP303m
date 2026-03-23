"""Shared runtime service for web UIs.

Provides a thin application layer over the existing MovieRAG pipeline so
multiple frontends (Gradio, Vite/React, future tools) can reuse the same
runtime behavior.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from movierag.config import get_config
from movierag.generation.llm_generator import LLMGenerator
from movierag.indexing.knowledge_indexer import KnowledgeIndexer
from movierag.pipeline.agentic_pipeline import AgenticVideoRAGPipeline
from preprocess_data.config import PreprocessConfig as PreCfg
from preprocess_data.pipeline import PipelineRunner

logger = logging.getLogger(__name__)


class RuntimeService:
    """Application-facing service for ingest, library inspection, and QA."""

    def __init__(
        self,
        project_root: Path,
        runtime_index_dir: str,
        runtime_output_root: str,
        knowledge_index_dir: str,
        knowledge_index_name: str,
        visual_index_name: Optional[str],
        model_id: str,
    ):
        self.project_root = Path(project_root).resolve()
        self.cfg = get_config()
        self.runtime_index_dir = str(runtime_index_dir)
        self.runtime_output_root = str(runtime_output_root)
        self.knowledge_index_dir = str(knowledge_index_dir)
        self.knowledge_index_name = knowledge_index_name
        self.visual_index_name = visual_index_name
        self.model_id = model_id

        PreCfg.set_output_dir(self.runtime_output_root)

        self.allowed_roots = [
            (self.project_root / "data").resolve(),
            Path(self.runtime_output_root).resolve(),
        ]

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> AgenticVideoRAGPipeline:
        knowledge_indexer = KnowledgeIndexer(
            index_dir=self.knowledge_index_dir,
            index_name=self.knowledge_index_name,
        )
        if knowledge_indexer.index_path.exists():
            knowledge_indexer.load()
        else:
            logger.warning(
                "Knowledge index `%s` was not found at %s",
                self.knowledge_index_name,
                self.knowledge_index_dir,
            )

        shared_text_encoder = getattr(knowledge_indexer, "encoder", None)

        visual_indexer = None
        if self.visual_index_name:
            try:
                from movierag.indexing.visual_indexer import VisualIndexer

                visual_indexer = VisualIndexer(
                    index_dir=self.runtime_index_dir,
                    index_name=self.visual_index_name,
                    encoder=shared_text_encoder,
                )
                visual_indexer.load()
            except ModuleNotFoundError as exc:
                logger.warning(
                    "Visual index dependencies unavailable; visual retrieval disabled: %s",
                    exc,
                )
            except Exception as exc:
                logger.warning("Visual index not loaded: %s", exc)

        script_scene_indexer = None
        try:
            from movierag.indexing.script_scene_indexer import ScriptSceneIndexer

            candidate = ScriptSceneIndexer(
                index_dir=self.runtime_index_dir,
                encoder=shared_text_encoder,
            )
            if candidate.index_path.exists():
                candidate.load()
                script_scene_indexer = candidate
        except Exception as exc:
            logger.warning("Script-scene index unavailable: %s", exc)

        dialogue_indexer = None
        if os.getenv("GEMINI_API_KEY"):
            try:
                from movierag.indexing.dialogue_indexer import DialogueIndexer

                dialogue_indexer = DialogueIndexer()
            except Exception as exc:
                logger.warning("Dialogue index unavailable: %s", exc)

        llm_generator = LLMGenerator()
        return AgenticVideoRAGPipeline(
            visual_indexer=visual_indexer,
            knowledge_indexer=knowledge_indexer,
            script_scene_indexer=script_scene_indexer,
            dialogue_indexer=dialogue_indexer,
            llm_generator=llm_generator,
            model_id=self.model_id,
        )

    def _count_json_items(self, path: Path) -> Optional[int]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("chunks", "all_chunks", "clips", "subscenes", "keyframes"):
                value = data.get(key)
                if isinstance(value, list):
                    return len(value)
        return None

    def _discover_movie_ids(self) -> List[str]:
        pipeline_out = PreCfg.get_output_root()
        ids = set()
        for directory, pattern in (
            (PreCfg.get_annotation_dir(), "*.json"),
            (pipeline_out / "annotation", "*.json"),
            (PreCfg.get_subtitle_dir(), "*.srt"),
            (pipeline_out / "subtitle", "*.srt"),
            (PreCfg.get_meta_dir(), "*.json"),
            (pipeline_out / "meta", "*.json"),
            (PreCfg.get_temporal_chunks_dir(), "*_chunks.json"),
            (PreCfg.get_script_subscenes_dir(), "*_script_subscenes.json"),
        ):
            if not directory.exists():
                continue
            for path in directory.glob(pattern):
                stem = path.stem
                if stem.endswith("_chunks"):
                    stem = stem[: -len("_chunks")]
                if stem.endswith("_script_subscenes"):
                    stem = stem[: -len("_script_subscenes")]
                if stem:
                    ids.add(stem)

        scene_graph_dir = PreCfg.get_scene_graph_dir()
        if scene_graph_dir.exists():
            ids |= {path.name for path in scene_graph_dir.iterdir() if path.is_dir()}

        raw_video_dir = PreCfg.RAW_VIDEOS_DIR
        if raw_video_dir.exists():
            ids |= {
                path.stem
                for path in raw_video_dir.glob("*.*")
                if path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"}
            }

        return sorted(ids)

    def _resolve_artifact_path(self, candidates: list) -> Path:
        """Return first existing path from candidates list, else last candidate."""
        for p in candidates:
            if p and Path(p).exists():
                return Path(p)
        return Path(candidates[-1]) if candidates else Path("")

    def artifact_inventory(self, movie_id: str) -> List[Dict[str, Any]]:
        movie_id = (movie_id or "").strip()
        pipeline_out = PreCfg.get_output_root()
        video_path = PreCfg.get_video_path(movie_id) if movie_id else None
        keyframe_dir = PreCfg.get_shot_keyf_dir() / movie_id if movie_id else None
        meta_path = self._resolve_artifact_path([
            PreCfg.get_meta_dir() / f"{movie_id}.json",
            pipeline_out / "meta" / f"{movie_id}.json",
        ])
        annotation_path = self._resolve_artifact_path([
            PreCfg.get_annotation_dir() / f"{movie_id}.json",
            pipeline_out / "annotation" / f"{movie_id}.json",
        ])
        subtitle_path = self._resolve_artifact_path([
            PreCfg.get_subtitle_dir() / f"{movie_id}.srt",
            pipeline_out / "subtitle" / f"{movie_id}.srt",
        ])
        items = [
            ("raw_video", "Raw video", video_path, None),
            ("metadata", "Metadata", meta_path, None),
            (
                "annotation",
                "Annotation",
                annotation_path,
                None,
            ),
            (
                "subtitle",
                "Subtitle",
                subtitle_path,
                None,
            ),
            (
                "scene_graph",
                "Scene graph",
                PreCfg.get_scene_graph_dir() / movie_id / f"{movie_id}_auto_graph.json",
                None,
            ),
            (
                "temporal_chunks",
                "Temporal chunks",
                PreCfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json",
                None,
            ),
            (
                "script_subscenes",
                "Script sub-scenes",
                PreCfg.get_script_subscenes_dir() / f"{movie_id}_script_subscenes.json",
                None,
            ),
            ("keyframes", "Keyframes", keyframe_dir, None),
        ]
        results = []
        for key, label, path, _ in items:
            count = None
            if key == "keyframes" and path and Path(path).exists():
                count = len(list(Path(path).rglob("*.jpg")))
            elif path and Path(path).exists() and str(path).endswith(".json"):
                count = self._count_json_items(Path(path))
            results.append(
                {
                    "key": key,
                    "label": label,
                    "path": str(path) if path else "",
                    "exists": bool(path and Path(path).exists()),
                    "count": count,
                }
            )
        return results

    def _movie_title(self, movie_id: str) -> str:
        """Return human-readable title from chunks file, falling back to movie_id."""
        try:
            chunks_path = PreCfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json"
            if chunks_path.exists():
                data = json.loads(chunks_path.read_text(encoding="utf-8"))
                chunks = data.get("chunks", data) if isinstance(data, dict) else data
                if chunks:
                    title = chunks[0].get("title", "")
                    if title:
                        return title
        except Exception:
            pass
        return movie_id

    def library_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for movie_id in self._discover_movie_ids():
            artifacts = self.artifact_inventory(movie_id)
            ready = sum(1 for artifact in artifacts if artifact["exists"])
            total = len(artifacts)
            chunk_count = next(
                (
                    artifact["count"]
                    for artifact in artifacts
                    if artifact["key"] == "temporal_chunks"
                ),
                0,
            )
            rows.append(
                {
                    "movie_id": movie_id,
                    "title": self._movie_title(movie_id),
                    "artifact_ratio": f"{ready}/{total}",
                    "video": artifacts[0]["exists"],
                    "subtitle": artifacts[3]["exists"],
                    "chunks": int(chunk_count or 0),
                    "scene_graph": artifacts[4]["exists"],
                    "script_subscenes": artifacts[6]["exists"],
                }
            )
        return rows

    def overview(self) -> Dict[str, Any]:
        rows = self.library_rows()
        index_dir = PreCfg.get_index_dir()
        index_files = list(index_dir.glob("*.faiss")) if index_dir.exists() else []
        return {
            "title": "MovieRAG Studio",
            "workflow": [
                "Nap video moi",
                "Kiem tra artifact da luu",
                "Tim scene / clip",
                "Hoi dap co evidence",
            ],
            "library": {
                "movie_count": len(rows),
                "index_file_count": len(index_files),
                "knowledge_ready": any("knowledge" in path.name for path in index_files),
                "visual_ready": any("visual" in path.name for path in index_files),
                "script_ready": any("script_scene" in path.name for path in index_files),
            },
            "output_root": str(PreCfg.get_output_root()),
            "index_dir": str(PreCfg.get_index_dir()),
        }

    def storage(self, movie_id: str = "") -> Dict[str, Any]:
        return {
            "movie_id": movie_id,
            "output_root": str(PreCfg.get_output_root()),
            "index_dir": str(PreCfg.get_index_dir()),
            "artifacts": self.artifact_inventory(movie_id) if movie_id else [],
            "paths": {
                "raw_videos_dir": str(PreCfg.RAW_VIDEOS_DIR),
                "temporal_chunks_dir": str(PreCfg.get_temporal_chunks_dir()),
                "scene_graphs_dir": str(PreCfg.get_scene_graph_dir()),
                "script_subscenes_dir": str(PreCfg.get_script_subscenes_dir()),
            },
        }

    def _is_allowed_path(self, path_str: str) -> bool:
        path = Path(path_str).expanduser().resolve()
        for root in self.allowed_roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def media_url(self, path_str: str) -> Optional[str]:
        if not path_str:
            return None
        path = Path(path_str).expanduser()
        if not path.exists():
            return None
        resolved = str(path.resolve())
        if not self._is_allowed_path(resolved):
            return None
        return f"/api/media?path={resolved}"

    def _serialize_search_result(self, result: Any) -> Dict[str, Any]:
        metadata = getattr(result, "metadata", {}) or {}
        payload = {
            "movie_id": getattr(result, "movie_id", metadata.get("movie_id", "")),
            "clip_id": getattr(result, "clip_id", metadata.get("clip_id", "")),
            "text": getattr(result, "text", metadata.get("text", "")),
            "path": getattr(result, "path", metadata.get("path", "")),
            "score": float(getattr(result, "score", 0.0) or 0.0),
            "metadata": metadata,
        }
        media_url = self.media_url(payload["path"])
        if media_url:
            payload["url"] = media_url
        return payload

    def reload_runtime_assets(self) -> List[str]:
        messages = []
        for attr_name, label in (
            ("knowledge_indexer", "knowledge"),
            ("visual_indexer", "visual"),
            ("script_scene_indexer", "script-scene"),
        ):
            indexer = getattr(self.pipeline, attr_name, None)
            if not indexer or not hasattr(indexer, "load"):
                messages.append(f"{label}: unavailable")
                continue
            try:
                indexer.load()
                messages.append(f"{label}: reloaded")
            except Exception as exc:
                messages.append(f"{label}: {exc}")
        return messages

    def ingest(
        self,
        movie_id: str,
        video_path: str,
        subtitle_path: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        runner = PipelineRunner(
            movie_id=movie_id,
            video_path=Path(video_path),
            srt_path=Path(subtitle_path) if subtitle_path else None,
            force=bool(force),
        )
        success = runner.run_all()
        reload_messages = self.reload_runtime_assets() if success else ["reload skipped"]
        return {
            "success": success,
            "movie_id": movie_id,
            "completed_steps": runner.completed_steps,
            "failed_step": runner.failed_step,
            "last_error": runner.last_error,
            "reload_messages": reload_messages,
            "storage": self.storage(movie_id),
        }

    @staticmethod
    def _format_time_range(start_time: str, end_time: str) -> str:
        if start_time and end_time:
            return f"{start_time} -> {end_time}"
        return start_time or end_time or "N/A"

    def _build_gallery_items(self, result: Dict[str, Any], visual_results: List[Any], intent: str) -> List[Dict[str, str]]:
        max_items = 6 if intent in ("VISUAL", "MULTIMODAL") else 4
        gallery_items = []
        seen_paths = set()

        def maybe_add(image_path: str) -> bool:
            if not image_path or image_path in seen_paths:
                return False
            media_url = self.media_url(image_path)
            if not media_url:
                return False
            seen_paths.add(image_path)
            gallery_items.append({"path": image_path, "url": media_url})
            return len(gallery_items) >= max_items

        for result_item in visual_results:
            meta = getattr(result_item, "metadata", {}) or {}
            if maybe_add(getattr(result_item, "path", "") or meta.get("path", "")):
                return gallery_items

        for scene in result.get("scene_results", []) or []:
            candidate_paths = [scene.get("representative_frame", "")]
            candidate_paths.extend(scene.get("keyframe_paths", []) or [])
            for image_path in candidate_paths:
                if maybe_add(str(image_path or "")):
                    return gallery_items

        return gallery_items

    def ask(
        self,
        query: str,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        movie_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.pipeline.respond(
            query=query or "",
            image_path=image_path,
            video_path=video_path,
            history=history or [],
            movie_id=movie_id or None,
        )

        answer = result.get("answer", "")
        thoughts = result.get("thoughts", [])
        intent = result.get("intent", "CHAT")
        knowledge_results = result.get("knowledge_results", []) or []
        visual_results = result.get("visual_results", []) or []
        script_results = result.get("script_results", []) or []
        scene_results = result.get("scene_results", []) or []
        temporal_info = result.get("temporal_grounding")

        clean_answer = re.sub(r"```json\s*[\s\S]*?```", "", str(answer or "")).strip()

        clip_path = None
        if (
            visual_results
            and getattr(self.pipeline, "visual_indexer", None)
            and hasattr(self.pipeline.visual_indexer, "extract_clip_at_time")
            and temporal_info
        ):
            top = visual_results[0]
            movie_id = getattr(top, "movie_id", "")
            if movie_id:
                try:
                    clip_path = self.pipeline.visual_indexer.extract_clip_at_time(
                        movie_id,
                        temporal_info.get("start_time", "00:00:00"),
                        end_time=temporal_info.get("end_time"),
                        duration=15,
                    )
                except Exception:
                    clip_path = None

        payload = {
            "intent": intent,
            "answer": clean_answer,
            "thoughts": thoughts,
            "status": {
                "knowledge_docs": len(knowledge_results),
                "visual_frames": len(visual_results),
                "script_scenes": len(script_results),
                "scene_clusters": len(scene_results),
            },
            "temporal_grounding": temporal_info,
            "knowledge_results": [
                self._serialize_search_result(item)
                for item in knowledge_results
                if (getattr(item, "metadata", {}) or {}).get("category") != "moviegraph"
            ][:5],
            "visual_results": [
                self._serialize_search_result(item) for item in visual_results[:8]
            ],
            "script_results": script_results[:6],
            "scene_results": scene_results[:6],
            "gallery_items": self._build_gallery_items(result, visual_results, intent),
            "scene_summary": self._scene_summary(scene_results, temporal_info),
            "clip_url": self.media_url(str(clip_path or "")) if clip_path else None,
        }
        return payload

    def _scene_summary(self, scene_results: List[Dict[str, Any]], temporal_info: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not scene_results:
            return []
        summary = []
        for cluster in scene_results[:4]:
            # Support both dict and TextSearchResult objects
            if hasattr(cluster, "text"):
                meta = cluster.metadata if hasattr(cluster, "metadata") else {}
                c = meta
                score = float(getattr(cluster, "score", 0.0) or 0.0)
                keyframe_paths = meta.get("keyframe_paths", []) or []
            else:
                c = cluster
                score = float(cluster.get("score", 0.0) or 0.0)
                keyframe_paths = cluster.get("keyframe_paths", []) or []
            summary.append(
                {
                    "heading": c.get("heading", "")
                    or c.get("scene_label", "")
                    or c.get("situation", "")
                    or "Scene cluster",
                    "time_range": self._format_time_range(
                        str(c.get("start_time", "") or c.get("start_seconds", "") or ""),
                        str(c.get("end_time", "") or c.get("end_seconds", "") or ""),
                    ),
                    "location": c.get("location", "") or c.get("vision_setting", ""),
                    "score": score,
                    "representative_frame_url": self.media_url(
                        str(c.get("representative_frame", "") or "")
                    ),
                    "keyframe_urls": [
                        self.media_url(str(path))
                        for path in keyframe_paths[:4]
                        if self.media_url(str(path))
                    ],
                }
            )
        if temporal_info and summary:
            summary[0]["grounding"] = self._format_time_range(
                temporal_info.get("start_time", ""),
                temporal_info.get("end_time", ""),
            )
        return summary

    def save_upload_to_temp(self, filename: str, data: bytes) -> str:
        suffix = Path(filename or "").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(data)
            return handle.name
