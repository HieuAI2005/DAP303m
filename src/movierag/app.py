"""
MovieRAG Complete Web Application
=================================
Premium dark-mode interface with chat-embedded upload and compact evidence panel.
"""

import logging
import json
import os
import re
from pathlib import Path

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

logger = logging.getLogger(__name__)

CUSTOM_CSS = """
body, .gradio-container {
    background: #f6f7fb !important;
    color: #1f2937 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    margin: 0 !important;
}

.gradio-container {
    max-width: 1240px !important;
    padding: 18px 16px 32px !important;
}

.main-col {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.panel-card,
#chatbot,
#chat-bar,
#gallery,
#vid-player,
#media-acc,
.gr-dataframe {
    background: #ffffff !important;
    border: 1px solid #dbe1ea !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}

.panel-card {
    padding: 12px 14px !important;
}

.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #dbe1ea !important;
}

.tab-nav button {
    border-radius: 10px !important;
    padding: 8px 12px !important;
    color: #4b5563 !important;
}

.tab-nav button.selected {
    background: #e8f0ff !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfd2ff !important;
}

#chatbot {
    min-height: 460px !important;
}

.message.user .message-bubble-border {
    background: #1d4ed8 !important;
    border-radius: 16px 16px 4px 16px !important;
}

.message.bot .message-bubble-border {
    background: #f8fafc !important;
    border: 1px solid #dbe1ea !important;
    border-radius: 16px 16px 16px 4px !important;
    color: #1f2937 !important;
}

#chat-bar {
    padding: 8px 10px !important;
}

#chat-txt textarea {
    background: transparent !important;
    border: none !important;
    color: #111827 !important;
}

#img-drop, #vid-drop {
    background: #ffffff !important;
    border: 1px dashed #c5cfdd !important;
    border-radius: 12px !important;
}

#send-btn,
.gr-button-primary {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border: 1px solid #1d4ed8 !important;
    border-radius: 10px !important;
}

#clear-btn {
    background: #fff7ed !important;
    color: #9a3412 !important;
    border: 1px solid #fdba74 !important;
    border-radius: 10px !important;
}

#status-txt {
    border: none !important;
    background: transparent !important;
}

#status-txt textarea {
    color: #6b7280 !important;
    font-size: .78rem !important;
}

.examples .example {
    background: #ffffff !important;
    border: 1px solid #dbe1ea !important;
    border-radius: 10px !important;
    color: #111827 !important;
}

.gr-dataframe {
    overflow: hidden;
}

.gr-dataframe table {
    background: #ffffff !important;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 999px;
}
"""


def create_integrated_app(pipeline=None):
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio not installed.")

    def _format_time_range(start_time: str, end_time: str) -> str:
        if start_time and end_time:
            return f"{start_time} → {end_time}"
        return start_time or end_time or "N/A"

    def _build_gallery_items(result: dict, visual_results: list, intent: str) -> list:
        if intent not in ("VISUAL", "MULTIMODAL"):
            return []

        gallery_items = []
        seen_paths = set()

        for r in visual_results:
            meta = getattr(r, "metadata", {}) or {}
            image_path = getattr(r, "path", "") or meta.get("path", "")
            if not image_path or image_path in seen_paths:
                continue

            seen_paths.add(image_path)
            gallery_items.append(image_path)
            if len(gallery_items) >= 6:
                return gallery_items

        for scene in result.get("scene_results", []) or []:
            candidate_paths = [scene.get("representative_frame", "")]
            candidate_paths.extend(scene.get("keyframe_paths", []) or [])
            for image_path in candidate_paths:
                if not image_path or image_path in seen_paths:
                    continue
                seen_paths.add(image_path)
                gallery_items.append(image_path)
                if len(gallery_items) >= 6:
                    return gallery_items

        for image_path in result.get("keyframe_paths", []) or []:
            if not image_path or image_path in seen_paths:
                continue
            seen_paths.add(image_path)
            gallery_items.append(image_path)
            if len(gallery_items) >= 6:
                break

        return gallery_items

    def _scene_time_range(scene_result: dict) -> str:
        return _format_time_range(
            str(scene_result.get("start_time", "") or ""),
            str(scene_result.get("end_time", "") or ""),
        )

    def _build_scene_summary(scene_results: list, temporal_info: dict | None) -> str:
        sections = ["### Scene Clusters"]
        if temporal_info:
            sections.append(
                f"- Current grounding: `{_format_time_range(temporal_info.get('start_time', ''), temporal_info.get('end_time', ''))}`"
            )

        if not scene_results:
            sections.append("- No scene evidence available.")
            return "\n".join(sections)

        for idx, cluster in enumerate(scene_results[:4], start=1):
            heading = (
                cluster.get("heading", "")
                or cluster.get("scene_label", "")
                or "Scene cluster"
            )
            location = (
                f" · {cluster.get('location', '')}" if cluster.get("location") else ""
            )
            sections.append("")
            sections.append(f"#### {idx}. {heading}{location}")
            time_range = _scene_time_range(cluster)
            if time_range and time_range != "N/A":
                sections.append(f"- Clip: `{time_range}`")

            semantic_ids = [
                value for value in cluster.get("semantic_scene_ids", []) if value
            ]
            if semantic_ids:
                sections.append(f"- Semantic scene: `{', '.join(sorted(semantic_ids)[:3])}`")

            script_scene_ids = [
                value for value in cluster.get("script_scene_uids", []) if value
            ]
            if script_scene_ids:
                sections.append(
                    f"- Script scene: `{', '.join(sorted(script_scene_ids)[:3])}`"
                )

            clip_ids = [value for value in cluster.get("clip_ids", []) if value]
            if clip_ids:
                sections.append(f"- Clip ids: `{', '.join(sorted(clip_ids)[:3])}`")

            best_visual = float(cluster.get("best_visual_score", 0.0) or 0.0)
            best_script = float(cluster.get("best_script_score", 0.0) or 0.0)
            sections.append(
                f"- Scene score: `{float(cluster.get('score', 0.0) or 0.0):.2f}` | Visual hits: `{len(cluster.get('visual_hits', []))}` · best `{best_visual:.2f}` | Script hits: `{len(cluster.get('script_hits', []))}` · best `{best_script:.2f}`"
            )

            representative_frame = cluster.get("representative_frame", "")
            if representative_frame:
                sections.append(
                    f"- Representative frame: `{os.path.basename(representative_frame)}`"
                )

            script_lines = []
            for script_hit in cluster.get("script_subscenes", [])[:2]:
                line = (
                    f"{script_hit.get('heading', 'Script scene')} · "
                    f"{_format_time_range(script_hit.get('start_time', ''), script_hit.get('end_time', ''))}"
                )
                if script_hit.get("location"):
                    line += f" · {script_hit['location']}"
                if line not in script_lines:
                    script_lines.append(line)
            if script_lines:
                sections.append(
                    f"- Script sub-scenes: {' | '.join(script_lines)}"
                )

            visual_cues = cluster.get("visual_cues", []) or []
            if visual_cues:
                sections.append(f"- Visual cue: {str(visual_cues[0])[:260]}")

        return "\n".join(sections)

    def _count_json_items(path: Path) -> int | None:
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

    def _discover_movie_ids() -> list[str]:
        from preprocess_data.config import PreprocessConfig as PreCfg

        ids = set()

        for directory, pattern in (
            (PreCfg.get_annotation_dir(), "*.json"),
            (PreCfg.get_subtitle_dir(), "*.srt"),
            (PreCfg.get_meta_dir(), "*.json"),
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

    def _artifact_inventory(movie_id: str) -> list[dict]:
        from preprocess_data.config import PreprocessConfig as PreCfg

        movie_id = (movie_id or "").strip()
        video_path = PreCfg.get_video_path(movie_id) if movie_id else None
        keyframe_dir = PreCfg.get_shot_keyf_dir() / movie_id if movie_id else None

        return [
            {
                "label": "Raw video",
                "path": video_path,
                "status": bool(video_path and video_path.exists()),
                "count": None,
            },
            {
                "label": "Metadata",
                "path": PreCfg.get_meta_dir() / f"{movie_id}.json",
                "status": bool(movie_id)
                and (PreCfg.get_meta_dir() / f"{movie_id}.json").exists(),
                "count": None,
            },
            {
                "label": "Annotation",
                "path": PreCfg.get_annotation_dir() / f"{movie_id}.json",
                "status": bool(movie_id)
                and (PreCfg.get_annotation_dir() / f"{movie_id}.json").exists(),
                "count": None,
            },
            {
                "label": "Subtitle",
                "path": PreCfg.get_subtitle_dir() / f"{movie_id}.srt",
                "status": bool(movie_id)
                and (PreCfg.get_subtitle_dir() / f"{movie_id}.srt").exists(),
                "count": None,
            },
            {
                "label": "Scene graph",
                "path": PreCfg.get_scene_graph_dir() / movie_id / f"{movie_id}_auto_graph.json",
                "status": bool(movie_id)
                and (PreCfg.get_scene_graph_dir() / movie_id / f"{movie_id}_auto_graph.json").exists(),
                "count": _count_json_items(
                    PreCfg.get_scene_graph_dir() / movie_id / f"{movie_id}_auto_graph.json"
                )
                if movie_id
                else None,
            },
            {
                "label": "Temporal chunks",
                "path": PreCfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json",
                "status": bool(movie_id)
                and (PreCfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json").exists(),
                "count": _count_json_items(
                    PreCfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json"
                )
                if movie_id
                else None,
            },
            {
                "label": "Script sub-scenes",
                "path": PreCfg.get_script_subscenes_dir() / f"{movie_id}_script_subscenes.json",
                "status": bool(movie_id)
                and (PreCfg.get_script_subscenes_dir() / f"{movie_id}_script_subscenes.json").exists(),
                "count": _count_json_items(
                    PreCfg.get_script_subscenes_dir() / f"{movie_id}_script_subscenes.json"
                )
                if movie_id
                else None,
            },
            {
                "label": "Keyframes",
                "path": keyframe_dir,
                "status": bool(keyframe_dir and keyframe_dir.exists()),
                "count": len(list(keyframe_dir.rglob("*.jpg")))
                if keyframe_dir and keyframe_dir.exists()
                else None,
            },
        ]

    def _build_storage_report(movie_id: str = "") -> str:
        from preprocess_data.config import PreprocessConfig as PreCfg

        movie_id = (movie_id or "").strip()
        lines = [
            "### Artifact storage",
            f"- Thu muc output dang duoc dung: `{PreCfg.get_output_root()}`",
            f"- Thu muc index dang duoc dung: `{PreCfg.get_index_dir()}`",
        ]

        if not movie_id:
            lines.extend(
                [
                    "",
                    "- Nhap `movie_id` de xem storage cua tung phim.",
                    f"- Raw videos dir: `{PreCfg.RAW_VIDEOS_DIR}`",
                    f"- Temporal chunks dir: `{PreCfg.get_temporal_chunks_dir()}`",
                    f"- Scene graphs dir: `{PreCfg.get_scene_graph_dir()}`",
                ]
            )
            return "\n".join(lines)

        lines.append("")
        lines.append(f"#### `{movie_id}`")
        for artifact in _artifact_inventory(movie_id):
            state = "Co" if artifact["status"] else "Thieu"
            extra = (
                f" · `{artifact['count']}` items"
                if artifact.get("count") is not None
                else ""
            )
            lines.append(f"- {artifact['label']}: **{state}**{extra}")
            if artifact.get("path"):
                lines.append(f"  Path: `{artifact['path']}`")
        return "\n".join(lines)

    def _build_library_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for movie_id in _discover_movie_ids():
            artifacts = _artifact_inventory(movie_id)
            ready = sum(1 for artifact in artifacts if artifact["status"])
            total = len(artifacts)
            chunk_count = next(
                (
                    artifact["count"]
                    for artifact in artifacts
                    if artifact["label"] == "Temporal chunks"
                ),
                None,
            )
            rows.append(
                [
                    movie_id,
                    f"{ready}/{total}",
                    "yes" if artifacts[0]["status"] else "no",
                    "yes" if artifacts[3]["status"] else "no",
                    str(chunk_count or 0),
                    "yes" if artifacts[4]["status"] else "no",
                    "yes" if artifacts[6]["status"] else "no",
                ]
            )
        return rows

    def _build_library_summary() -> str:
        from preprocess_data.config import PreprocessConfig as PreCfg

        rows = _build_library_rows()
        index_dir = PreCfg.get_index_dir()
        index_files = list(index_dir.glob("*.faiss")) if index_dir.exists() else []
        visual_ready = any("visual" in path.name for path in index_files)
        script_ready = any("script_scene" in path.name for path in index_files)
        knowledge_ready = any(
            name in path.name
            for path in index_files
            for name in ("knowledge", "movierag_index")
        )

        lines = [
            "### Tinh trang thu vien",
            f"- So `movie_id` da co artifact: `{len(rows)}`",
            f"- So file FAISS trong index dir hien tai: `{len(index_files)}`",
            f"- Knowledge index: `{'san sang' if knowledge_ready else 'thieu'}`",
            f"- Visual index: `{'san sang' if visual_ready else 'thieu'}`",
            f"- Script-scene index: `{'san sang' if script_ready else 'thieu'}`",
        ]
        return "\n".join(lines)

    def _build_quickstart_markdown() -> str:
        return (
            "## Bắt đầu nhanh\n"
            "1. **Nạp video mới**: đưa một phim mới vào pipeline.\n"
            "2. **Thư viện dữ liệu**: xem video đó đã sinh ra artifact gì.\n"
            "3. **Tìm, khớp và hỏi đáp**: test retrieval và QA.\n\n"
            "### Kết quả cần nhìn\n"
            "- Scene / chunk / keyframe / clip\n"
            "- Trạng thái index và artifact\n"
            "- Evidence đi kèm câu trả lời"
        )

    def _build_ingest_help() -> str:
        return (
            "### Cách dùng tab này\n"
            "- Nhập `movie_id` để đặt tên nhất quán cho toàn bộ artifact.\n"
            "- Tải video lên. Có subtitle `.srt` thì tải kèm để dialogue grounding tốt hơn.\n"
            "- Bấm **Chạy full ingest** để chạy pipeline.\n\n"
            "### Sau khi chạy xong\n"
            "- Xem panel trạng thái ingest\n"
            "- Kiểm tra panel storage để biết artifact nào đã được tạo\n"
            "- Sang tab thư viện hoặc tab hỏi đáp để dùng ngay"
        )

    def _build_library_help() -> str:
        return (
            "### Tab này để làm gì?\n"
            "- Cho biết mỗi `movie_id` hiện đã có những artifact nào.\n"
            "- Giúp user mới kiểm tra dữ liệu trước khi retrieval hoặc QA.\n\n"
            "### Cách đọc bảng\n"
            "- `artifact`: số loại artifact đã có / tổng số loại đang theo dõi\n"
            "- `chunks`: số temporal chunks đã tạo\n"
            "- `scene_graph`, `script_subscenes`: cho biết đã sẵn sàng cho reasoning hay chưa"
        )

    def _build_qa_help() -> str:
        return (
            "### Nên hỏi gì trước?\n"
            "- Ai xuất hiện trong cảnh X?\n"
            "- Tìm cảnh có hành động Y\n"
            "- Đoạn video này giống scene nào?\n"
            "- Vì sao một sự kiện xảy ra?\n\n"
            "### Kết quả sẽ hiện ở đâu?\n"
            "- Chat panel: câu trả lời\n"
            "- Bên phải: scene matches, keyframes và clip evidence"
        )

    def _reload_runtime_assets() -> str:
        if not pipeline:
            return "No runtime pipeline attached."

        messages = []
        for attr_name, label in (
            ("knowledge_indexer", "knowledge"),
            ("visual_indexer", "visual"),
            ("script_scene_indexer", "script-scene"),
        ):
            indexer = getattr(pipeline, attr_name, None)
            if not indexer or not hasattr(indexer, "load"):
                messages.append(f"{label}: unavailable")
                continue
            try:
                indexer.load()
                messages.append(f"{label}: reloaded")
            except Exception as exc:
                messages.append(f"{label}: {exc}")
        return " | ".join(messages)

    def _refresh_library(selected_movie: str = ""):
        selected_movie = (selected_movie or "").strip()
        return (
            _build_library_rows(),
            _build_library_summary(),
            _build_storage_report(selected_movie),
        )

    def _run_ingest(movie_id: str, video_input, subtitle_input, force: bool):
        movie_id = (movie_id or "").strip()
        video_path = str(video_input or "").strip()
        subtitle_path = str(subtitle_input or "").strip()

        if not movie_id:
            return (
                "### Trạng thái ingest\n- Thieu `movie_id`.",
                _build_storage_report(""),
                _build_library_rows(),
                _build_library_summary(),
                _build_storage_report(""),
                "❌ Thieu movie_id",
            )
        if not video_path:
            return (
                f"### Trạng thái ingest\n- Chua co video cho `{movie_id}`.",
                _build_storage_report(movie_id),
                _build_library_rows(),
                _build_library_summary(),
                _build_storage_report(movie_id),
                "❌ Thieu video",
            )

        from preprocess_data.pipeline import PipelineRunner

        runner = PipelineRunner(
            movie_id=movie_id,
            video_path=Path(video_path),
            srt_path=Path(subtitle_path) if subtitle_path else None,
            force=bool(force),
        )
        success = runner.run_all()
        reload_msg = _reload_runtime_assets() if success else "reload skipped"

        lines = [
            "### Trạng thái ingest",
            f"- Movie ID: `{movie_id}`",
            f"- Ket qua: `{'thanh cong' if success else 'that bai'}`",
            f"- So buoc da xong: `{len(runner.completed_steps)}`",
        ]
        if runner.completed_steps:
            lines.append(f"- Buoc xong gan nhat: `{runner.completed_steps[-1]}`")
        if runner.failed_step:
            lines.append(f"- Buoc loi: `{runner.failed_step}`")
        if runner.last_error:
            lines.append(f"- Loi: `{runner.last_error}`")
        lines.append(f"- Trang thai reload runtime: `{reload_msg}`")

        storage_report = _build_storage_report(movie_id)
        library_rows = _build_library_rows()
        library_summary = _build_library_summary()
        return (
            "\n".join(lines),
            storage_report,
            library_rows,
            library_summary,
            storage_report,
            "✅ Ingest xong" if success else "❌ Ingest that bai",
        )

    # ── Backend ──────────────────────────────────────────────────────────────
    def respond(user_message, image_input, video_input, chat_history):
        if not user_message and image_input is None and video_input is None:
            return chat_history, [], None, "### Scene Clusters\n- Ready", "🕹️ Ready"

        if not pipeline:
            chat_history = chat_history + [
                {"role": "user", "content": user_message or "📷 Media"},
                {"role": "assistant", "content": "⚠️ Pipeline chưa khởi tạo."},
            ]
            return (
                chat_history,
                [],
                None,
                "### Scene Clusters\n- Pipeline unavailable.",
                "❌ No pipeline",
            )

        result = pipeline.respond(
            query=user_message or "",
            image_path=image_input,
            video_path=video_input,
            history=chat_history,
        )

        intent = result["intent"]
        answer = result["answer"]
        thoughts = result["thoughts"]
        knowledge_results = result["knowledge_results"]
        visual_results = result["visual_results"]
        script_results = result.get("script_results", [])
        scene_results = result.get("scene_results", [])

        # ── Parse temporal grounding ──
        temporal_info = None
        clean_answer = answer

        # Look for temporal_grounding key within any curly braces block
        m = re.search(r'(\{[\s\S]*?"temporal_grounding"[\s\S]*?\})', answer)

        # 1st Priority: Native temporal grounding from pipeline response
        if "temporal_grounding" in result and result["temporal_grounding"]:
            temporal_info = result["temporal_grounding"]
            if m:
                # Still scrub the JSON out of the chat message if present
                json_str = m.group(1)
                full_match = re.search(
                    rf"```json\s*{re.escape(json_str)}\s*```", answer
                )
                if full_match:
                    clean_answer = answer.replace(full_match.group(0), "").strip()
                else:
                    clean_answer = answer.replace(json_str, "").strip()

        # 2nd Priority: Fallback to old regex parsing if pipeline didn't provide it
        elif m:
            json_str = m.group(1)
            try:
                parsed = json.loads(json_str)
                if "temporal_grounding" in parsed:
                    temporal_info = parsed["temporal_grounding"]
                    full_match = re.search(
                        rf"```json\s*{re.escape(json_str)}\s*```", answer
                    )
                    if full_match:
                        clean_answer = answer.replace(full_match.group(0), "").strip()
                    else:
                        clean_answer = answer.replace(json_str, "").strip()
            except json.JSONDecodeError:
                pass

        # Final defensive cleanup: strip any leftover fenced JSON block
        # produced by JudgeAgent temporal grounding formatting.
        clean_answer = re.sub(r"```json\s*[\s\S]*?```", "", clean_answer).strip()

        # ── Build message ──
        bot_msg = ""
        if thoughts:
            t_html = "<br>→ ".join(thoughts)
            bot_msg += f"<details><summary>🧠 <b>Agent Thoughts</b></summary>\n\n_{t_html}_\n\n</details>\n\n"
        bot_msg += clean_answer + "\n"

        if knowledge_results:
            bot_msg += "\n---\n**📚 Sources:**\n"
            for i, r in enumerate(knowledge_results[:3]):
                try:
                    meta = getattr(r, "metadata", r)
                    title = (
                        meta.get("title", getattr(r, "movie_id", "?"))
                        if isinstance(meta, dict)
                        else str(r)[:35]
                    )
                    score = getattr(r, "score", 0.0)
                    bot_msg += f"`[{i + 1}]` {title} · {score:.2f}\n"
                except Exception:
                    bot_msg += f"`[{i + 1}]` [Ref]\n"

        if visual_results:
            bot_msg += "\n**🖼️ Matched Frames:**\n"
            for i, r in enumerate(visual_results[:5]):
                try:
                    meta = getattr(r, "metadata", r)
                    shot = meta.get("shot_id", "") if isinstance(meta, dict) else ""
                    start_time = meta.get("start_time", "") if isinstance(meta, dict) else ""
                    end_time = meta.get("end_time", "") if isinstance(meta, dict) else ""
                    heading = (
                        meta.get("script_primary_heading", "")
                        or meta.get("scene_label", "")
                        if isinstance(meta, dict)
                        else ""
                    )
                    score = getattr(r, "score", 0.0)
                    mid = getattr(r, "movie_id", "")
                    timing = (
                        f" · {start_time}→{end_time}" if start_time and end_time else ""
                    )
                    heading_text = f" · {heading}" if heading else ""
                    bot_msg += (
                        f"`[{i + 1}]` {mid}›{shot}{timing}{heading_text} · **{score:.2f}**\n"
                    )
                except Exception:
                    bot_msg += f"`[{i + 1}]` [Frame]\n"

        if scene_results:
            bot_msg += "\n**🎬 Scene Matches:**\n"
            for i, scene in enumerate(scene_results[:3]):
                heading = scene.get("heading", "") or scene.get("scene_label", "") or "Scene cluster"
                timing = _scene_time_range(scene)
                location = scene.get("location", "")
                location_text = f" · {location}" if location else ""
                bot_msg += (
                    f"`[{i + 1}]` {heading}{location_text} · {timing} · **{float(scene.get('score', 0.0) or 0.0):.2f}**\n"
                )

        # ── Gallery ──
        gallery_images = _build_gallery_items(result, visual_results, intent)
        scene_summary = _build_scene_summary(scene_results, temporal_info)

        # ── Clip extraction ──
        video_output = None
        if (
            visual_results
            and hasattr(pipeline, "visual_indexer")
            and pipeline.visual_indexer
        ):
            top = visual_results[0]
            mid = getattr(top, "movie_id", "")
            if (
                temporal_info
                and mid
                and hasattr(pipeline.visual_indexer, "extract_clip_at_time")
            ):
                video_output = pipeline.visual_indexer.extract_clip_at_time(
                    mid,
                    temporal_info.get("start_time", "00:00:00"),
                    end_time=temporal_info.get("end_time"),
                    duration=15,
                )
            elif mid and intent in ("VISUAL", "MULTIMODAL"):
                fp = getattr(top, "path", "") or top.metadata.get("path", "")
                if fp and hasattr(pipeline.visual_indexer, "extract_video_clip"):
                    video_output = pipeline.visual_indexer.extract_video_clip(
                        mid, fp, duration=12
                    )
        elif (
            scene_results
            and temporal_info
            and hasattr(pipeline, "visual_indexer")
            and pipeline.visual_indexer
            and hasattr(pipeline.visual_indexer, "extract_clip_at_time")
        ):
            top_scene = scene_results[0]
            mid = top_scene.get("movie_id", "")
            if mid:
                video_output = pipeline.visual_indexer.extract_clip_at_time(
                    mid,
                    temporal_info.get("start_time", "00:00:00"),
                    end_time=temporal_info.get("end_time"),
                    duration=15,
                )

        # ── Status ──
        icon = {"VISUAL": "🖼️", "KNOWLEDGE": "📚", "MULTIMODAL": "🎭", "CHAT": "💬"}.get(
            intent, "🔀"
        )
        unique_gallery_count = len(gallery_images)
        status = (
            f"{icon} {intent}  ·  {len(knowledge_results)} docs  ·  "
            f"{len(visual_results)} frames ({unique_gallery_count} unique)  ·  "
            f"{len(script_results)} script scenes  ·  {len(scene_results)} scene clusters"
        )

        disp = user_message if user_message else "📎 [media]"
        chat_history = chat_history + [
            {"role": "user", "content": disp},
            {"role": "assistant", "content": bot_msg},
        ]
        return chat_history, gallery_images, video_output, scene_summary, status

    def clear_all():
        """Clear entire session — no context bleeding between queries."""
        # Reset pipeline's internal chat history if available
        if pipeline and hasattr(pipeline, "_chat_history"):
            pipeline._chat_history.clear()
        return (
            [],
            [],
            None,
            "### Scene Clusters\n- New session started.",
            "",
            None,
            None,
            "🕹️ New session started",
        )

    # ── UI ───────────────────────────────────────────────────────────────────
    with gr.Blocks(css=CUSTOM_CSS, title="MovieRAG") as app:
        gr.Markdown(
            "# MovieRAG Studio\n"
            "Luồng dùng: **1. Nạp video** → **2. Kiểm tra dữ liệu** → **3. Tìm và hỏi đáp**",
            elem_classes="panel-card",
        )

        with gr.Tabs():
            with gr.Tab("0. Tổng quan"):
                gr.Markdown(_build_quickstart_markdown(), elem_classes="panel-card")
                with gr.Row():
                    overview_library = gr.Markdown(
                        value=_build_library_summary(), elem_classes="panel-card"
                    )
                    overview_storage = gr.Markdown(
                        value=_build_storage_report(), elem_classes="panel-card"
                    )

            with gr.Tab("1. Nạp video mới"):
                gr.Markdown(_build_ingest_help(), elem_classes="panel-card")
                with gr.Row():
                    ingest_movie_id = gr.Textbox(
                        label="Movie ID",
                        placeholder="Vi du: tt0120338",
                        scale=1,
                    )
                    ingest_subtitle = gr.File(
                        label="Subtitle tuy chon (.srt)",
                        file_types=[".srt"],
                        type="filepath",
                        scale=1,
                    )
                with gr.Row():
                    ingest_video = gr.Video(
                        label="Tai video moi len",
                        include_audio=True,
                        height=300,
                    )
                    with gr.Column():
                        ingest_force = gr.Checkbox(
                            label="Buoc tao lai toan bo artifact",
                            value=False,
                        )
                        ingest_btn = gr.Button("Chạy full ingest", variant="primary")
                        reload_btn = gr.Button("Tải lại runtime indexes")
                        ingest_status_panel = gr.Markdown(
                            value="### Trạng thái ingest\n- Sẵn sàng nhận video mới.",
                            elem_classes="panel-card",
                        )
                ingest_storage_panel = gr.Markdown(
                    value=_build_storage_report(),
                    elem_classes="panel-card",
                )

            with gr.Tab("2. Thư viện dữ liệu"):
                gr.Markdown(_build_library_help(), elem_classes="panel-card")
                with gr.Row():
                    selected_movie = gr.Textbox(
                        label="Xem chi tiet theo Movie ID",
                        placeholder="Nhap movie_id de xem artifact da luu",
                        scale=1,
                    )
                    refresh_library_btn = gr.Button("Lam moi thu vien", variant="secondary")
                library_table = gr.Dataframe(
                    headers=[
                        "movie_id",
                        "artifact",
                        "video",
                        "subtitle",
                        "chunks",
                        "scene_graph",
                        "script_subscenes",
                    ],
                    datatype=["str"] * 7,
                    value=_build_library_rows(),
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    library_summary_panel = gr.Markdown(
                        value=_build_library_summary(), elem_classes="panel-card"
                    )
                    library_storage_panel = gr.Markdown(
                        value=_build_storage_report(),
                        elem_classes="panel-card",
                    )

            with gr.Tab("3. Tìm, khớp và hỏi đáp"):
                gr.Markdown(_build_qa_help(), elem_classes="panel-card")
                with gr.Row(equal_height=False):
                    # ════════════════ LEFT: Chat ════════════════
                    with gr.Column(scale=7, elem_classes="main-col"):
                        chatbot = gr.Chatbot(
                            label="",
                            elem_id="chatbot",
                            height=460,
                        )

                        # ── Compact input bar ──
                        with gr.Group(elem_id="chat-bar"):
                            with gr.Row():
                                txt = gr.Textbox(
                                    placeholder="Hoi ve phim, tim canh, hoi nhan vat, hoi vi sao mot su kien xay ra...",
                                    show_label=False,
                                    scale=1,
                                    container=False,
                                    elem_id="chat-txt",
                                    lines=1,
                                )
                                send = gr.Button(
                                    "➤",
                                    variant="primary",
                                    scale=0,
                                    min_width=50,
                                    elem_id="send-btn",
                                )
                                new_chat = gr.Button(
                                    "Mới",
                                    variant="secondary",
                                    scale=0,
                                    min_width=64,
                                    elem_id="new-chat-btn",
                                )

                            # ── Upload row inside Accordion ──
                            with gr.Accordion(
                                "Tai media de truy van bang anh / doan video",
                                open=False,
                                elem_id="media-acc",
                            ):
                                with gr.Row():
                                    img = gr.Image(
                                        type="filepath",
                                        label="Tai anh len",
                                        scale=1,
                                        height=140,
                                        elem_id="img-drop",
                                        show_label=True,
                                    )
                                    vid = gr.Video(
                                        label="Tai video snippet len",
                                        scale=1,
                                        height=140,
                                        include_audio=False,
                                        elem_id="vid-drop",
                                        show_label=True,
                                    )
                                with gr.Row():
                                    clear = gr.Button(
                                        "Xoa media dinh kem",
                                        size="sm",
                                        elem_id="clear-btn",
                                    )

                        # Status strip
                        status = gr.Textbox(
                            value="🕹️ San sang",
                            label="",
                            interactive=False,
                            max_lines=1,
                            elem_id="status-txt",
                        )

                        gr.Examples(
                            label="Vi du de user moi thu ngay",
                            examples=[
                                ["Ai dong vai Jack trong Titanic?", None, None],
                                ["Tim canh con tau Titanic chim", None, None],
                                ["Scene nao giong doan video toi vua tai len?", None, None],
                                ["Vi sao nhan vat nay bo chay o doan nay?", None, None],
                            ],
                            inputs=[txt, img, vid],
                        )

                    # ════════════════ RIGHT: Evidence ════════════════
                    with gr.Column(scale=3, elem_classes="main-col"):
                        gr.Markdown(
                            "### Evidence panel\n"
                            "Dùng khu vực này để đối chiếu câu trả lời với **scene matches**, **keyframes** và **clip**.",
                            elem_classes="panel-card",
                        )

                        with gr.Tabs():
                            with gr.Tab("Scene matches"):
                                scene_panel = gr.Markdown(
                                    value="### Scene clusters\n- San sang",
                                    elem_id="scene-panel",
                                )
                            with gr.Tab("Keyframes"):
                                gallery = gr.Gallery(
                                    label="",
                                    elem_id="gallery",
                                    columns=2,
                                    rows=3,
                                    height=340,
                                    object_fit="cover",
                                    show_label=False,
                                )

                            with gr.Tab("Clip"):
                                video_player = gr.Video(
                                    label="",
                                    elem_id="vid-player",
                                    height=300,
                                    interactive=False,
                                    show_label=False,
                                )

        # ── Events ──────────────────────────────────────────────────────────
        _out = [chatbot, gallery, video_player, scene_panel, status]

        send.click(respond, [txt, img, vid, chatbot], _out).then(
            lambda: "", None, [txt]
        )
        txt.submit(respond, [txt, img, vid, chatbot], _out).then(
            lambda: "", None, [txt]
        )
        clear.click(
            clear_all,
            None,
            [chatbot, gallery, video_player, scene_panel, txt, img, vid, status],
        )
        new_chat.click(
            clear_all,
            None,
            [chatbot, gallery, video_player, scene_panel, txt, img, vid, status],
        )
        refresh_library_btn.click(
            _refresh_library,
            [selected_movie],
            [library_table, library_summary_panel, library_storage_panel],
        )
        selected_movie.submit(
            _refresh_library,
            [selected_movie],
            [library_table, library_summary_panel, library_storage_panel],
        )
        ingest_btn.click(
            _run_ingest,
            [ingest_movie_id, ingest_video, ingest_subtitle, ingest_force],
            [
                ingest_status_panel,
                ingest_storage_panel,
                library_table,
                library_summary_panel,
                library_storage_panel,
                status,
            ],
        )
        reload_btn.click(
            lambda: (
                "### Ingest Status\n- Runtime indexes reloaded.",
                _build_storage_report(),
                _build_library_rows(),
                _build_library_summary(),
                _build_storage_report(),
                f"🔄 {_reload_runtime_assets()}",
            ),
            None,
            [
                ingest_status_panel,
                ingest_storage_panel,
                library_table,
                library_summary_panel,
                library_storage_panel,
                status,
            ],
        )

    return app
