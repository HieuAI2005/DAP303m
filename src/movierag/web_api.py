"""FastAPI backend for the Vite + React frontend."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from movierag.main import (
    PROJECT_ROOT,
    _find_existing_index_name,
    _resolve_knowledge_index_dir,
    _resolve_runtime_artifact_dirs,
    _runtime_model_id,
)
from movierag.web_service import RuntimeService

logger = logging.getLogger(__name__)

app = FastAPI(title="MovieRAG Web API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_service() -> RuntimeService:
    runtime_index_dir, runtime_output_root = _resolve_runtime_artifact_dirs(None)
    knowledge_index_dir = _resolve_knowledge_index_dir(
        index_name="movierag_index", preferred_dir=runtime_index_dir
    )
    knowledge_index_name = _find_existing_index_name(
        knowledge_index_dir,
        ["knowledge_videorag", "movierag_index", "knowledge_index", "knowledge_unified", "videorag_knowledge"],
    ) or "movierag_index"
    visual_index_name = _find_existing_index_name(
        runtime_index_dir,
        ["visual_index", "videorag_visual"],
    )

    return RuntimeService(
        project_root=PROJECT_ROOT,
        runtime_index_dir=runtime_index_dir,
        runtime_output_root=runtime_output_root,
        knowledge_index_dir=knowledge_index_dir,
        knowledge_index_name=knowledge_index_name,
        visual_index_name=visual_index_name,
        model_id=_runtime_model_id(),
    )


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/overview")
def overview() -> dict:
    return get_service().overview()


@app.get("/api/library")
def library() -> dict:
    service = get_service()
    return {"rows": service.library_rows(), "summary": service.overview()["library"]}


@app.get("/api/storage")
def storage(movie_id: str = Query(default="")) -> dict:
    return get_service().storage(movie_id)


@app.post("/api/reload")
def reload_runtime() -> dict:
    return {"messages": get_service().reload_runtime_assets()}


@app.post("/api/qa")
async def qa(
    query: str = Form(default=""),
    history_json: str = Form(default="[]"),
    movie_id: str = Form(default=""),
    image: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
) -> dict:
    service = get_service()
    try:
        history = __import__("json").loads(history_json or "[]")
    except Exception:
        history = []

    image_path = None
    video_path = None
    if image is not None:
        image_path = service.save_upload_to_temp(image.filename or "query_image", await image.read())
    if video is not None:
        video_path = service.save_upload_to_temp(video.filename or "query_video", await video.read())

    return service.ask(
        query=query,
        image_path=image_path,
        video_path=video_path,
        history=history,
        movie_id=movie_id or None,
    )


@app.post("/api/ingest")
async def ingest(
    movie_id: str = Form(...),
    force: bool = Form(default=False),
    video: UploadFile = File(...),
    subtitle: Optional[UploadFile] = File(default=None),
) -> dict:
    service = get_service()
    video_path = service.save_upload_to_temp(video.filename or f"{movie_id}.mp4", await video.read())
    subtitle_path = None
    if subtitle is not None:
        subtitle_path = service.save_upload_to_temp(
            subtitle.filename or f"{movie_id}.srt",
            await subtitle.read(),
        )
    return service.ingest(
        movie_id=movie_id,
        video_path=video_path,
        subtitle_path=subtitle_path,
        force=force,
    )


@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    history_json: str = Form(default="[]"),
    session_id: str = Form(default="default"),
    movie_id: str = Form(default=""),
) -> dict:
    """Multi-turn chat endpoint — sends message + history to the RAG pipeline."""
    service = get_service()
    try:
        history = __import__("json").loads(history_json or "[]")
    except Exception:
        history = []

    result = service.ask(query=message, history=history, movie_id=movie_id or None)
    return {
        "answer": result.get("answer", ""),
        "intent": result.get("intent", ""),
        "thoughts": result.get("thoughts", []),
        "knowledge_results": result.get("knowledge_results", []),
        "visual_results": result.get("visual_results", []),
        "session_id": session_id,
    }


@app.get("/api/media")
def media(path: str = Query(...)) -> FileResponse:
    service = get_service()
    if not service._is_allowed_path(path):
        raise HTTPException(status_code=403, detail="Path is outside allowed roots")
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(file_path)
