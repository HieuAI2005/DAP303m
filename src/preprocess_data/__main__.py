"""
CLI Entry Point for preprocess_data

Usage:
    python -m preprocess_data ingest <video_path> [--id ID] [--out OUT_DIR]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Auto-load .env via centralized env_loader ──
import sys as _sys
from pathlib import Path as _Path
_src_dir = _Path(__file__).resolve().parent.parent
if str(_src_dir) not in _sys.path:
    _sys.path.insert(0, str(_src_dir))
try:
    import env_loader as _env  # noqa: F401 – auto-loads .env on import
except ImportError:
    # Fallback: manual load from known location
    _env_path = _Path(__file__).resolve().parent.parent / "movierag" / ".env"
    if not _env_path.exists():
        _env_path = _Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        import os as _os
        for _line in open(_env_path, "r", encoding="utf-8"):
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# Configure logging
logger = logging.getLogger("preprocess_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _apply_runtime_overrides(args):
    from .config import PreprocessConfig as Cfg

    if getattr(args, "out", None):
        Cfg.set_output_dir(args.out)
        logger.info(f"Using output directory: {Cfg.get_output_root()}")

    env_mapping = {
        "clip_model": "MOVIERAG_CLIP_MODEL",
        "clip_batch_size": "MOVIERAG_CLIP_BATCH_SIZE",
        "clip_device": "MOVIERAG_CLIP_DEVICE",
        "llm_model": "MOVIERAG_LLM_MODEL",
        "llm_primary_model": "MOVIERAG_LLM_PRIMARY_MODEL",
        "llm_fallback_models": "MOVIERAG_LLM_FALLBACK_MODELS",
        "runtime_llm_model": "MOVIERAG_RUNTIME_LLM_MODEL",
        "llm_max_retries": "MOVIERAG_LLM_MAX_RETRIES",
        "llm_retry_base_sec": "MOVIERAG_LLM_RETRY_BASE_SEC",
        "visual_search_strategy": "MOVIERAG_VISUAL_SEARCH_STRATEGY",
        "visual_score_threshold": "MOVIERAG_VISUAL_SCORE_THRESHOLD",
    }

    for attr_name, env_name in env_mapping.items():
        value = getattr(args, attr_name, None)
        if value is not None and value != "":
            os.environ[env_name] = str(value)

    if getattr(args, "allow_gemini_vision", False):
        os.environ["MOVIERAG_ALLOW_GEMINI_VISION"] = "1"


def cmd_ingest(args):
    """
    🎬 INGEST A NEW VIDEO — Fully Automated Master Pipeline
    """
    from .config import PreprocessConfig as Cfg
    from .pipeline import PipelineRunner

    _apply_runtime_overrides(args)

    if not args.video:
        logger.error("❌ <video_path> argument is required")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"❌ Video file not found: {video_path}")
        sys.exit(1)

    movie_id = getattr(args, "id", None) or video_path.stem

    force = getattr(args, "force", False)

    runner = PipelineRunner(movie_id, video_path, getattr(args, "srt", None), force=force)
    success = runner.run_all()
    
    if not success:
        sys.exit(1)

    logger.info(f"    ✓ Keyframes extracted")
    logger.info(f"    ✓ FAISS indexed + graph enriched")
    logger.info(f"  Ready for queries in MovieRAG!")
    logger.info(f"{'=' * 60}")


def cmd_batch_ingest(args):
    from .batch_runner import BatchIngestRunner
    from .config import PreprocessConfig as Cfg

    _apply_runtime_overrides(args)

    output_dir = getattr(args, "out", None) or Cfg.get_output_root()
    runner = BatchIngestRunner(
        output_dir=output_dir,
        movie_ids=getattr(args, "movie_ids", None),
        force=getattr(args, "force", False),
        limit=getattr(args, "limit", None),
        stop_on_rate_limit=not getattr(args, "continue_after_rate_limit", False),
    )
    manifest = runner.run()
    manifest_path = Cfg.get_batch_state_dir() / "batch_manifest.json"
    summary = manifest.get("summary", {})
    logger.info("Batch ingest summary: %s", summary)
    logger.info("Batch manifest saved to: %s", manifest_path)

    if manifest.get("stopped_due_to_rate_limit"):
        logger.warning("Batch stopped after rate limit. Rerun the same command to resume.")
        sys.exit(0)

    failed = int(summary.get("failed", 0))
    sys.exit(0 if failed == 0 else 1)


def main():
    parser = argparse.ArgumentParser(
        prog="preprocess_data",
        description="MovieRAG Data Preprocessing Pipeline Router",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Unified Ingest Router
    p_ingest = sub.add_parser("ingest", help="Ingest data into the system")
    
    p_ingest.add_argument("video", type=str, help="Path to raw video file (.mp4)")
    p_ingest.add_argument("--id", dest="id", type=str, help="Movie ID (default: filename stem)")
    p_ingest.add_argument("--srt", type=str, help="Path to SRT subtitle file")
    p_ingest.add_argument("--out", type=str, help="Override output directory for all generated files")
    p_ingest.add_argument("--force", action="store_true", help="Force re-extraction of all data components")
    p_ingest.add_argument("--clip-model", dest="clip_model", type=str, help="Override CLIP model name")
    p_ingest.add_argument("--clip-batch-size", dest="clip_batch_size", type=int, help="Override CLIP batch size")
    p_ingest.add_argument("--clip-device", dest="clip_device", type=str, help="Override CLIP device")
    p_ingest.add_argument("--llm-model", dest="llm_model", type=str, help="Primary LLM model for ingest/generation")
    p_ingest.add_argument("--llm-primary-model", dest="llm_primary_model", type=str, help="Override universal client primary model")
    p_ingest.add_argument("--llm-fallback-models", dest="llm_fallback_models", type=str, help="Comma-separated fallback LLM models")
    p_ingest.add_argument("--runtime-llm-model", dest="runtime_llm_model", type=str, help="Runtime answer model")
    p_ingest.add_argument("--llm-max-retries", dest="llm_max_retries", type=int, help="Max LLM retry attempts")
    p_ingest.add_argument("--llm-retry-base-sec", dest="llm_retry_base_sec", type=float, help="Base retry backoff in seconds")
    p_ingest.add_argument("--visual-search-strategy", dest="visual_search_strategy", type=str, help="Visual search strategy: basic/hybrid/hierarchical")
    p_ingest.add_argument("--visual-score-threshold", dest="visual_score_threshold", type=float, help="Visual score threshold")
    p_ingest.add_argument("--allow-gemini-vision", dest="allow_gemini_vision", action="store_true", help="Allow Gemini vision fallback outside scene splitting")

    p_batch = sub.add_parser("batch-ingest", help="Run ingest for all discovered raw videos with resume support")
    p_batch.add_argument("--out", type=str, required=True, help="Shared output directory for this batch run")
    p_batch.add_argument("--movie-id", dest="movie_ids", action="append", help="Limit the batch to one or more specific movie IDs")
    p_batch.add_argument("--limit", type=int, help="Process only the first N discovered movies")
    p_batch.add_argument("--force", action="store_true", help="Force rerun even for completed movies")
    p_batch.add_argument("--continue-after-rate-limit", action="store_true", help="Do not stop the batch when an LLM rate limit occurs")
    p_batch.add_argument("--clip-model", dest="clip_model", type=str, help="Override CLIP model name")
    p_batch.add_argument("--clip-batch-size", dest="clip_batch_size", type=int, help="Override CLIP batch size")
    p_batch.add_argument("--clip-device", dest="clip_device", type=str, help="Override CLIP device")
    p_batch.add_argument("--llm-model", dest="llm_model", type=str, help="Primary LLM model for ingest/generation")
    p_batch.add_argument("--llm-primary-model", dest="llm_primary_model", type=str, help="Override universal client primary model")
    p_batch.add_argument("--llm-fallback-models", dest="llm_fallback_models", type=str, help="Comma-separated fallback LLM models")
    p_batch.add_argument("--runtime-llm-model", dest="runtime_llm_model", type=str, help="Runtime answer model")
    p_batch.add_argument("--llm-max-retries", dest="llm_max_retries", type=int, help="Max LLM retry attempts")
    p_batch.add_argument("--llm-retry-base-sec", dest="llm_retry_base_sec", type=float, help="Base retry backoff in seconds")
    p_batch.add_argument("--visual-search-strategy", dest="visual_search_strategy", type=str, help="Visual search strategy: basic/hybrid/hierarchical")
    p_batch.add_argument("--visual-score-threshold", dest="visual_score_threshold", type=float, help="Visual score threshold")
    p_batch.add_argument("--allow-gemini-vision", dest="allow_gemini_vision", action="store_true", help="Allow Gemini vision fallback outside scene splitting")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "batch-ingest":
        cmd_batch_ingest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
