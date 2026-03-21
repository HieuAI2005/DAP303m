"""
MovieRAG Main Pipeline
======================
Production-ready entry point for the MovieRAG system.

Usage:
    # Build index from data
    python -m movierag.main build --data-dir path/to/movienet/data

    # Run interactive demo
    python -m movierag.main demo

    # Run verification tests
    python -m movierag.main verify
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Load environment variables — search several candidate locations
try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve()
    _candidates = [
        _here.parent.parent / ".env",  # src/.env  (user's actual location)
        _here.parent.parent.parent / ".env",  # project_ky4/.env
        Path(".env"),  # CWD/.env
    ]
    for _env_path in _candidates:
        if _env_path.exists():
            load_dotenv(_env_path, override=True)
            import logging as _l

            _l.getLogger(__name__).info(f"Loaded .env from {_env_path}")
            break
except ImportError:
    pass


# Detect project root (contains src/ and data/)
def _find_project_root() -> Path:
    """Find project root by looking for src/ directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "src").exists() and (current / "data").exists():
            return current
        current = current.parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "movie_data_subset_20")
DEFAULT_INDEX_DIR = str(PROJECT_ROOT / "data" / "indexes")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("movierag")


def _runtime_index_dir() -> str:
    from preprocess_data.config import PreprocessConfig as PreCfg

    return str(PreCfg.get_index_dir())


def _runtime_output_root() -> str:
    from preprocess_data.config import PreprocessConfig as PreCfg

    return str(PreCfg.get_output_root())


def _runtime_model_id() -> str:
    return (
        os.getenv("MOVIERAG_RUNTIME_LLM_MODEL")
        or os.getenv("MOVIERAG_LLM_MODEL")
        or "moonshotai/kimi-k2-instruct"
    )


def _resolve_knowledge_index_dir(
    index_name: str = "movierag_index", preferred_dir: str | None = None
) -> str:
    candidates = []
    if preferred_dir:
        candidates.append(Path(preferred_dir))
    candidates.append(Path(_runtime_index_dir()))
    candidates.append(PROJECT_ROOT / "data" / "indexes")

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered_candidates.append(resolved)

    for candidate in ordered_candidates:
        if (candidate / f"{index_name}.faiss").exists():
            return str(candidate)
    return str(ordered_candidates[0]) if ordered_candidates else DEFAULT_INDEX_DIR


def _find_existing_index_name(
    index_dir: str | Path,
    candidates: list[str],
    metadata_suffixes: tuple[str, ...] = ("_metadata.json", "_map.json"),
) -> str | None:
    index_dir = Path(index_dir)
    for name in candidates:
        faiss_path = index_dir / f"{name}.faiss"
        if not faiss_path.exists():
            continue
        if any((index_dir / f"{name}{suffix}").exists() for suffix in metadata_suffixes):
            return name
    return None


def _resolve_runtime_artifact_dirs(
    preferred_index_dir: str | None = None,
) -> tuple[str, str]:
    """
    Resolve the active runtime index dir and output root for demo/eval.

    When `preferred_index_dir` is provided, it should win for visual/script
    artifacts as well, not just the knowledge index.
    """
    candidates = []
    if preferred_index_dir:
        preferred_path = Path(preferred_index_dir).resolve()
        candidates.append((preferred_path, preferred_path.parent))

    runtime_index = Path(_runtime_index_dir()).resolve()
    candidates.append((runtime_index, Path(_runtime_output_root()).resolve()))
    candidates.append((PROJECT_ROOT / "data" / "indexes", PROJECT_ROOT / "data"))

    seen = set()
    ordered_candidates = []
    for index_path, output_root in candidates:
        key = (index_path, output_root)
        if key in seen:
            continue
        seen.add(key)
        ordered_candidates.append((index_path, output_root))

    for index_path, output_root in ordered_candidates:
        if (
            (index_path / "visual_index.faiss").exists()
            or (index_path / "script_scene_index.faiss").exists()
        ):
            return str(index_path), str(output_root)

    if ordered_candidates:
        first_index, first_root = ordered_candidates[0]
        return str(first_index), str(first_root)
    return DEFAULT_INDEX_DIR, str(PROJECT_ROOT / "data")


def build_index(
    data_dir: str,
    index_dir: str | None = None,
    index_name: str = "movierag_index",
    use_sample: bool = False,
) -> bool:
    """
    Build the knowledge search index from all available data sources.

    Args:
        data_dir: Path to movie data directory (movie_data_subset_20)
        index_dir: Directory to store the index
        index_name: Name for the index files
        use_sample: Whether to use sample data structure

    Returns:
        True if successful, False otherwise
    """
    from movierag.config import get_config
    from movierag.data.unified_loader import UnifiedLoader
    from movierag.data.subtitle_loader import SubtitleLoader
    from movierag.indexing.knowledge_indexer import KnowledgeIndexer

    cfg = get_config()

    effective_index_dir = index_dir or _runtime_index_dir()
    logger.info(f"Building index from: {data_dir}")
    logger.info(f"Index will be saved to: {effective_index_dir}/{index_name}")

    try:
        # Initialize loaders
        unified_loader = UnifiedLoader(data_root=str(cfg.paths.data_dir))
        subtitle_loader = SubtitleLoader(subtitle_dir=str(cfg.paths.subtitle_dir))

        movie_ids = unified_loader.get_all_movie_ids()
        logger.info(f"Found {len(movie_ids)} movies from unified dataset")

        subtitle_movies = subtitle_loader.get_available_movies()
        logger.info(f"Found {len(subtitle_movies)} movies with subtitles")

        # Build knowledge index from all loaders
        indexer = KnowledgeIndexer(index_dir=effective_index_dir, index_name=index_name)
        indexer.build_from_loaders(
            unified_loader=unified_loader,
            subtitle_loader=subtitle_loader,
        )

        logger.info(f"[OK] Knowledge index built with {indexer.num_documents} vectors")
        return True

    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_demo(
    data_dir: str = DEFAULT_DATA_DIR,
    index_dir: str | None = None,
    index_name: str = "movierag_index",
    port: int = 7860,
) -> None:
    """
    Run the interactive Gradio demo.

    Args:
        data_dir: Path to MovieNet data
        index_dir: Directory containing the index
        index_name: Name of the index
        port: Port for the web server
    """
    from movierag.config import get_config
    from movierag.indexing.knowledge_indexer import KnowledgeIndexer

    try:
        import gradio as gr  # noqa: F401 - Used via create_demo
    except ImportError:
        logger.error("Gradio not installed. Run: pip install gradio")
        return

    cfg = get_config()
    from preprocess_data.config import PreprocessConfig as PreCfg

    logger.info("Starting MovieRAG System App...")

    runtime_index_dir, runtime_output_root = _resolve_runtime_artifact_dirs(index_dir)
    PreCfg.set_output_dir(runtime_output_root)

    # Initialize Knowledge Search
    knowledge_index_dir = _resolve_knowledge_index_dir(
        index_name=index_name, preferred_dir=runtime_index_dir
    )
    effective_knowledge_index_name = _find_existing_index_name(
        knowledge_index_dir,
        [
            index_name,
            "knowledge_index",
            "knowledge_unified",
            "videorag_knowledge",
        ],
    ) or index_name
    knowledge_indexer = KnowledgeIndexer(
        index_dir=knowledge_index_dir, index_name=effective_knowledge_index_name
    )

    # Build or load knowledge index
    if not knowledge_indexer.index_path.exists():
        logger.info(
            "Knowledge index `%s` not found in %s. Attempting rebuild...",
            effective_knowledge_index_name,
            knowledge_index_dir,
        )
        try:
            from movierag.data.unified_loader import UnifiedLoader
            from movierag.data.subtitle_loader import SubtitleLoader

            unified_loader = UnifiedLoader(data_root=str(cfg.paths.data_dir))
            subtitle_loader = SubtitleLoader(subtitle_dir=str(cfg.paths.subtitle_dir))
            knowledge_indexer.build_from_loaders(
                unified_loader=unified_loader,
                subtitle_loader=subtitle_loader,
            )
        except ModuleNotFoundError as exc:
            logger.warning(
                "Legacy data loaders are unavailable (%s). Starting demo without rebuilding the knowledge index.",
                exc,
            )
    else:
        logger.info(
            "Loading existing knowledge FAISS index `%s`...",
            effective_knowledge_index_name,
        )
        knowledge_indexer.load()

    shared_text_encoder = getattr(knowledge_indexer, "encoder", None)

    # Initialize Visual Search after knowledge index so both share the same CLIP encoder.
    effective_visual_index_name = _find_existing_index_name(
        runtime_index_dir,
        [
            "visual_index",
            "videorag_visual",
        ],
    )
    visual_indexer = None
    if effective_visual_index_name:
        try:
            from movierag.indexing.visual_indexer import VisualIndexer

            logger.info("Loading preprocess visual FAISS index...")
            visual_indexer = VisualIndexer(
                index_dir=runtime_index_dir,
                index_name=effective_visual_index_name,
                encoder=shared_text_encoder,
            )
            try:
                visual_indexer.load()
            except Exception as e:
                logger.warning(f"Visual index not loaded: {e}")
        except ModuleNotFoundError as exc:
            logger.warning(
                "Visual search dependency is unavailable (%s). Continuing without visual retrieval.",
                exc,
            )
    else:
        try:
            from movierag.indexing.parallel_indexer import ParallelVisualIndexer

            vis_index_dir = str(cfg.paths.data_dir / "unified_dataset")
            logger.info("Loading legacy visual FAISS index...")
            visual_indexer = ParallelVisualIndexer(
                index_dir=vis_index_dir, index_name="movie_hybrid_index"
            )
            try:
                visual_indexer.load()
            except Exception as e:
                logger.warning(f"Visual index not loaded: {e}")
        except ModuleNotFoundError as exc:
            logger.warning(
                "Legacy visual search dependency is unavailable (%s). Continuing without visual retrieval.",
                exc,
            )

    # Import and run integrated app
    from movierag.app import create_integrated_app
    from movierag.generation.llm_generator import LLMGenerator

    # Initialize LLM Generator
    llm_generator = LLMGenerator()

    # Initialize Dialogue Indexer
    from movierag.indexing.dialogue_indexer import DialogueIndexer
    from movierag.indexing.script_scene_indexer import ScriptSceneIndexer

    dialogue_indexer = DialogueIndexer()
    script_scene_indexer = ScriptSceneIndexer(
        index_dir=runtime_index_dir,
        encoder=shared_text_encoder,
    )
    if script_scene_indexer.index_path.exists():
        try:
            script_scene_indexer.load()
        except Exception as e:
            logger.warning(f"Script scene index not loaded: {e}")
    else:
        script_scene_indexer = None

    # Initialize Agentic Pipeline
    from movierag.pipeline.agentic_pipeline import AgenticVideoRAGPipeline

    pipeline = AgenticVideoRAGPipeline(
        visual_indexer=visual_indexer,
        knowledge_indexer=knowledge_indexer,
        script_scene_indexer=script_scene_indexer,
        dialogue_indexer=dialogue_indexer,
        llm_generator=llm_generator,
        model_id=_runtime_model_id(),
    )

    app = create_integrated_app(pipeline=pipeline)
    host = "127.0.0.1"
    launch_error = None
    for candidate_port in range(port, port + 10):
        try:
            if candidate_port != port:
                logger.warning(
                    "Requested port %s is busy. Trying %s.",
                    port,
                    candidate_port,
                )
            app.launch(
                server_name=host,
                server_port=candidate_port,
                allowed_paths=[
                    str(cfg.paths.project_root / "data"),
                    str(cfg.paths.movie_subset_dir),
                    runtime_output_root,
                ],
            )
            return
        except OSError as exc:
            launch_error = exc
            logger.warning("Failed to launch on port %s: %s", candidate_port, exc)
            continue

    if launch_error is not None:
        raise launch_error


def run_api(port: int = 8000) -> None:
    """Run the FastAPI backend for the Vite/React frontend."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is not installed. Run: pip install uvicorn fastapi")
        return

    uvicorn.run("movierag.web_api:app", host="127.0.0.1", port=port, reload=False)


def run_verify(data_dir: str = DEFAULT_DATA_DIR) -> bool:
    """
    Run verification tests on the pipeline.

    Args:
        data_dir: Path to test data

    Returns:
        True if all tests pass, False otherwise
    """
    from movierag.indexing.knowledge_indexer import KnowledgeIndexer

    logger.info("Running verification tests...")
    tests_passed = 0
    tests_total = 0

    # Test 1: Data Loading
    tests_total += 1
    try:
        from preprocess_data.config import PreprocessConfig as PreCfg

        movie_ids = PreCfg.get_all_movie_ids()
        chunks_dir = PreCfg.get_temporal_chunks_dir()
        chunk_files = list(chunks_dir.glob("*_chunks.json")) if chunks_dir.exists() else []

        if movie_ids or chunk_files:
            logger.info(
                f"[OK] Data Loading: {len(movie_ids)} movie ids, {len(chunk_files)} chunk files"
            )
            tests_passed += 1
        else:
            logger.error("[FAIL] Data Loading: No artifacts found")
    except Exception as e:
        logger.error(f"[FAIL] Data Loading: {e}")

    # Test 2: Indexing
    tests_total += 1
    try:
        indexer = KnowledgeIndexer(
            index_dir=str(PROJECT_ROOT / "data" / "indexes_verify"),
            index_name="verify_test",
        )
        indexer.build_index(documents[:5])  # Only 5 for speed

        if indexer._index and indexer._index.ntotal == 5:
            logger.info("[OK] Indexing: Built index with 5 vectors")
            tests_passed += 1
        else:
            logger.error("[FAIL] Indexing: Failed to build index")
    except Exception as e:
        logger.error(f"[FAIL] Indexing: {e}")

    # Test 3: Search
    tests_total += 1
    try:
        results = indexer.search("What happens in the movie?", k=1)
        if results:
            logger.info(f"[OK] Search: Found {len(results)} results")
            tests_passed += 1
        else:
            logger.error("[FAIL] Search: No results")
    except Exception as e:
        logger.error(f"[FAIL] Search: {e}")

    # Summary
    logger.info(f"\n{'=' * 40}")
    logger.info(f"VERIFICATION: {tests_passed}/{tests_total} tests passed")

    if tests_passed == tests_total:
        logger.info("ALL TESTS PASSED - System is ready!")
        return True
    else:
        logger.warning("Some tests failed - Please check the errors above")
        return False


def run_evaluation(
    data_dir: str = DEFAULT_DATA_DIR,
    dataset_file: str = "data/eval_regression.json",
    index_dir: str | None = None,
) -> bool:
    """Run deterministic regression evaluation for scene/script/visual/graph retrieval."""
    from preprocess_data.config import PreprocessConfig as PreCfg
    from movierag.indexing.visual_indexer import VisualIndexer
    from movierag.indexing.knowledge_indexer import KnowledgeIndexer
    from movierag.indexing.dialogue_indexer import DialogueIndexer
    from movierag.indexing.script_scene_indexer import ScriptSceneIndexer
    from movierag.generation.llm_generator import LLMGenerator
    from movierag.pipeline.agentic_pipeline import AgenticVideoRAGPipeline
    from movierag.evaluation.eval_framework import MovieRAGEvaluator

    logger.info("Starting Regression Evaluation...")

    runtime_index_dir, runtime_output_root = _resolve_runtime_artifact_dirs(index_dir)
    PreCfg.set_output_dir(runtime_output_root)
    visual_indexer = VisualIndexer(index_dir=runtime_index_dir)
    visual_indexer.load()

    knowledge_index_dir = _resolve_knowledge_index_dir(
        index_name="movierag_index", preferred_dir=runtime_index_dir
    )
    knowledge_indexer = KnowledgeIndexer(index_dir=knowledge_index_dir)
    knowledge_indexer.load()

    dialogue_indexer = DialogueIndexer()
    shared_text_encoder = getattr(knowledge_indexer, "encoder", None)
    script_scene_indexer = ScriptSceneIndexer(
        index_dir=runtime_index_dir,
        encoder=shared_text_encoder,
    )
    if script_scene_indexer.index_path.exists():
        script_scene_indexer.load()
    else:
        script_scene_indexer = None
    llm_generator = LLMGenerator()

    pipeline = AgenticVideoRAGPipeline(
        visual_indexer=visual_indexer,
        knowledge_indexer=knowledge_indexer,
        script_scene_indexer=script_scene_indexer,
        dialogue_indexer=dialogue_indexer,
        llm_generator=llm_generator,
        model_id=_runtime_model_id(),
    )

    eval_file_path = str(PROJECT_ROOT / dataset_file)

    evaluator = MovieRAGEvaluator(
        pipeline=pipeline, llm_client=None, eval_file_path=eval_file_path
    )
    report = evaluator.run_eval()
    return bool(report.get("summary", {}).get("total_cases", 0))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MovieRAG - Visual Search with Timestamp Retrieval"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build the visual search index")
    build_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to MovieNet data directory",
    )
    build_parser.add_argument(
        "--index-dir",
        "-i",
        type=str,
        default=_runtime_index_dir(),
        help="Directory to store the index",
    )
    build_parser.add_argument(
        "--index-name",
        "-n",
        type=str,
        default="movierag_index",
        help="Name for the index files",
    )
    build_parser.add_argument(
        "--sample", action="store_true", help="Use sample data structure"
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to MovieNet data",
    )
    demo_parser.add_argument(
        "--port", "-p", type=int, default=7860, help="Web server port"
    )
    demo_parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Override the knowledge index directory (defaults to active preprocess output root)",
    )

    api_parser = subparsers.add_parser("api", help="Run FastAPI backend for Vite frontend")
    api_parser.add_argument(
        "--port", "-p", type=int, default=8000, help="API server port"
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Run verification tests")
    verify_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to test data",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run regression evaluation")
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_regression.json",
        help="Path to the JSON evaluation dataset",
    )
    eval_parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Override the knowledge index directory (defaults to active preprocess output root)",
    )

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a new raw movie into the RAG system"
    )
    ingest_parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=True,
        help="Path to the raw video file (.mp4, .mkv)",
    )
    ingest_parser.add_argument(
        "--id",
        "-i",
        type=str,
        required=True,
        help="IMDb ID or unique identifier for the movie (e.g. tt0120338)",
    )
    ingest_parser.add_argument(
        "--srt",
        "-s",
        type=str,
        required=False,
        help="Path to the corresponding .srt subtitle file (optional)",
    )

    args = parser.parse_args()

    if args.command == "build":
        success = build_index(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            index_name=args.index_name,
            use_sample=args.sample,
        )
        sys.exit(0 if success else 1)

    elif args.command == "demo":
        run_demo(data_dir=args.data_dir, index_dir=args.index_dir, port=args.port)

    elif args.command == "verify":
        success = run_verify(data_dir=args.data_dir)
        sys.exit(0 if success else 1)

    elif args.command == "api":
        run_api(port=args.port)

    elif args.command == "eval":
        success = run_evaluation(dataset_file=args.dataset, index_dir=args.index_dir)
        sys.exit(0 if success else 1)

    elif args.command == "ingest":
        from preprocess_data.ingest_movie import MovieIngester

        ingester = MovieIngester(args.video, args.id, args.srt)
        logger.info(f"Starting End-to-End Ingestion for {args.id}...")

        # Run steps
        success = ingester.extract_frames()
        if success:
            raw_dialogues = ingester.parse_srt()
            raw_chunks = ingester.chunk_dialogues(raw_dialogues)
            enriched = ingester.enrich_and_save_chunks(raw_chunks)
            ingester.push_to_faiss(enriched)
            logger.info("Ingestion completed successfully.")
            sys.exit(0)
        else:
            logger.error("Ingestion failed during frame extraction.")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
