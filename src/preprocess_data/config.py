"""
Shared configuration and paths for preprocess_data pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Robust .env loading
env_paths = [Path(".env"), Path("src/.env"), Path("../.env")]
for p in env_paths:
    if p.exists():
        load_dotenv(p)
        break


class PreprocessConfig:
    """Central configuration for all preprocessing paths and settings."""

    # ── Root directories ──
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # project_ky4/
    GLOBAL_DATA_DIR = PROJECT_ROOT / "data"
    SRC_DIR = PROJECT_ROOT / "src"
    
    @classmethod
    def get_flow_data_dir(cls) -> Path:
        """Returns the central data directory for the unified pipeline."""
        d = cls.GLOBAL_DATA_DIR / "pipeline_output"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Override Output Directory for Isolated Ingestion ──
    # If set, all generated files will go into this single directory
    OUTPUT_DIR: Path | None = (
        Path(os.getenv("MOVIERAG_OUTPUT_DIR")).resolve()
        if os.getenv("MOVIERAG_OUTPUT_DIR")
        else None
    )

    @classmethod
    def set_output_dir(cls, output_dir: str | Path | None) -> Path | None:
        """Persist the active output root for the current process and child processes."""
        if output_dir is None:
            cls.OUTPUT_DIR = None
            os.environ.pop("MOVIERAG_OUTPUT_DIR", None)
            return None

        cls.OUTPUT_DIR = Path(output_dir).resolve()
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["MOVIERAG_OUTPUT_DIR"] = str(cls.OUTPUT_DIR)
        return cls.OUTPUT_DIR

    @classmethod
    def get_output_root(cls) -> Path:
        """Return the active pipeline output root."""
        if cls.OUTPUT_DIR:
            cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            return cls.OUTPUT_DIR
        return cls.get_flow_data_dir()

    @classmethod
    def get_dir(cls, default_dir: Path, subdir: str = "") -> Path:
        """Return OUTPUT_DIR/subdir if OUTPUT_DIR is set, else default_dir."""
        if cls.OUTPUT_DIR:
            d = cls.OUTPUT_DIR / subdir if subdir else cls.OUTPUT_DIR
            d.mkdir(parents=True, exist_ok=True)
            return d
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir

    # ── Raw inputs (Global) ──
    RAW_VIDEOS_DIR = GLOBAL_DATA_DIR / "raw_videos"
    RAW_MOVIES_DIR = GLOBAL_DATA_DIR / "Raw_Movies"

    # ── Default MovieNet data paths (Global) ──
    MOVIENET_DIR = GLOBAL_DATA_DIR / "movienet"
    MOVIENET_SUBSET_DIR = GLOBAL_DATA_DIR / "movienet_subset"

    @classmethod
    def get_annotation_dir(cls) -> Path:
        return cls.get_dir(cls.MOVIENET_SUBSET_DIR / "annotation", "annotation")

    @classmethod
    def get_subtitle_dir(cls) -> Path:
        return cls.get_dir(cls.MOVIENET_SUBSET_DIR / "subtitle", "subtitle")

    @classmethod
    def get_meta_dir(cls) -> Path:
        return cls.get_dir(cls.MOVIENET_SUBSET_DIR / "meta", "meta")

    @classmethod
    def get_script_dir(cls) -> Path:
        # Screenplays are source data, not generated artifacts. They must remain
        # readable even when OUTPUT_DIR points at an isolated pipeline run.
        cls.MOVIENET_SUBSET_DIR.joinpath("script").mkdir(parents=True, exist_ok=True)
        return cls.MOVIENET_SUBSET_DIR / "script"

    @classmethod
    def get_shot_keyf_dir(cls) -> Path:
        # Flow-specific keyframe extraction storage
        return cls.get_dir(cls.get_flow_data_dir() / "shot_keyf", "shot_keyf")

    STANDALONE_KEYF_DIR = GLOBAL_DATA_DIR / "Standalone_Dataset" / "shot_keyf"

    # ── Unified dataset ──
    UNIFIED_DATASET_DIR = GLOBAL_DATA_DIR / "unified_dataset"

    @classmethod
    def get_unified_dataset_dir(cls) -> Path:
        return cls.get_dir(cls.UNIFIED_DATASET_DIR, "unified_dataset")

    UNIFIED_DATASET_JSON = UNIFIED_DATASET_DIR / "movierag_dataset.json"

    # ── MovieGraphs ──
    MOVIEGRAPHS_DIR = GLOBAL_DATA_DIR / "MovieGraphs_repo" / "py3loader_new"
    MOVIEGRAPHS_PKL = MOVIEGRAPHS_DIR / "all_movies.pkl"

    # ── Temporal chunks output ──
    @classmethod
    def get_temporal_chunks_dir(cls) -> Path:
        # Flow-specific chunk storage
        return cls.get_dir(cls.get_flow_data_dir() / "temporal_chunks", "temporal_chunks")

    @classmethod
    def get_script_subscenes_dir(cls) -> Path:
        """Directory for derived script sub-scene artifacts."""
        return cls.get_dir(
            cls.get_flow_data_dir() / "script_subscenes", "script_subscenes"
        )

    @classmethod
    def get_analysis_dir(cls) -> Path:
        """Directory for derived analytical artifacts that are not source metadata."""
        return cls.get_dir(cls.get_flow_data_dir() / "analysis", "analysis")

    @classmethod
    def get_script_alignment_dir(cls) -> Path:
        """Directory for screenplay alignment caches."""
        return cls.get_dir(
            cls.get_flow_data_dir() / "analysis" / "script_alignment",
            "analysis/script_alignment",
        )

    @classmethod
    def get_semantic_script_mapping_dir(cls) -> Path:
        """Directory for semantic-scene to screenplay-scene mapping artifacts."""
        return cls.get_dir(
            cls.get_flow_data_dir() / "analysis" / "semantic_script_mapping",
            "analysis/semantic_script_mapping",
        )

    @classmethod
    def get_logs_dir(cls) -> Path:
        """Directory for batch/system logs."""
        return cls.get_dir(cls.get_flow_data_dir() / "logs", "logs")

    # ── Index output ──
    @classmethod
    def get_index_dir(cls) -> Path:
        # Flow-specific faiss/graph storage
        return cls.get_dir(cls.get_flow_data_dir() / "indexes", "indexes")

    @classmethod
    def get_graph_path(cls) -> Path:
        return cls.get_dir(cls.get_flow_data_dir() / "graphs", "graphs") / "movie_graph_index.graphml"

    @classmethod
    def get_batch_state_dir(cls) -> Path:
        return cls.get_dir(cls.get_flow_data_dir() / "batch_state", "batch_state")

    @classmethod
    def get_scene_graph_dir(cls) -> Path:
        """Directory for per-movie scene graph JSON files (from fusion_llm_grapher)."""
        return cls.get_dir(cls.get_flow_data_dir() / "scene_graphs", "scene_graphs")

    @classmethod
    def get_actor_references_dir(cls) -> Path:
        """Directory for saving downloaded TMDB actor profile images."""
        return cls.get_dir(cls.get_flow_data_dir() / "actor_references", "actor_references")

    # ── Video processing settings ──

    KEYFRAME_HEIGHT = 720  # Target height for extracted keyframes
    KEYFRAME_QUALITY = 2  # JPEG quality (1-31, lower = better)
    KEYFRAME_INTERVAL_SEC = 3.0  # Fallback interval for old-style extraction
    # Scene segmentation must stay on Gemini 3.1 Flash Lite; other Gemini models
    # are intentionally not used in this pipeline path.
    SCENE_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
    SCENE_GEMINI_MAX_CALLS_PER_HOUR = int(
        os.getenv("GEMINI_SCENE_MAX_CALLS_PER_HOUR", "15")
    )
    SCENE_GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_SCENE_TIMEOUT_MS", "45000"))
    SCENE_GEMINI_MAX_OUTPUT_TOKENS = int(
        os.getenv("GEMINI_SCENE_MAX_OUTPUT_TOKENS", "0")
    )
    SCENE_REASONING_MAX_COMPLETION_TOKENS = int(
        os.getenv("MOVIERAG_SCENE_REASONING_MAX_COMPLETION_TOKENS", "4096")
    )
    FUSION_LLM_MAX_COMPLETION_TOKENS = int(
        os.getenv("MOVIERAG_FUSION_LLM_MAX_COMPLETION_TOKENS", "4096")
    )
    VLM_MAX_COMPLETION_TOKENS = int(
        os.getenv("MOVIERAG_VLM_MAX_COMPLETION_TOKENS", "0")
    )
    CHUNK_MAX_DURATION_SEC = float(
        os.getenv("MOVIERAG_CHUNK_MAX_DURATION_SEC", "180")
    )
    CHUNK_MIN_SCRIPT_OVERLAP_SEC = float(
        os.getenv("MOVIERAG_CHUNK_MIN_SCRIPT_OVERLAP_SEC", "20")
    )
    CHUNK_MAX_KEYFRAMES = int(os.getenv("MOVIERAG_CHUNK_MAX_KEYFRAMES", "8"))
    SCRIPT_SCENE_VLM_MAX_IMAGES = int(
        os.getenv("MOVIERAG_SCRIPT_SCENE_VLM_MAX_IMAGES", "8")
    )
    SCRIPT_SCENE_VLM_MIN_FRAMES = int(
        os.getenv("MOVIERAG_SCRIPT_SCENE_VLM_MIN_FRAMES", "1")
    )
    VISUAL_FALLBACK_MAX_OBJECTS = int(
        os.getenv("MOVIERAG_VISUAL_FALLBACK_MAX_OBJECTS", "10")
    )

    # ── Neo4j runtime/store settings ──
    NEO4J_URI = os.getenv("MOVIERAG_NEO4J_URI", os.getenv("NEO4J_URI", "bolt://localhost:7688"))
    NEO4J_USER = os.getenv("MOVIERAG_NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
    NEO4J_PASSWORD = os.getenv(
        "MOVIERAG_NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "movierag123")
    )
    NEO4J_DATABASE = os.getenv(
        "MOVIERAG_NEO4J_DATABASE", os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # ── Meta directories to search (in priority order) ──
    META_SEARCH_DIRS = [
        MOVIENET_SUBSET_DIR / "meta",
        UNIFIED_DATASET_DIR / "meta",
    ]

    @classmethod
    def get_keyframe_search_dirs(cls):
        """Keyframe search directories (in priority order)."""
        return [
            cls.get_shot_keyf_dir(),
            cls.MOVIENET_DIR / "shot_keyf",
            cls.GLOBAL_DATA_DIR / "Standalone_Dataset" / "shot_keyf",
        ]

    @classmethod
    def resolve_keyframe_path(cls, movie_id: str, keyframe_path: str) -> str:
        """Resolve a possibly stale absolute keyframe path to the active output root."""
        raw_path = Path(str(keyframe_path or "")).expanduser()
        if raw_path.exists():
            return str(raw_path.resolve())

        filename = raw_path.name
        if not filename:
            return str(raw_path)

        candidate_roots = []
        seen = set()
        for root in cls.get_keyframe_search_dirs():
            resolved_root = root.resolve()
            if resolved_root in seen:
                continue
            seen.add(resolved_root)
            candidate_roots.append(resolved_root)

        candidate_paths = []
        for root in candidate_roots:
            if movie_id:
                candidate_paths.extend(
                    [
                        root / movie_id / "vector_clean" / filename,
                        root / movie_id / filename,
                    ]
                )
            candidate_paths.append(root / filename)

        for candidate in candidate_paths:
            if candidate.exists():
                return str(candidate.resolve())

        return str(raw_path)

    @classmethod
    def ensure_dirs(cls):
        """Create all output directories if they don't exist."""
        for d in [
            cls.get_temporal_chunks_dir(),
            cls.get_script_subscenes_dir(),
            cls.get_analysis_dir(),
            cls.get_script_alignment_dir(),
            cls.get_semantic_script_mapping_dir(),
            cls.get_index_dir(),
            cls.get_logs_dir(),
            cls.get_shot_keyf_dir(),
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_scene_gemini_api_keys(cls) -> list[str]:
        """Return Gemini API keys reserved for semantic scene segmentation."""
        candidates: list[str] = []

        inline_keys = os.getenv("GEMINI_SCENE_API_KEYS", "")
        if inline_keys:
            normalized = inline_keys.replace("\n", ",").replace(";", ",")
            candidates.extend(part.strip() for part in normalized.split(","))

        for env_name in sorted(
            name for name in os.environ if name.startswith("GEMINI_SCENE_API_KEY_")
        ):
            candidates.append(os.environ.get(env_name, "").strip())

        fallback_key = os.getenv("GEMINI_API_KEY", "").strip()
        if fallback_key:
            candidates.append(fallback_key)

        deduped: list[str] = []
        seen = set()
        for key in candidates:
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    @classmethod
    def get_neo4j_config(cls) -> dict[str, str]:
        """Return Neo4j connection settings for graph sync and runtime query."""
        return {
            "uri": cls.NEO4J_URI,
            "user": cls.NEO4J_USER,
            "password": cls.NEO4J_PASSWORD,
            "database": cls.NEO4J_DATABASE,
        }

    @classmethod
    def get_video_path(cls, movie_id: str) -> Path | None:
        """Find the raw video file for a movie."""
        for d in [cls.RAW_VIDEOS_DIR, cls.RAW_MOVIES_DIR]:
            for ext in [".mp4", ".mkv", ".avi", ".mov"]:
                p = d / f"{movie_id}{ext}"
                if p.exists():
                    return p
        return None

    @classmethod
    def get_all_movie_ids(cls) -> list[str]:
        """Discover all movie IDs from annotation + unified dataset + videos."""
        ids = set()
        if hasattr(cls, 'ANNOTATION_DIR') and cls.ANNOTATION_DIR.exists():
            ids |= {p.stem for p in getattr(cls, 'ANNOTATION_DIR').glob("*.json")}
        if cls.UNIFIED_DATASET_JSON.exists():
            import json

            data = json.loads(cls.UNIFIED_DATASET_JSON.read_text(encoding="utf-8"))
            ids |= set(data.get("movies", {}).keys())
        if cls.RAW_VIDEOS_DIR.exists():
            ids |= {
                p.stem
                for p in cls.RAW_VIDEOS_DIR.glob("*.*")
                if p.suffix in {".mp4", ".mkv", ".avi", ".mov"}
            }
        return sorted(ids)
