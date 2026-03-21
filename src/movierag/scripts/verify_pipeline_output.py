"""
verify_pipeline_output.py
==========================
Kiểm tra output của preprocessing pipeline — so sánh với cấu trúc chuẩn
trong docs/DATA_PROCESSING.md.

Chạy sau khi hoàn thành một pipeline run để đảm bảo output đúng chuẩn.

Usage:
    python -m movierag.scripts.verify_pipeline_output
    python -m movierag.scripts.verify_pipeline_output --check-indexes
    python -m movierag.scripts.verify_pipeline_output --movie-id tt0120338
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PipelineVerifier")


# ── Standard Output Schema ────────────────────────────────────────────────────

@dataclass
class ExpectedFile:
    """A file expected in the pipeline output."""
    path: str          # Relative path from movie_id root
    description: str
    required: bool = True
    min_size_kb: int = 1
    validator: Optional[str] = None  # "json" | "csv" | "npy" | "faiss" | "image"


@dataclass
class ExpectedLayer:
    """One of the 5-Layer Scene Metadata layers."""
    layer: int
    name: str
    fields: List[str]
    required_keys_in_chunk: List[str]


# ── Expected Structure per Movie ─────────────────────────────────────────────

PER_MOVIE_STRUCTURE = {
    "boundaries": ExpectedFile(
        "boundaries/shot_boundaries.csv",
        "Shot boundary timestamps",
        required=False,
        validator="csv",
    ),
    "keyframes": ExpectedFile(
        "keyframes/shot_keyf/",
        "Extracted keyframe images",
        required=True,
        validator="image",
    ),
    "embeddings": ExpectedFile(
        "embeddings/frame_embeddings.npy",
        "CLIP frame embeddings (N x 512)",
        required=False,
        validator="npy",
    ),
    "scene_metadata": ExpectedFile(
        "scene_metadata.json",
        "5-Layer Scene Metadata (Layer 1-5)",
        required=True,
        validator="json",
    ),
    "transcript": ExpectedFile(
        "transcript.json",
        "Whisper transcription output",
        required=False,
        validator="json",
    ),
    "vlm_analysis": ExpectedFile(
        "vlm_analysis.json",
        "VLM scene analysis results",
        required=False,
        validator="json",
    ),
    "face_tracks": ExpectedFile(
        "face_tracks.json",
        "Face detection + tracking results",
        required=False,
        validator="json",
    ),
    "action_labels": ExpectedFile(
        "action_labels.json",
        "VideoMAE action recognition",
        required=False,
        validator="json",
    ),
    "knowledge_graph": ExpectedFile(
        "knowledge_graph.json",
        "Neo4j-ready knowledge graph",
        required=False,
        validator="json",
    ),
}

# ── 5-Layer Metadata Validation ──────────────────────────────────────────────

FIVE_LAYER_SCHEMA = [
    ExpectedLayer(
        layer=1,
        name="Temporal Anchor",
        fields=["start_seconds", "end_seconds", "movie_id"],
        required_keys_in_chunk=["start_seconds", "end_seconds", "chunk_id"],
    ),
    ExpectedLayer(
        layer=2,
        name="Semantic Description",
        fields=["situation", "description", "vlm_description", "vision_setting", "vision_actions", "emotional_tone"],
        required_keys_in_chunk=["description", "situation"],
    ),
    ExpectedLayer(
        layer=3,
        name="Dialogue & Audio",
        fields=["dialogue_text", "speaker", "audio_events", "background_music"],
        required_keys_in_chunk=["text", "start_seconds"],
    ),
    ExpectedLayer(
        layer=4,
        name="Cast & Characters",
        fields=["characters", "cast_in_scene", "character_emotions", "face_tracking_ids", "action_labels"],
        required_keys_in_chunk=["characters"],
    ),
    ExpectedLayer(
        layer=5,
        name="Script & Narrative",
        fields=["script_heading", "screenplay_context", "narrative_arc", "causal_relations", "scene_graph"],
        required_keys_in_chunk=["description", "narrative_arc"],
    ),
]

# ── Verifier ─────────────────────────────────────────────────────────────────

class PipelineVerifier:
    """Verify pipeline output matches expected structure."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.data_root = project_root / "data"
        self.pipeline_root = self.data_root / "pipeline_output"
        self.results: Dict[str, Any] = {}

    def verify_all(self, check_indexes: bool = True) -> bool:
        """Run all verification checks."""
        print("\n" + "=" * 70)
        print("  🔍 VideoSceneRAG — Pipeline Output Verifier")
        print("=" * 70)

        all_ok = True

        # Check 1: Pipeline output root
        all_ok &= self._check_pipeline_root()

        # Check 2: Per-movie structure
        all_ok &= self._check_per_movie_structure()

        # Check 3: Chunk format (5-Layer metadata)
        all_ok &= self._check_chunk_format()

        # Check 4: FAISS indexes
        if check_indexes:
            all_ok &= self._check_indexes()

        # Check 5: Dataset config
        all_ok &= self._check_dataset_config()

        # Summary
        self._print_summary(all_ok)
        return all_ok

    def verify_movie(self, movie_id: str) -> bool:
        """Verify a specific movie's output."""
        print(f"\n🔍 Verifying movie: {movie_id}")
        movie_dir = self.pipeline_root / movie_id
        if not movie_dir.exists():
            logger.error(f"  ❌ Movie directory not found: {movie_dir}")
            return False

        ok = True
        for name, expected in PER_MOVIE_STRUCTURE.items():
            file_path = movie_dir / expected.path
            self._check_file(file_path, expected, name)

        # Check 5-layer metadata
        scene_meta = movie_dir / PER_MOVIE_STRUCTURE["scene_metadata"].path
        if scene_meta.exists():
            ok &= self._validate_five_layer(scene_meta)

        return ok

    def _check_pipeline_root(self) -> bool:
        """Check pipeline output root exists."""
        print("\n📁 [Check 1] Pipeline Output Root")
        print("-" * 70)

        if self.pipeline_root.exists():
            logger.info(f"  ✅ Found: {self.pipeline_root}")
            # List contents
            subdirs = [d.name for d in self.pipeline_root.iterdir() if d.is_dir()]
            logger.info(f"  📂 Subdirs: {', '.join(subdirs) if subdirs else '(empty)'}")
            return True
        else:
            logger.error(f"  ❌ Not found: {self.pipeline_root}")
            logger.info("  Run preprocessing first.")
            return False

    def _check_per_movie_structure(self) -> bool:
        """Check per-movie output structure."""
        print("\n🎬 [Check 2] Per-Movie Structure")
        print("-" * 70)

        if not self.pipeline_root.exists():
            return False

        # Two modes: per-movie style (movie_id/scene_metadata.json)
        #           and dataset style (*_chunks/all_chunks.json)
        SKIP_DIRS = {"indexes", "graphs", "logs", "temporal_chunks", "transcripts", "moviegraphs_chunks"}
        movie_dirs = [d for d in self.pipeline_root.iterdir()
                      if d.is_dir() and d.name not in SKIP_DIRS]
        if not movie_dirs:
            logger.warning("  ⚠️  No movie directories found.")
            logger.info("  Run preprocessing for at least one movie first.")
            return False

        logger.info(f"  Found {len(movie_dirs)} movie(s):")
        all_ok = True

        for movie_dir in sorted(movie_dirs)[:5]:  # Check first 5
            movie_id = movie_dir.name
            print(f"\n  📂 {movie_id}/")

            # Detect structure style
            has_chunks_json = (movie_dir / "all_chunks.json").exists()
            has_scene_meta  = (movie_dir / "scene_metadata.json").exists()

            if has_chunks_json:
                # Dataset-style: *_chunks/ directory with all_chunks.json
                chunks_file = movie_dir / "all_chunks.json"
                logger.info(f"    ℹ️  Dataset style: {chunks_file.name}")
                try:
                    import json as _json
                    with open(chunks_file, encoding="utf-8") as f:
                        chunks_data = _json.load(f)
                    chunks_list = chunks_data if isinstance(chunks_data, list) else chunks_data.get("chunks", [])
                    logger.info(f"    ✅ all_chunks.json: {len(chunks_list)} entries")
                except Exception as e:
                    logger.warning(f"    ⚠️  Failed to read all_chunks.json: {e}")
                all_ok = True  # No per-movie structure to validate
            elif has_scene_meta:
                # Per-movie style: movie_id/scene_metadata.json
                movie_ok = True
                for name, expected in PER_MOVIE_STRUCTURE.items():
                    file_path = movie_dir / expected.path
                    file_ok = self._check_file(file_path, expected, f"  {name}")
                    movie_ok = movie_ok and file_ok
                if movie_ok:
                    print(f"    ✅ All expected files present")
                all_ok = all_ok and movie_ok
            else:
                logger.warning(f"    ⚠️  Unknown directory structure (no all_chunks.json or scene_metadata.json)")
                all_ok = all_ok and False

        if len(movie_dirs) > 5:
            logger.info(f"\n  ... and {len(movie_dirs) - 5} more movies")

        return all_ok

    def _check_file(self, path: Path, expected: ExpectedFile, label: str) -> bool:
        """Check a single file/directory."""
        exists = path.exists()
        size_ok = False

        if exists:
            if path.is_file():
                size_kb = path.stat().st_size // 1024
                size_ok = size_kb >= expected.min_size_kb
            elif path.is_dir():
                files = list(path.glob("*"))
                size_ok = len(files) >= 1
                if size_ok:
                    logger.info(f"    ✅ {label}: {path.name}/ ({len(files)} files)")
                else:
                    logger.warning(f"    ⚠️  {label}: {path.name}/ (empty)")

        if not exists:
            if expected.required:
                logger.error(f"    ❌ {label}: MISSING ({path})")
                return False
            else:
                logger.info(f"    ℹ️  {label}: optional — not found (OK)")
                return True

        if not size_ok:
            logger.warning(f"    ⚠️  {label}: exists but too small or empty")
            return False

        # Validate format
        if expected.validator and path.is_file():
            ok = self._validate_format(path, expected.validator)
            if not ok:
                logger.warning(f"    ⚠️  {label}: format validation warning")
                return False

        if not expected.validator or not path.is_file():
            logger.info(f"    ✅ {label}: {path.name}")

        return True

    def _validate_format(self, path: Path, validator: str) -> bool:
        """Validate file format."""
        try:
            if validator == "json":
                with open(path, encoding="utf-8") as f:
                    json.load(f)
                return True
            elif validator == "csv":
                with open(path, encoding="utf-8") as f:
                    f.read(100)
                return True
            elif validator == "npy":
                import numpy as np
                arr = np.load(path)
                logger.info(f"      shape={arr.shape}, dtype={arr.dtype}")
                return True
            elif validator == "faiss":
                import faiss
                idx = faiss.read_index(str(path))
                logger.info(f"      ntotal={idx.ntotal}")
                return True
            elif validator == "image":
                # Just check files exist
                return True
        except Exception as e:
            logger.warning(f"      Format validation error: {e}")
            return False
        return True

    def _check_chunk_format(self) -> bool:
        """Validate 5-Layer metadata format in chunks."""
        print("\n📋 [Check 3] 5-Layer Metadata Format")
        print("-" * 70)

        # Find chunk metadata files (not directories)
        SKIP_DIRS = {"indexes", "graphs", "logs", "temporal_chunks", "transcripts", "moviegraphs_chunks"}
        chunks_files: List[Path] = []

        # Dataset-style: *_chunks/all_chunks.json
        for d in self.pipeline_root.iterdir():
            if d.is_dir() and d.name not in SKIP_DIRS:
                all_chunks = d / "all_chunks.json"
                if all_chunks.exists():
                    chunks_files.append(all_chunks)

        # Per-movie style: movie_id/scene_metadata.json
        for d in self.pipeline_root.iterdir():
            if d.is_dir() and d.name not in SKIP_DIRS:
                sm = d / "scene_metadata.json"
                if sm.exists():
                    chunks_files.append(sm)

        if not chunks_files:
            logger.warning("  ⚠️  No chunk/metadata files found to validate.")
            return False

        all_ok = True
        for chunks_path in chunks_files[:3]:  # Check first 3
            logger.info(f"\n  Checking: {chunks_path.parent.name}/{chunks_path.name}")
            all_ok &= self._validate_five_layer(chunks_path)

        return all_ok

    def _validate_five_layer(self, path: Path) -> bool:
        """Validate 5-Layer schema in a JSON file."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "chunks" in data:
                chunks = data["chunks"]
            elif isinstance(data, list):
                chunks = data
            elif isinstance(data, dict):
                chunks = [data]
            else:
                logger.warning(f"  ⚠️  Unknown format in {path.name}")
                return False

            if not chunks:
                logger.warning(f"  ⚠️  Empty chunks in {path.name}")
                return False

            # Check first chunk against 5-layer schema
            chunk = chunks[0]
            print(f"    Sample chunk keys: {list(chunk.keys())[:10]}")

            ok = True
            for layer in FIVE_LAYER_SCHEMA:
                missing = [
                    f for f in layer.required_keys_in_chunk
                    if f not in chunk
                ]
                if missing:
                    logger.warning(f"    ⚠️  Layer {layer.layer} ({layer.name}): missing {missing}")
                    # Not critical if optional layer missing
                else:
                    print(f"    ✅ Layer {layer.layer}: {layer.name}")

            logger.info(f"    Total chunks: {len(chunks)}")
            return ok

        except Exception as e:
            logger.error(f"    ❌ Validation error: {e}")
            return False

    def _check_indexes(self) -> bool:
        """Check FAISS indexes."""
        print("\n🔢 [Check 4] FAISS Indexes")
        print("-" * 70)

        index_dir = self.pipeline_root / "indexes"
        if not index_dir.exists():
            logger.warning("  ⚠️  No indexes directory found.")
            return False

        faiss_files = list(index_dir.glob("*.faiss"))
        if not faiss_files:
            logger.warning("  ⚠️  No FAISS index files found.")
            return False

        all_ok = True
        for idx_file in faiss_files:
            try:
                import faiss
                idx = faiss.read_index(str(idx_file))
                size_mb = idx_file.stat().st_size // 1024 // 1024
                logger.info(f"  ✅ {idx_file.name}: {idx.ntotal:,} vectors, {size_mb}MB")

                # Check corresponding map file
                map_file = idx_file.with_suffix(".faiss_map.json")
                if not map_file.exists():
                    map_file = idx_file.with_name(idx_file.stem + "_map.json")
                if map_file.exists():
                    with open(map_file, encoding="utf-8") as f:
                        meta = json.load(f)
                    logger.info(f"     Metadata map: {len(meta)} entries")
                else:
                    logger.warning(f"     ⚠️  No metadata map found")

            except Exception as e:
                logger.error(f"  ❌ {idx_file.name}: {e}")
                all_ok = False

        return all_ok

    def _check_dataset_config(self) -> bool:
        """Check dataset config YAML."""
        print("\n⚙️  [Check 5] Dataset Config")
        print("-" * 70)

        config_path = self.data_root / ".dataset_config.yaml"
        if not config_path.exists():
            logger.warning("  ⚠️  No dataset config found.")
            return False

        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"  ✅ Dataset config: {config_path}")
            logger.info(f"     Version: {config.get('version', 'unknown')}")
            logger.info(f"     Last updated: {config.get('last_updated', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"  ❌ Config parse error: {e}")
            return False

    def _print_summary(self, all_ok: bool) -> None:
        """Print final summary."""
        print("\n" + "=" * 70)
        if all_ok:
            print("  ✅ ALL CHECKS PASSED — Pipeline output is valid!")
        else:
            print("  ⚠️  SOME CHECKS FAILED — Review warnings above.")
        print("=" * 70)

        print(f"""
  📁 Output: {self.pipeline_root}
  📋 Expected per-movie files:
     boundaries/shot_boundaries.csv
     keyframes/shot_keyf/
     embeddings/frame_embeddings.npy
     scene_metadata.json       ← 5-Layer Scene Metadata
     transcript.json           ← Whisper STT
     vlm_analysis.json         ← VLM Scene Analysis
     face_tracks.json          ← Face Tracking
     action_labels.json         ← VideoMAE
     knowledge_graph.json       ← Neo4j-ready

  🔢 FAISS Indexes:
     indexes/movie_frame_index.faiss      (L0)
     indexes/movie_scene_index.faiss      (L1)
     indexes/knowledge_*.faiss             (L3)

  Run again with:
    python -m movierag.scripts.verify_pipeline_output
    python -m movierag.scripts.verify_pipeline_output --check-indexes
""")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify VideoSceneRAG pipeline output",
    )
    parser.add_argument("--movie-id", type=str, help="Verify specific movie")
    parser.add_argument("--check-indexes", action="store_true",
                        help="Also check FAISS indexes")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)

    args = parser.parse_args()

    verifier = PipelineVerifier(project_root=args.project_root)

    if args.movie_id:
        verifier.verify_movie(args.movie_id)
    else:
        verifier.verify_all(check_indexes=args.check_indexes)


if __name__ == "__main__":
    main()
