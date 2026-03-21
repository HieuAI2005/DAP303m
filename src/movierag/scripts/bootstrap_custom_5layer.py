"""Bootstrap a 1-day 5-layer movie-understanding MVP dataset.

This script samples the highest-quality chunks from the existing
``videorag_chunks/all_chunks.json`` store and exports a reviewable subset.

Outputs:
  - all_chunks.json: canonical MVP chunk file
  - coverage_report.json: field and layer coverage summary
  - movie_summary.json: selected movies and quality stats
  - review_queue.json: chunks that still need manual review

The goal is not to build a perfect final dataset. The goal is to create a
small but usable movie-centric subset that is strong enough to validate the
repo's 5-layer design in one working day.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data" / "pipeline_output" / "videorag_chunks" / "all_chunks.json"
DEFAULT_OUTPUT = ROOT / "data" / "custom_5layer_mvp"


CORE_FIELDS = [
    "chunk_id",
    "movie_id",
    "start_seconds",
    "end_seconds",
    "description",
    "dialogue_text",
    "characters",
    "cast_in_scene",
    "narrative_arc",
    "causal_relations",
    "script_primary_heading",
    "screenplay_context_excerpt",
]

LAYER_FIELDS = {
    "layer_1_temporal": ["movie_id", "chunk_id", "start_seconds", "end_seconds"],
    "layer_2_semantic": [
        "description",
        "situation",
        "vision_setting",
        "vision_actions",
    ],
    "layer_3_dialogue": ["dialogue_text"],
    "layer_4_character": ["characters", "cast_in_scene"],
    "layer_5_narrative_script": [
        "narrative_arc",
        "causal_relations",
        "script_primary_heading",
        "screenplay_context_excerpt",
    ],
}

LAYER_WEIGHTS = {
    "layer_1_temporal": 0.20,
    "layer_2_semantic": 0.25,
    "layer_3_dialogue": 0.20,
    "layer_4_character": 0.20,
    "layer_5_narrative_script": 0.15,
}


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("chunks"), list):
            return data["chunks"]
        if isinstance(data.get("all_chunks"), list):
            return data["all_chunks"]
    raise ValueError(f"Unsupported chunk file format: {path}")


def layer_presence(chunk: Dict[str, Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for layer_name, fields in LAYER_FIELDS.items():
        present = sum(1 for field in fields if has_value(chunk.get(field)))
        scores[layer_name] = present / len(fields)
    return scores


def chunk_quality_score(chunk: Dict[str, Any]) -> float:
    layer_scores = layer_presence(chunk)
    return round(
        sum(layer_scores[name] * weight for name, weight in LAYER_WEIGHTS.items()),
        4,
    )


def normalize_evidence_sources(chunk: Dict[str, Any]) -> List[str]:
    values = [
        chunk.get("timestamp_source"),
        chunk.get("dialogue_source"),
        chunk.get("vision_source"),
    ]
    screenplay_evidence = chunk.get("screenplay_evidence")
    if has_value(screenplay_evidence):
        values.append("screenplay")

    result: List[str] = []
    seen = set()
    for value in values:
        if not has_value(value):
            continue
        text = str(value).strip()
        if text not in seen:
            seen.add(text)
            result.append(text)
    if "bootstrap" not in seen:
        result.append("bootstrap")
    return result


def review_status_for_chunk(chunk: Dict[str, Any]) -> Tuple[str, str, List[str], List[str]]:
    layer_scores = layer_presence(chunk)
    missing_layers = [
        layer_name for layer_name, score in layer_scores.items() if score < 1.0
    ]
    missing_fields = [field for field in CORE_FIELDS if not has_value(chunk.get(field))]

    confidence = chunk_quality_score(chunk)
    if not missing_layers and confidence >= 0.90:
        return "approved_auto", "low", missing_layers, missing_fields
    if "layer_1_temporal" in missing_layers or confidence < 0.60:
        return "needs_manual_review", "high", missing_layers, missing_fields
    if "layer_4_character" in missing_layers or "layer_5_narrative_script" in missing_layers:
        return "needs_enrichment", "high", missing_layers, missing_fields
    return "needs_review", "medium", missing_layers, missing_fields


def enrich_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(chunk)
    for field in CORE_FIELDS:
        if field in ("characters", "cast_in_scene", "causal_relations"):
            enriched.setdefault(field, [])
        else:
            enriched.setdefault(field, "")

    enriched["layer_status"] = layer_presence(enriched)
    enriched["quality_score"] = chunk_quality_score(enriched)
    enriched["evidence_source"] = normalize_evidence_sources(enriched)

    review_status, review_priority, missing_layers, missing_fields = review_status_for_chunk(
        enriched
    )
    enriched["review_status"] = review_status
    enriched["review_priority"] = review_priority
    enriched["missing_layers"] = missing_layers
    enriched["missing_fields"] = missing_fields

    return enriched


def movie_quality_table(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.get("movie_id", ""))].append(chunk)

    rows: List[Dict[str, Any]] = []
    for movie_id, movie_chunks in grouped.items():
        scores = [chunk_quality_score(chunk) for chunk in movie_chunks]
        avg_score = sum(scores) / len(scores)
        rows.append(
            {
                "movie_id": movie_id,
                "num_chunks": len(movie_chunks),
                "avg_quality_score": round(avg_score, 4),
                "selection_score": round(avg_score * math.log2(len(movie_chunks) + 1), 4),
            }
        )
    rows.sort(key=lambda row: (row["selection_score"], row["num_chunks"]), reverse=True)
    return rows


def select_movies(
    chunks: List[Dict[str, Any]],
    max_movies: int,
    requested_movies: List[str],
) -> List[str]:
    if requested_movies:
        available = {str(chunk.get("movie_id", "")) for chunk in chunks}
        return [movie_id for movie_id in requested_movies if movie_id in available]

    ranked = movie_quality_table(chunks)
    return [row["movie_id"] for row in ranked[:max_movies]]


def select_chunks(
    chunks: List[Dict[str, Any]],
    selected_movies: List[str],
    max_chunks: int,
) -> List[Dict[str, Any]]:
    by_movie: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        movie_id = str(chunk.get("movie_id", ""))
        if movie_id in selected_movies:
            by_movie[movie_id].append(chunk)

    for movie_id in by_movie:
        by_movie[movie_id].sort(key=chunk_quality_score, reverse=True)

    if not selected_movies:
        return []

    quota = max(1, max_chunks // len(selected_movies))
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for movie_id in selected_movies:
        movie_chunks = by_movie.get(movie_id, [])
        selected.extend(movie_chunks[:quota])
        leftovers.extend(movie_chunks[quota:])

    if len(selected) < max_chunks:
        leftovers.sort(key=chunk_quality_score, reverse=True)
        selected.extend(leftovers[: max_chunks - len(selected)])

    selected.sort(
        key=lambda chunk: (
            str(chunk.get("movie_id", "")),
            float(chunk.get("start_seconds", 0.0)),
            str(chunk.get("chunk_id", "")),
        )
    )
    return [enrich_chunk(chunk) for chunk in selected[:max_chunks]]


def field_presence_report(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    total = max(1, len(chunks))
    for field in CORE_FIELDS:
        present = sum(1 for chunk in chunks if has_value(chunk.get(field)))
        report[field] = {
            "present": present,
            "missing": total - present,
            "coverage": round(present / total, 4),
        }
    return report


def layer_coverage_report(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    total = max(1, len(chunks))
    report: Dict[str, Dict[str, Any]] = {}
    for layer_name, fields in LAYER_FIELDS.items():
        full = sum(
            1
            for chunk in chunks
            if all(has_value(chunk.get(field)) for field in fields)
        )
        partial = sum(
            1
            for chunk in chunks
            if any(has_value(chunk.get(field)) for field in fields)
        )
        report[layer_name] = {
            "fields": fields,
            "full_coverage": round(full / total, 4),
            "partial_coverage": round(partial / total, 4),
        }
    return report


def build_review_queue(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    queue: List[Dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("review_priority") == "low":
            continue
        queue.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "movie_id": chunk.get("movie_id"),
                "start_seconds": chunk.get("start_seconds"),
                "end_seconds": chunk.get("end_seconds"),
                "quality_score": chunk.get("quality_score"),
                "review_status": chunk.get("review_status"),
                "review_priority": chunk.get("review_priority"),
                "missing_layers": chunk.get("missing_layers", []),
                "missing_fields": chunk.get("missing_fields", []),
            }
        )
    queue.sort(
        key=lambda row: (
            {"high": 0, "medium": 1, "low": 2}.get(str(row["review_priority"]), 3),
            float(row.get("quality_score", 0.0)),
        )
    )
    return queue


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a 1-day custom 5-layer MVP dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Source chunk file. Defaults to videorag_chunks/all_chunks.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for the MVP dataset.",
    )
    parser.add_argument(
        "--movies",
        nargs="*",
        default=[],
        help="Optional explicit movie IDs to keep.",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=3,
        help="Maximum number of movies to sample when --movies is omitted.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=50,
        help="Maximum number of chunks to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input chunk file not found: {input_path}")

    chunks = load_chunks(input_path)
    selected_movies = select_movies(
        chunks=chunks,
        max_movies=max(1, args.max_movies),
        requested_movies=args.movies,
    )
    selected_chunks = select_chunks(
        chunks=chunks,
        selected_movies=selected_movies,
        max_chunks=max(1, args.max_chunks),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    coverage = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "selected_movies": selected_movies,
        "num_selected_chunks": len(selected_chunks),
        "field_presence": field_presence_report(selected_chunks),
        "layer_coverage": layer_coverage_report(selected_chunks),
    }

    movie_rows = movie_quality_table(
        [chunk for chunk in chunks if str(chunk.get("movie_id", "")) in selected_movies]
    )

    write_json(output_dir / "all_chunks.json", selected_chunks)
    write_json(output_dir / "coverage_report.json", coverage)
    write_json(output_dir / "movie_summary.json", movie_rows)
    write_json(output_dir / "review_queue.json", build_review_queue(selected_chunks))

    print(f"Wrote MVP dataset to: {output_dir}")
    print(f"Selected movies: {', '.join(selected_movies) if selected_movies else '(none)'}")
    print(f"Selected chunks: {len(selected_chunks)}")


if __name__ == "__main__":
    main()
