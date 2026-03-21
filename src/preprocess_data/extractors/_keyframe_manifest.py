import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def normalize_keyframe_entry(entry: Dict) -> Dict:
    normalized = dict(entry)

    path = normalized.get("path") or normalized.get("frame_path")
    if path:
        normalized["path"] = str(path)

    timestamp_sec = normalized.get("timestamp_sec")
    if timestamp_sec is None:
        timestamp_sec = normalized.get("timestamp", 0.0)

    timestamp_sec = float(timestamp_sec or 0.0)
    normalized["timestamp_sec"] = timestamp_sec
    normalized["timestamp"] = float(normalized.get("timestamp", timestamp_sec))

    scene_id = normalized.get("scene_id")
    scene_idx = normalized.get("scene_idx")
    if not scene_id and scene_idx is not None:
        scene_id = f"scene_{int(scene_idx)}"
    normalized["scene_id"] = scene_id or "scene_0"

    return normalized


def load_keyframe_entries(
    movie_dir: Path, preferred_names: Iterable[str]
) -> Tuple[Path | None, List[Dict]]:
    seen = set()
    candidate_names = list(preferred_names) + [
        "vector_clean_index.json",
        "vlm_quality_index.json",
        "keyframe_index.json",
    ]

    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)

        index_path = movie_dir / name
        if not index_path.exists():
            continue

        data = json.loads(index_path.read_text(encoding="utf-8"))
        keyframes = [
            normalize_keyframe_entry(entry) for entry in data.get("keyframes", [])
        ]
        return index_path, keyframes

    return None, []
