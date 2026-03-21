import json
from pathlib import Path


def load_mock_moviegraphs(json_path: str):
    """
    Loads mock MovieGraphs data designed to intersect perfectly
    with the MovieNet `test1` (Titanic) sample data.
    """
    path = Path(json_path)
    if not path.exists():
        print(f"File not found: {json_path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        documents.append(
            {
                "movie_id": item["movie_id"],
                "clip_id": item["clip_id"],
                "text": item["text"],
                "metadata": item["metadata"],
            }
        )
    return documents


if __name__ == "__main__":
    docs = load_mock_moviegraphs("data/mock_moviegraphs_titanic.json")
    print(f"Loaded {len(docs)} documents.")
