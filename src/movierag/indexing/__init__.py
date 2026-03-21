"""Indexing package for MovieRAG.

Keep package import light-weight.
Individual modules should be imported directly to avoid hard failures from
optional dependencies such as Pillow, moviepy, or CV stacks.
"""

__all__ = [
    "visual_indexer",
    "clip_encoder",
    "knowledge_indexer",
    "parallel_indexer",
    "whisper_transcriber",
    "temporal_grounding",
    "vlm_scene_analyzer",
    "action_recognizer",
    "face_tracker",
    "video_captioner",
    "causal_reasoner",
    "script_aligner",
    "script_scene_indexer",
    "dialogue_indexer",
    "neo4j_graph_store",
]
