from importlib import import_module

__all__ = [
    "MetadataCrawler",
    "KeyframeExtractor",
    "CVFaceExtractor",
    "IdentityVLMMapper",
    "VLMVisionExtractor",
    "FusionLLMGrapher",
    "VisualPruner",
    "VisualKnowledgeExtractor",
    "STTGenerator",
    "ClipExtractor",
    "SemanticSceneSegmenter",
]

_EXPORT_MAP = {
    "MetadataCrawler": ".metadata_extractor",
    "KeyframeExtractor": ".keyframe_extractor",
    "CVFaceExtractor": ".cv_face_extractor",
    "IdentityVLMMapper": ".identity_vlm_mapper",
    "VLMVisionExtractor": ".vlm_vision_extractor",
    "FusionLLMGrapher": ".fusion_llm_grapher",
    "VisualPruner": ".visual_pruner",
    "VisualKnowledgeExtractor": ".visual_knowledge_extractor",
    "STTGenerator": ".audio_stt_extractor",
    "ClipExtractor": ".clip_extractor",
    "SemanticSceneSegmenter": ".semantic_scene_segmenter",
}


def __getattr__(name):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
