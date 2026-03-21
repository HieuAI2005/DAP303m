"""
Temporal Vision Extractor (VLM).

Extracts pure visual setting and continuous actions from a sequence of chronological
keyframes representing a single semantic scene. Deliberately avoids hallucinatory
identity guessing, leaving that to the Fusion Grapher.
"""

import json
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any

from preprocess_data.config import PreprocessConfig as Cfg
from preprocess_data.extractors._keyframe_manifest import load_keyframe_entries
from movierag.generation.universal_client import (
    LLMRateLimitError,
    UniversalLLMClient,
    is_rate_limit_error,
)

logger = logging.getLogger(__name__)

MULTI_IMAGE_PROMPT = """You are a highly perceptive visual observer.
I am providing you with a chronological sequence of frames from a single continuous movie scene.
Describe the scene strictly based on the visual evidence across these frames.

Focus on:
1. Setting: Describe the physical environment, weather, lighting, and mood/atmosphere.
2. Continuous Action: Describe what the people in the scene are doing across the frames. Use neutral labels like "Person 1", "Person 2", "a man in a red shirt", "a woman with glasses". Do NOT try to guess their names.
3. Interactions & Objects: What objects are they interacting with? What is the physical dynamic?

Return EXACTLY a JSON dictionary with this schema:
{
  "setting": "Detailed description of the location and atmosphere",
  "actions": "Detailed chronological description of physical actions and interactions",
  "visual_objects": ["object1", "object2"]
}
Return ONLY valid JSON.
"""

class VLMVisionExtractor:
    def __init__(self):
        self.vlm = UniversalLLMClient()

    def process_movie(self, movie_id: str, force: bool = False) -> bool:
        """Send chronological batches of images to VLM to get temporal visual descriptions."""
        logger.info(f"\n[6a/8] MapReduce VLM Vision Extraction for {movie_id}...")
        
        kf_dir = Cfg.get_shot_keyf_dir() / movie_id
        out_path = kf_dir / "vlm_temporal_descriptions.json"

        existing_output = self._load_existing_output(out_path) if out_path.exists() else {}
        results = {}
        if existing_output and not force:
            if existing_output.get("status") == "complete":
                logger.info("  ⏩ VLM temporal descriptions already exist. Use force=True to overwrite.")
                return True
            results = dict(existing_output.get("scenes", {}))
            if results:
                logger.info(
                    "  ↻ Resuming VLM extraction with %s cached scene descriptions.",
                    len(results),
                )
            
        index_path, keyframes = load_keyframe_entries(
            kf_dir,
            preferred_names=["vlm_quality_index.json", "keyframe_index.json"],
        )
        if index_path is None:
            logger.error(f"  ❌ Keyframe index not found.")
            return False
        
        # Group keyframes by scene_id (Semantic Scene)
        scene_groups = {}
        for kf in keyframes:
            sid = kf.get("scene_id")
            if not sid:
                continue
            if sid not in scene_groups:
                scene_groups[sid] = []
            scene_groups[sid].append(kf)
            
        if not scene_groups:
            logger.warning("  No valid scene groups found.")
            return False
            
        logger.info(f"  Processing {len(scene_groups)} semantic scenes through VLM...")
        
        for i, (scene_id, kfs) in enumerate(scene_groups.items()):
            if scene_id in results and not force:
                continue
            logger.info(f"    -> Analyzing Scene {i+1}/{len(scene_groups)}: {scene_id} ({len(kfs)} frames)")
            
            # Sort chronologically just in case
            kfs = sorted(kfs, key=lambda x: x.get("timestamp_sec", 0.0))
            
            images_base64 = []
            for kf in kfs:
                img_path = Path(kf["path"])
                if img_path.exists():
                    try:
                        with open(img_path, "rb") as bf:
                            encoded = base64.b64encode(bf.read()).decode('utf-8')
                            images_base64.append(f"data:image/jpeg;base64,{encoded}")
                    except Exception:
                        logger.warning(f"      Failed to read {img_path.name}")

            if not images_base64:
                logger.warning(f"      No valid images for {scene_id}. Skipping.")
                continue

            # Increase sampling for Gemini (was restricted by Groq 5-image limit)
            # Gemini-1.5-Flash/3.1-Lite handles much more, but 12-15 is a sweet spot for high-quality temporal understanding
            MAX_IMAGES = 15 
            if len(images_base64) > MAX_IMAGES:
                # Sample evenly to preserve temporal coverage
                step = len(images_base64) / MAX_IMAGES
                images_base64 = [images_base64[int(i * step)] for i in range(MAX_IMAGES)]
                logger.info(f"      Sampled {MAX_IMAGES} frames from {len(kfs)} for VLM analysis")

            try:
                response_text = self.vlm.generate_multi_vision(
                    prompt=MULTI_IMAGE_PROMPT,
                    images_base64=images_base64,
                    temperature=0.2,
                    max_tokens=Cfg.VLM_MAX_COMPLETION_TOKENS or None,
                    max_completion_tokens=Cfg.VLM_MAX_COMPLETION_TOKENS or None,
                )
                
                # Clean up JSON
                # Clean up JSON
                import re
                json_str = response_text
                
                # Try to extract JSON from markdown block
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    json_str = match.group(1)
                else:
                    # Fallback to greedy curly brace match if no markdown block
                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                
                visual_data = json.loads(json_str)
                results[scene_id] = visual_data
                self._save_results(
                    out_path,
                    movie_id=movie_id,
                    total_scenes=len(scene_groups),
                    results=results,
                    status="partial",
                )
                
            except Exception as e:
                if is_rate_limit_error(e):
                    self._save_results(
                        out_path,
                        movie_id=movie_id,
                        total_scenes=len(scene_groups),
                        results=results,
                        status="partial",
                        error_message=str(e),
                    )
                    raise LLMRateLimitError(
                        f"VLM vision extraction hit rate limit for {movie_id}: {e}"
                    ) from e
                logger.error(f"      ❌ VLM inference failed for {scene_id}: {e}")
                results[scene_id] = {
                    "setting": "Unknown",
                    "actions": "Failed to extract visual actions.",
                    "visual_objects": []
                }
                self._save_results(
                    out_path,
                    movie_id=movie_id,
                    total_scenes=len(scene_groups),
                    results=results,
                    status="partial",
                    error_message=str(e),
                )
                
        # Save results
        self._save_results(
            out_path,
            movie_id=movie_id,
            total_scenes=len(scene_groups),
            results=results,
            status="complete",
        )
            
        logger.info(f"  ✅ Saved temporal VLM descriptions to {out_path.name}")
        return True

    @staticmethod
    def _load_existing_output(out_path: Path) -> Dict[str, Any]:
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _save_results(
        out_path: Path,
        movie_id: str,
        total_scenes: int,
        results: Dict[str, Any],
        status: str,
        error_message: str = "",
    ) -> None:
        output = {
            "movie_id": movie_id,
            "total_scenes": total_scenes,
            "completed_scenes": len(results),
            "status": status,
            "last_error": error_message,
            "scenes": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
