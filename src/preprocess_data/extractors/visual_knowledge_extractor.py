"""
movierag/preprocessing/visual_knowledge_extractor.py
Knowledge Extraction from Visual Content using VLM (Gemini)
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
import mimetypes

from movierag.generation.universal_client import UniversalLLMClient

logger = logging.getLogger(__name__)

class VisualKnowledgeExtractor:
    """
    Extracts semantic knowledge (narratives, entities, relations) from visual frames.
    """

    def __init__(self, client: Optional[UniversalLLMClient] = None):
        self.client = client or UniversalLLMClient(model_id="gemini-1.5-flash")

    def extract_scene_narrative(
        self, 
        frame_paths: List[str], 
        scene_idx: int,
        context: Optional[str] = None
    ) -> str:
        """
        Analyze a set of frames from a scene and return a detailed descriptive narrative.
        """
        if not frame_paths:
            return ""

        logger.info(f"🎨 Extracting visual knowledge for Scene {scene_idx} ({len(frame_paths)} frames)...")

        prompt = f"""
        You are a film analyst. Analyze these frames from Scene {scene_idx} of a movie.
        Describe the following in detail:
        1. Setting & Atmosphere: Where is this taking place? What is the mood?
        2. Key Characters: Who is present? What are they wearing? What are their expressions?
        3. Significant Actions: What is happening? What are the key movements or interactions?
        4. Important Objects: Are there any symbolic or functional objects highlighted?
        
        Provide a concise but comprehensive descriptive narrative in Vietnamese.
        """
        if context:
            prompt += f"\nAdditional Context (e.g. from previous scenes): {context}"

        try:
            # We use the raw gemini client if available to handle multi-image parts
            if hasattr(self.client, "_gemini_client") and self.client._gemini_client:
                from google.genai import types
                
                contents = [prompt]
                # Limit to 5 frames per scene for VLM efficiency/context window
                sampled_frames = frame_paths[:5] 
                
                for path in sampled_frames:
                    mime_type, _ = mimetypes.guess_type(path)
                    with open(path, "rb") as f:
                        contents.append(
                            types.Part.from_bytes(
                                data=f.read(),
                                mime_type=mime_type or "image/jpeg"
                            )
                        )
                
                response = self.client._gemini_client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=contents
                )
                return response.text
            else:
                # Fallback to single image analysis if needed (though not ideal)
                return self.client.generate_vision_content(prompt, frame_paths[0])
                
        except Exception as e:
            logger.error(f"VLM Analysis failed for Scene {scene_idx}: {e}")
            return f"[Lỗi VLM: {e}]"

    def extract_entities_and_relations(self, narrative: str) -> Dict[str, Any]:
        """
        Extract structured data from the narrative text.
        """
        prompt = f"""
        From the following movie scene description, extract key entities and their relationships.
        Return ONLY a JSON object:
        {{
            "entities": [
                {{"name": "...", "type": "Character/Location/Object", "description": "..."}}
            ],
            "relationships": [
                {{"source": "...", "target": "...", "relation": "...", "description": "..."}}
            ]
        }}
        
        Narrative:
        {narrative}
        """
        
        try:
            resp_text = self.client.generate_text(prompt)
            # Find JSON block
            import re
            json_match = re.search(r'\{.*\}', resp_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(resp_text)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "relationships": []}
