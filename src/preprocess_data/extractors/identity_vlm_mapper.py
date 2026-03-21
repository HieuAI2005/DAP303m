"""
One-Shot VLM Identity Mapper.

Sends the top representative face from each DBSCAN cluster to a VLM, 
along with the TMDB Cast List, to deterministically map Clusters to Character Names.
"""

import json
import logging
import base64
from pathlib import Path
from typing import Dict, List, Any

from preprocess_data.config import PreprocessConfig as Cfg
from movierag.generation.universal_client import UniversalLLMClient

logger = logging.getLogger(__name__)

class IdentityVLMMapper:
    def __init__(self):
        self.vlm = UniversalLLMClient()

    def map_identities(self, movie_id: str) -> bool:
        """Map generic cluster IDs to exact Character names from TMDB using VLM."""
        logger.info(f"\n[5c/8] One-Shot VLM Identity Mapping for {movie_id}...")
        
        kf_dir = Cfg.get_shot_keyf_dir() / movie_id
        clusters_path = kf_dir / "face_clusters.json"
        mapped_out_path = kf_dir / "mapped_identities.json"
        
        if mapped_out_path.exists():
            logger.info("  ⏩ Identities already mapped deterministically.")
            return True
             
        if not clusters_path.exists():
            logger.warning("  face_clusters.json not found. Continuing with empty identity map.")
            self._save_mapping(movie_id, mapped_out_path, {}, "No face clusters available.")
            return True
            
        # 1. Load Clusters
        with open(clusters_path, "r", encoding="utf-8") as f:
            cluster_data = json.load(f)
            
        characters = cluster_data.get("characters", {})
        if not characters:
            logger.warning("  No clusters found to map.")
            self._save_mapping(movie_id, mapped_out_path, {}, "No clusters found to map.")
            return True
            
        # 2. Load TMDB Cast List from Metadata
        meta_path = Cfg.get_meta_dir() / f"{movie_id}.json"
        cast_list_text = "Unknown Cast"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cast = meta.get("cast", [])
                # Take top 10 billed cast
                cast_list_text = ", ".join([f"{c['name']} as {c['character']}" for c in cast[:10]])
            except Exception as e:
                logger.warning(f"  Failed to parse TMDB cast: {e}")
                
        # 3. Prepare One-Shot VLM Prompt
        # We will take the best/most confident face crop from each top cluster
        # To avoid rate limits/context size, we limit to the top 10 clusters 
        top_clusters = list(characters.items())[:10]
        
        prompt = f"You are an expert movie analyst. We need to map {len(top_clusters)} unknown face clusters from a movie to their true actor identities/characters.\n"
        
        images_base64 = []
        
        # --- Add TMDB Actor Reference Images ---
        ref_dir = Cfg.get_actor_references_dir() / movie_id
        ref_count = 0
        if ref_dir.exists():
            prompt += "\n--- SET 1: REFERENCE IMAGES (TMDB ACTORS) ---\n"
            try:
                # We need `cast` from the loaded meta
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cast = meta.get("cast", [])
                
                for cast_member in cast[:15]: # Match the max we downloaded
                    actor_id = cast_member.get('id')
                    actor_name = cast_member.get('name')
                    character = cast_member.get('character')
                    img_path = ref_dir / f"{actor_id}.jpg"
                    
                    if img_path.exists():
                        try:
                            with open(img_path, "rb") as img_file:
                                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                                images_base64.append(f"data:image/jpeg;base64,{encoded_string}")
                                prompt += f"Image {len(images_base64)}: Reference image for Actor '{actor_name}' (playing '{character}').\n"
                                ref_count += 1
                        except Exception as e:
                            logger.warning(f"  Failed to read ref image {img_path}: {e}")
            except Exception as e:
                logger.warning(f"  Failed to process reference images: {e}")
                
        if ref_count == 0:
            prompt += f"\nNo reference images available. Here is the TMDB Cast list for the movie: {cast_list_text}\n"
            
        # --- Add Unknown Face Clusters ---
        prompt += "\n--- SET 2: UNKNOWN FACE CLUSTERS (MOVIE) ---\n"
        prompt += "The following appended images are representative face crops for each unknown cluster.\n"
        
        face_crops_dir = kf_dir / "faces"
        
        for i, (cluster_id, faces) in enumerate(top_clusters):
            # Find the face with the highest prob (confidence)
            best_face = max(faces, key=lambda x: x.get("prob", 0))
            crop_path = face_crops_dir / best_face["crop_file"]
            
            if crop_path.exists():
                try:
                    with open(crop_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        images_base64.append(f"data:image/jpeg;base64,{encoded_string}")
                        prompt += f"Image {len(images_base64)} corresponds to unknown cluster '{cluster_id}'.\n"
                except Exception as e:
                    logger.warning(f"  Failed to read {crop_path}: {e}")
                    
        prompt += "\nYour task is to visually match the unknown clusters in Set 2 to the reference actors in Set 1 (or cast list) provided.\n"
        prompt += "Analyze facial features, hairstyles, and bone structure carefully.\n"
        prompt += "Return EXACTLY a JSON dictionary mapping the cluster ID to the Character name. Example: {\"character_001\": \"Malcolm Crowe\", ...}\n"
        prompt += "If you cannot confidently identify a face, map it to 'Unknown'. Return ONLY valid JSON."
        
        if not images_base64:
            logger.warning("  No valid face images found to send to VLM.")
            self._save_mapping(movie_id, mapped_out_path, {}, "No valid face images found.")
            return True
            
        logger.info(f"  Sending {ref_count} reference faces and {len(images_base64) - ref_count} cluster faces to VLM for TMDB mapping...")
        
        # 4. Invoke VLM with multi-image support
        try:
            response_text = self.vlm.generate_multi_vision(
                prompt=prompt,
                images_base64=images_base64,
                temperature=0.1
            )
            
            logger.info(f"VLM Mapping Output:\n{response_text}")

            mapping = self._parse_mapping_response(response_text)
            
            # Save Mapping
            self._save_mapping(movie_id, mapped_out_path, mapping, response_text)
            logger.info(f"  ✅ Saved identity mappings to {mapped_out_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Identity Mapping failed: {e}")
            self._save_mapping(movie_id, mapped_out_path, {}, f"Identity mapping failed: {e}")
            return True

    def _save_mapping(
        self,
        movie_id: str,
        mapped_out_path: Path,
        mapping: Dict[str, str],
        raw_response: str,
    ) -> None:
        output = {
            "movie_id": movie_id,
            "mappings": mapping,
            "raw_vlm_response": raw_response,
        }

        with open(mapped_out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def _parse_mapping_response(self, response_text: str) -> Dict[str, str]:
        import re

        cleaned = response_text.strip()
        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            cleaned,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            cleaned = fenced.group(1)

        decoder = json.JSONDecoder()
        for idx, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return {str(key): str(value) for key, value in parsed.items()}

        raise ValueError("No JSON object found in VLM response")
