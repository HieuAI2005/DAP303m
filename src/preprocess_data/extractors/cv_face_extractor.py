"""
Deterministic Character Tracking via CV.

Leverages facenet-pytorch (MTCNN + InceptionResnetV1) to detect faces across all keyframes 
and cluster them globally using DBSCAN. This provides deterministic character tracking 
without relying on VLM hallucinations.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from PIL import Image

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from sklearn.cluster import DBSCAN
except ImportError:
    MTCNN = None
    InceptionResnetV1 = None
    DBSCAN = None

from preprocess_data.config import PreprocessConfig as Cfg
from preprocess_data.extractors._keyframe_manifest import load_keyframe_entries

logger = logging.getLogger(__name__)


class CVFaceExtractor:
    def __init__(self, device: str = None):
        """Initialize CV Models for face detection and embedding."""
        if MTCNN is None or InceptionResnetV1 is None or DBSCAN is None:
            raise ImportError(
                "Missing dependencies. Please run: pip install facenet-pytorch scikit-learn"
            )

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CV Models on {self.device}...")
        
        # MTCNN for Face Detection & Cropping
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device, keep_all=True
        )
        
        # InceptionResnetV1 for Feature Extraction (Embeddings)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def process_movie(self, movie_id: str, force: bool = False) -> bool:
        """Process all keyframes in a movie, detect faces, cluster them, and save."""
        logger.info(f"\n[CV] Starting Face Extraction & Clustering for {movie_id}")
        
        # Load Keyframe Index
        kf_dir = Cfg.get_shot_keyf_dir() / movie_id
        index_path, keyframes = load_keyframe_entries(
            kf_dir,
            preferred_names=["vector_clean_index.json", "keyframe_index.json"],
        )
        if index_path is None:
            logger.error(f"  ❌ Keyframe index not found for {movie_id}")
            return False
             
        out_path = kf_dir / "face_clusters.json"
        if out_path.exists() and not force:
            logger.info("  ⏩ Face clusters already exist. Use force=True to overwrite.")
            return True

        if not keyframes:
            logger.warning("  No keyframes found in index.")
            self._save_clusters(movie_id, out_path, {}, 0)
            return True
             
        # 1. Detect and Embed Faces
        all_faces = []
        t0 = time.time()
        
        # We will save cropped face images for VLM reference
        face_crops_dir = kf_dir / "faces"
        face_crops_dir.mkdir(parents=True, exist_ok=True)
        
        face_id_counter = 0
        for kf in keyframes:
            img_path = Path(kf["path"])
            if not img_path.exists():
                continue
                
            try:
                img = Image.open(img_path).convert('RGB')
                # Detect faces. keep_all=True means returning a list of faces.
                faces = self.mtcnn(img) 
                
                if faces is not None:
                    # MTCNN returns a tensor of cropped faces [N, 3, 160, 160]
                    # We pass them through ResNet
                    with torch.no_grad():
                        embeddings = self.resnet(faces.to(self.device)).cpu().numpy()
                        
                    # Also get bboxes for visualization later if needed
                    boxes, probs = self.mtcnn.detect(img)
                    
                    for i in range(len(embeddings)):
                        # Save the cropped face for the VLM step
                        crop_img = faces[i].permute(1, 2, 0).numpy()
                        # Un-normalize MTCNN post_process array (-1 to 1) -> (0 to 255)
                        crop_img = ((crop_img + 1) * 127.5).astype(np.uint8)
                        crop_pil = Image.fromarray(crop_img)
                        crop_name = f"face_{face_id_counter:05d}.jpg"
                        crop_pil.save(face_crops_dir / crop_name)
                        
                        all_faces.append({
                            "face_id": face_id_counter,
                            "crop_file": crop_name,
                            "source_img": img_path.name,
                            "scene_id": kf.get("scene_id", "scene_0"),
                            "timestamp_sec": float(kf.get("timestamp_sec", 0.0)),
                            "embedding": embeddings[i],
                            "prob": float(probs[i]),
                            "box": [float(b) for b in boxes[i]]
                        })
                        face_id_counter += 1
                        
            except Exception as e:
                logger.warning(f"  Failed processing {img_path.name}: {e}")
                
        logger.info(f"  📸 Detected {len(all_faces)} faces in {time.time()-t0:.1f}s")
        
        if not all_faces:
            logger.warning("  No faces detected in the entire movie.")
            self._save_clusters(movie_id, out_path, {}, 0)
            return True
            
        # 2. Global Clustering (DBSCAN)
        logger.info("  🧠 Clustering faces to identify unique characters...")
        t_cluster = time.time()
        
        # Prepare embedding matrix
        X = np.vstack([f["embedding"] for f in all_faces])
        
        # Epsilon is the distance threshold. For InceptionResnetV1, 0.7-0.9 is typical for cosine-like L2 norm
        # Normalizing embeddings helps DBSCAN use euclidean distance effectively as cosine similarity
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        clustering = DBSCAN(eps=0.45, min_samples=3, metric="euclidean").fit(X_norm)
        labels = clustering.labels_
        
        clusters = {}
        outliers = 0
        
        for i, label in enumerate(labels):
            if label == -1:
                outliers += 1
                continue
                
            cluster_id = f"character_{label:03d}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
                
            # Exclude raw embedding from JSON output to save space, we just keep index info
            face_info = all_faces[i].copy()
            del face_info["embedding"]
            clusters[cluster_id].append(face_info)
            
        logger.info(f"  identities found: {len(clusters)} (ignored {outliers} outlier faces) in {time.time()-t_cluster:.1f}s")
        
        # Sort clusters by frequency (Main characters appear most)
        sorted_clusters = dict(sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True))
        
        # Save output
        self._save_clusters(movie_id, out_path, sorted_clusters, outliers, len(all_faces))
        logger.info(f"  ✅ Saved face clusters to {out_path.name}")
        return True

    def _save_clusters(
        self,
        movie_id: str,
        out_path: Path,
        characters: Dict[str, List[Dict[str, Any]]],
        outliers: int,
        total_faces: int = 0,
    ) -> None:
        output_data = {
            "movie_id": movie_id,
            "total_faces_detected": total_faces,
            "total_unique_characters": len(characters),
            "noise_faces": outliers,
            "characters": characters,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
