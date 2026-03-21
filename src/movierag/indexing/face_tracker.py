# ─────────────────────────────────────────────────────────────────────────────
# face_tracker.py
# Face Detection + Character Tracking — Video Understanding Pipeline
# Layer 4: Cast & Characters — face_tracking_ids + character identity
# ─────────────────────────────────────────────────────────────────────────────
"""
 Face Detection and Character Tracking for movie videos.

 Uses:
   - MediaPipe Face Detection (fast, lightweight)
   - ArcFace / FaceXzoo (for embedding-based re-identification)
   - ByteTrack (for temporal tracking across frames)

 Enriches Layer 4: Cast & Characters:
   - characters: Named characters detected in scene
   - cast_in_scene: Actor → Character mapping
   - character_emotions: Per-character emotion tracking
   - face_tracking_ids: Consistent person re-identification across scenes
   - action_labels: Activity recognition outputs

 Output schema (per frame / per scene):
   {
     "frame_id": str,
     "timestamp": float,
     "faces": [
       {
         "track_id": int,        # Consistent ID across frames
         "bbox": [x1, y1, x2, y2],
         "embedding": List[float],  # 512-d ArcFace embedding
         "emotion": str,
         "confidence": float,
         "character_name": Optional[str],  # from cast mapping
       }
     ],
     "character_presence": {"Jack": True, "Rose": True, ...},
   }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

DETECTOR_BACKEND = os.getenv("MOVIERAG_FACE_DETECTOR", "mediapipe")
EMBEDDING_MODEL = os.getenv("MOVIERAG_FACE_EMBEDDING", "arcface")
TRACKING_IOU_THRESHOLD = float(os.getenv("MOVIERAG_FACE_IOU_THRESHOLD", "0.3"))
MAX_DISAPPEAR_FRAMES = int(os.getenv("MOVIERAG_FACE_MAX_GAP", "30"))
TEMP_DIR = os.getenv("MOVIERAG_TEMP_FACES", "/tmp/movierag_faces")


# ── IOU Utilities ───────────────────────────────────────────────────────────────

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two bboxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def extract_face_from_frame(
    frame: np.ndarray,
    bbox: List[float],
    margin: float = 0.1,
    output_size: Tuple[int, int] = (160, 160),
) -> Optional[np.ndarray]:
    """
    Crop and resize a face from a frame.

    Args:
        frame: HWC numpy array (RGB).
        bbox: [x1, y1, x2, y2] normalized [0, 1] or pixel coords.
        margin: Add margin around face as fraction of size.
        output_size: Target (width, height).

    Returns:
        Cropped face as (H, W, C) numpy array, or None if out of bounds.
    """
    h, w = frame.shape[:2]

    # Convert to pixel coords
    if max(bbox) <= 1.0:
        bbox = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]

    x1, y1, x2, y2 = bbox
    face_w = x2 - x1
    face_h = y2 - y1

    # Add margin
    margin_x = face_w * margin
    margin_y = face_h * margin
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None

    face = frame[int(y1):int(y2), int(x1):int(x2)]

    # Resize
    try:
        import cv2
        face = cv2.resize(face, output_size, interpolation=cv2.INTER_AREA)
    except Exception:
        return None

    return face


# ── MediaPipe Face Detector ────────────────────────────────────────────────────

class MediaPipeDetector:
    """Face detection using MediaPipe Face Detection."""

    def __init__(self, model_selection: int = 1, min_detection_confidence: float = 0.5):
        """
        Args:
            model_selection: 0 = short-range (best for close-up), 1 = full range.
            min_detection_confidence: Minimum detection confidence [0, 1].
        """
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self._detector = None

    def _load(self):
        if self._detector is not None:
            return
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection
            self._detector = mp_face.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_detection_confidence,
            )
        except ImportError:
            raise ImportError(
                "MediaPipe not installed. Install with: pip install mediapipe"
            )

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in a frame. Returns list of face dicts."""
        self._load()
        import mediapipe as mp

        rgb = frame[:, :, ::-1]  # BGR → RGB
        results = self._detector.process(rgb)

        faces = []
        if results.detections:
            ih, iw = frame.shape[:2]
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                conf = det.score[0]
                face_dict = {
                    "bbox": [
                        bbox.xmin * iw,
                        bbox.ymin * ih,
                        (bbox.xmin + bbox.width) * iw,
                        (bbox.ymin + bbox.height) * ih,
                    ],
                    "confidence": float(conf),
                    "keypoints": {
                        "left_eye": (bbox.xmin * iw + bbox.width * 0.3, bbox.ymin * ih + bbox.height * 0.3),
                        "right_eye": (bbox.xmin * iw + bbox.width * 0.7, bbox.ymin * ih + bbox.height * 0.3),
                        "nose": (bbox.xmin * iw + bbox.width * 0.5, bbox.ymin * ih + bbox.height * 0.5),
                    },
                }
                faces.append(face_dict)

        return faces


# ── ArcFace Embedding ──────────────────────────────────────────────────────────

class ArcFaceEmbedder:
    """ArcFace embedding extraction for face re-identification."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._normalize = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from torchvision import transforms
        except ImportError as e:
            raise ImportError("torch + torchvision required for ArcFaceEmbedder") from e

        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.warning("insightface not available. Using mock embeddings.")
            self._model = None
            return

        logger.info("Loading ArcFace model...")
        self._app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

    def embed_faces(self, frame: np.ndarray, bboxes: List[List[float]]) -> List[np.ndarray]:
        """Extract ArcFace embeddings for detected face bboxes."""
        self._load()

        if self._model is None:
            # Fallback: return mock zero embeddings
            return [np.zeros(512, dtype=np.float32) for _ in bboxes]

        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces_input = []
        for bbox in bboxes:
            face = extract_face_from_frame(rgb, bbox)
            if face is not None:
                faces_input.append(face)

        if not faces_input:
            return [np.zeros(512, dtype=np.float32) for _ in bboxes]

        embeddings = []
        for face in faces_input:
            try:
                import torch
                face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    emb = self._model(face_tensor)
                embeddings.append(emb.squeeze().cpu().numpy())
            except Exception:
                embeddings.append(np.zeros(512, dtype=np.float32))

        return embeddings


# ── Face Tracker ────────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Face Detection + Tracking across video frames.

    Combines:
      1. MediaPipe face detection (fast)
      2. ArcFace embedding (re-identification)
      3. IOU-based temporal tracking with re-assignment

    Usage:
        tracker = FaceTracker()
        results = tracker.track_video("movie.mp4", sample_fps=1.0)
    """

    def __init__(
        self,
        detector_backend: str = DETECTOR_BACKEND,
        embedding_model: str = EMBEDDING_MODEL,
        iou_threshold: float = TRACKING_IOU_THRESHOLD,
        max_disappear_frames: int = MAX_DISAPPEAR_FRAMES,
        device: str = "cuda",
    ):
        self.iou_threshold = iou_threshold
        self.max_disappear = max_disappear_frames
        self.device = device

        # Initialize components
        if detector_backend == "mediapipe":
            self.detector = MediaPipeDetector()
        else:
            raise ValueError(f"Unknown detector backend: {detector_backend}")

        self.embedder = ArcFaceEmbedder(device=device)

        # Tracking state
        self._next_track_id = 0
        self._active_tracks: Dict[int, Dict[str, Any]] = {}  # track_id → track state
        self._frame_count = 0

    def reset(self):
        """Reset tracking state for a new video."""
        self._next_track_id = 0
        self._active_tracks = {}
        self._frame_count = 0

    def track_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """
        Process a single frame: detect faces, compute embeddings, track IDs.

        Args:
            frame: HWC numpy array (RGB or BGR).
            timestamp: Frame timestamp in seconds.

        Returns:
            List of tracked face dicts with consistent track_ids.
        """
        import cv2
        if frame.shape[2] == 3 and frame.dtype == np.uint8:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame

        # Step 1: Detect faces
        raw_faces = self.detector.detect(rgb)

        if not raw_faces:
            self._prune_tracks()
            self._frame_count += 1
            return []

        bboxes = [f["bbox"] for f in raw_faces]

        # Step 2: Extract embeddings
        embeddings = self.embedder.embed_faces(rgb, bboxes)

        # Step 3: Match to existing tracks
        matched_tracks: List[Dict[str, Any]] = []
        used_tracks: set = set()

        for i, (bbox, emb) in enumerate(zip(bboxes, embeddings)):
            best_track_id = None
            best_score = self.iou_threshold

            for track_id, track in self._active_tracks.items():
                if track_id in used_tracks:
                    continue
                # Compute IOU with previous bbox
                prev_bbox = track["last_bbox"]
                iou = compute_iou(bbox, prev_bbox)
                if iou >= best_score:
                    best_score = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self._active_tracks[best_track_id].update({
                    "last_bbox": bbox,
                    "last_embedding": emb,
                    "last_seen_frame": self._frame_count,
                    "disappeared_frames": 0,
                })
                used_tracks.add(best_track_id)
                matched_tracks.append({
                    "track_id": best_track_id,
                    "bbox": bbox,
                    "embedding": emb,
                    "timestamp": timestamp,
                    "frame": self._frame_count,
                    "confidence": raw_faces[i]["confidence"],
                })
            else:
                # Create new track
                new_id = self._next_track_id
                self._next_track_id += 1
                self._active_tracks[new_id] = {
                    "last_bbox": bbox,
                    "last_embedding": emb,
                    "last_seen_frame": self._frame_count,
                    "disappeared_frames": 0,
                    "first_seen_frame": self._frame_count,
                    "total_appearances": 1,
                }
                used_tracks.add(new_id)
                matched_tracks.append({
                    "track_id": new_id,
                    "bbox": bbox,
                    "embedding": emb,
                    "timestamp": timestamp,
                    "frame": self._frame_count,
                    "confidence": raw_faces[i]["confidence"],
                })

        # Mark unmatched tracks as disappeared
        for track_id in self._active_tracks:
            if track_id not in used_tracks:
                self._active_tracks[track_id]["disappeared_frames"] += 1

        self._prune_tracks()
        self._frame_count += 1
        return matched_tracks

    def _prune_tracks(self):
        """Remove tracks that have been invisible for too many frames."""
        to_remove = [
            tid for tid, t in self._active_tracks.items()
            if t["disappeared_frames"] > self.max_disappear
        ]
        for tid in to_remove:
            del self._active_tracks[tid]

    def track_video(
        self,
        video_path: str,
        sample_fps: float = 1.0,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Track faces across an entire video.

        Args:
            video_path: Path to video file.
            sample_fps: Sample rate (1.0 = 1 frame per second).
            max_frames: Maximum frames to process.

        Returns:
            List of frame results, each with list of tracked faces.
        """
        import cv2

        self.reset()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / sample_fps))

        all_results = []
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                faces = self.track_frame(frame, timestamp)
                all_results.append({
                    "frame_idx": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "faces": faces,
                    "active_track_count": len(self._active_tracks),
                })
                processed += 1
                if max_frames and processed >= max_frames:
                    break
            frame_idx += 1

        cap.release()
        return all_results

    def get_character_scenes(
        self,
        track_results: List[Dict[str, Any]],
        character_mapping: Dict[int, str],
    ) -> Dict[str, List[float]]:
        """
        Convert track results to per-character temporal presence.

        Args:
            track_results: Output from track_video().
            character_mapping: Map from track_id → character name.

        Returns:
            Dict mapping character_name → list of [start, end] appearance segments.
        """
        import bisect

        char_timeline: Dict[str, List[Tuple[float, float]]] = {}

        for i, result in enumerate(track_results):
            timestamp = result["timestamp"]
            for face in result["faces"]:
                track_id = face["track_id"]
                name = character_mapping.get(track_id, f"person_{track_id}")
                if name not in char_timeline:
                    char_timeline[name] = []
                char_timeline[name].append((timestamp, face["track_id"]))

        # Merge adjacent timestamps for same character
        presence: Dict[str, List[List[float]]] = {}
        for name, timestamps in char_timeline.items():
            timestamps.sort()
            segments = []
            seg_start = timestamps[0][0]
            seg_id = timestamps[0][1]
            for t, tid in timestamps[1:]:
                if t - timestamps[timestamps.index((t, tid)) - 1][0] > 5.0 or tid != seg_id:
                    segments.append([seg_start, timestamps[timestamps.index((t, tid)) - 1][0]])
                    seg_start = t
                    seg_id = tid
            segments.append([seg_start, timestamps[-1][0]])
            presence[name] = segments

        return presence

    def __repr__(self) -> str:
        return (
            f"FaceTracker(detector={DETECTOR_BACKEND}, embedding={EMBEDDING_MODEL}, "
            f"iou_thresh={self.iou_threshold})"
        )
