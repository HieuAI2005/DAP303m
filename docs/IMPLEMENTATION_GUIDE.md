# 🚀 Hướng Dẫn Triển Khai: Video Understanding Pipeline

## 1. Tổng Quan Lộ Trình

```
Tuần 1-2: Core Infrastructure
├── 1.1 Hoàn thiện Scene Index (L1)
├── 1.2 Tích hợp Whisper STT
├── 1.3 VLM Scene Understanding
└── 1.4 Temporal Grounding baseline

Tuần 3-4: Multi-modal Understanding
├── 2.1 Action Recognition (Charades)
├── 2.2 Face Detection + Character Tracking
├── 2.3 Cross-encoder Reranking
└── 2.4 VQAv2-style benchmarks

Tuần 5-6: Knowledge Integration
├── 3.1 Neo4j GraphRAG hoàn chỉnh
├── 3.2 Script-Scene Alignment
├── 3.3 Multi-hop Reasoning
└── 3.4 Narrative QA evaluation

Tuần 7-8: Advanced Features + Final
├── 4.1 Video Captioning
├── 4.2 Causal Reasoning
├── 4.3 Full benchmark evaluation
└── 4.4 Report + Demo
```

---

## 2. Phase 1: Core Infrastructure (Tuần 1-2)

### 2.1 Step 1.1: Hoàn thiện Scene Index (L1)

**Mục tiêu:** Di chuyển các method từ `project_ky4/visual_indexer.py` sang `VideoRag`

**File cần cập nhật:** `VideoRag/src/movierag/indexing/visual_indexer.py`

```python
# Thêm vào class VisualIndexer trong VideoRag:

# 1. Thêm scene_index_path và scene_metadata_path vào __init__
self.scene_index_path = self.index_dir / f"{index_name}_scenes.faiss"
self.scene_metadata_path = self.index_dir / f"{index_name}_scenes_meta.json"

# 2. Thêm các biến cho scene weighting
scene_image_weight = float(os.getenv("MOVIERAG_SCENE_IMAGE_WEIGHT", "0.72"))
scene_text_weight = float(os.getenv("MOVIERAG_SCENE_TEXT_WEIGHT", "0.28"))
self.scene_image_weight = scene_image_weight / (scene_image_weight + scene_text_weight)
self.scene_text_weight = scene_text_weight / (scene_image_weight + scene_text_weight)

# 3. Thêm method _aggregate_scene_metadata (từ project_ky4)
def _aggregate_scene_metadata(self, scene_group_id, indices):
    """Gộp metadata từ nhiều frames thành 1 scene metadata."""
    # Chi tiết: xem project_ky4/src/movierag/indexing/visual_indexer.py
    # dòng 192-336
    pass

# 4. Thêm method _compose_scene_text
def _compose_scene_text(self, metadata):
    """Tạo scene text tổng hợp từ 5-layer metadata."""
    # Chi tiết: xem project_ky4/src/movierag/indexing/visual_indexer.py
    # dòng 338-363
    pass

# 5. Thêm method _build_scene_index
def _build_scene_index(self, embeddings):
    """Build Scene FAISS Index (L1) từ frame embeddings."""
    # 1. Group frames by scene_id
    # 2. Mean pool embeddings per scene
    # 3. Fuse with CLIP text embeddings (72/28)
    # 4. Build FAISS IndexFlatIP
    pass

# 6. Cập nhật build_index() để gọi _build_scene_index
def build_index(self, items, id_key="keyframe_id", path_key="keyframe_path", movie_id_key="movie_id"):
    # ... existing code ...
    # Sau khi build frame index, gọi:
    self._build_scene_index(embeddings.astype(np.float32))
    self.save()  # Lưu cả scene index
```

**Validation:**
```bash
# Test scene index build
python -c "
from movierag.indexing.visual_indexer import VisualIndexer
idx = VisualIndexer('data/indexes', 'test_scene_index')
# Build với sample data
sample_items = [
    {'keyframe_id': 'kf1', 'keyframe_path': 'frame1.jpg', 'movie_id': 'tt001', 'scene_id': 'scene1'},
    {'keyframe_id': 'kf2', 'keyframe_path': 'frame2.jpg', 'movie_id': 'tt001', 'scene_id': 'scene1'},
]
idx.build_index(sample_items)
stats = idx.get_statistics()
print(f'Frame vectors: {stats[\"num_vectors\"]}')
print(f'Scene vectors: {stats[\"num_scene_vectors\"]}')
"
```

---

### 2.2 Step 1.2: Tích hợp Whisper STT Pipeline

**Mục tiêu:** Thêm transcription pipeline cho raw video

**File mới:** `VideoRag/src/movierag/preprocessing/whisper_transcriber.py`

```python
"""
Whisper Transcription Pipeline for Video Understanding
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    Whisper-based speech-to-text transcription pipeline.
    """

    def __init__(
        self,
        model_name: str = "medium",
        language: Optional[str] = None,
        output_dir: str = "data/transcripts"
    ):
        """
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
            language: Target language (None = auto-detect)
            output_dir: Directory for transcript outputs
        """
        self.model_name = model_name
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load whisper
        self._model = None

    def _load_model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_name)
            logger.info(f"Loaded Whisper model: {self.model_name}")

    def transcribe_video(
        self,
        video_path: str,
        video_id: str,
        task: str = "transcribe",
        word_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe video audio to text with timestamps.

        Returns:
            {
                "video_id": str,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.5,
                        "text": "Hello, how are you?",
                        "words": [
                            {"word": "Hello,", "start": 0.0, "end": 0.4},
                            ...
                        ]
                    },
                    ...
                ],
                "full_text": str
            }
        """
        self._load_model()

        logger.info(f"Transcribing video: {video_path}")

        # Run transcription
        result = self._model.transcribe(
            video_path,
            task=task,
            language=self.language,
            word_timestamps=word_timestamps,
            verbose=False
        )

        # Format output
        segments = []
        for seg in result.get("segments", []):
            segment_data = {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip()
            }

            # Include word-level timestamps if available
            if word_timestamps and "words" in seg:
                segment_data["words"] = [
                    {"word": w["word"], "start": w["start"], "end": w["end"]}
                    for w in seg["words"]
                ]

            segments.append(segment_data)

        output = {
            "video_id": video_id,
            "language": result.get("language", "unknown"),
            "segments": segments,
            "full_text": " ".join(seg["text"] for seg in segments)
        }

        # Save to disk
        output_path = self.output_dir / f"{video_id}_transcript.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved transcript to: {output_path}")

        return output

    def transcribe_batch(
        self,
        video_paths: List[str],
        video_ids: List[str]
    ) -> List[Dict]:
        """Transcribe multiple videos in batch."""
        results = []
        for video_path, video_id in zip(video_paths, video_ids):
            try:
                result = self.transcribe_video(video_path, video_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe {video_id}: {e}")
                results.append({"video_id": video_id, "error": str(e)})
        return results

    def extract_dialogue_chunks(
        self,
        transcript: Dict,
        chunk_duration: float = 30.0,
        overlap: float = 1.0
    ) -> List[Dict]:
        """
        Split transcript into dialogue chunks for indexing.

        Args:
            transcript: Output from transcribe_video()
            chunk_duration: Max duration per chunk (seconds)
            overlap: Overlap between chunks (seconds)

        Returns:
            List of dialogue chunks with timestamps
        """
        chunks = []
        current_start = 0.0
        current_text = []
        current_words = []

        for segment in transcript["segments"]:
            # If adding this segment exceeds duration, save current chunk
            if (segment["end"] - current_start > chunk_duration) and current_text:
                chunks.append({
                    "start": current_start,
                    "end": segment["start"],
                    "text": " ".join(current_text),
                    "word_count": len(current_words)
                })

                # Start new chunk with overlap
                current_start = max(current_start + chunk_duration - overlap, segment["start"])
                current_text = []
                current_words = []

            current_text.append(segment["text"])
            if "words" in segment:
                current_words.extend(segment["words"])

        # Don't forget the last chunk
        if current_text:
            last_seg = transcript["segments"][-1]
            chunks.append({
                "start": current_start,
                "end": last_seg["end"],
                "text": " ".join(current_text),
                "word_count": len(current_words)
            })

        return chunks
```

**Integration vào Preprocessing Pipeline:**

```python
# Trong preprocess_data/pipeline.py, thêm:

from movierag.preprocessing.whisper_transcriber import WhisperTranscriber

def process_video_with_whisper(video_path, video_id, output_dir):
    """Process video with full STT pipeline."""

    # 1. Whisper Transcription
    transcriber = WhisperTranscriber(model_name="medium")
    transcript = transcriber.transcribe_video(video_path, video_id)

    # 2. Extract dialogue chunks
    dialogue_chunks = transcriber.extract_dialogue_chunks(transcript)

    # 3. Align with existing subtitle/SRT if available
    # (Implement SRT alignment here)

    # 4. Return for indexing
    return {
        "transcript": transcript,
        "dialogue_chunks": dialogue_chunks
    }
```

---

### 2.3 Step 1.3: VLM Scene Understanding Module

**Mục tiêu:** Tạo module VLM để phân tích frame và tạo scene description

**File mới:** `VideoRag/src/movierag/preprocessing/vlm_scene_analyzer.py`

```python
"""
VLM Scene Understanding Module
Uses Vision Language Model for deep scene analysis
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)


class VLMAnalyzer:
    """
    Vision Language Model for scene understanding.
    Supports: Qwen2-VL, LLaVA, GPT-4V
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        cache_dir: str = "data/vlm_cache"
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load VLM."""
        if self._model is None:
            if "qwen" in self.model_name.lower():
                from transformers import Qwen2VLForConditionalGeneration
                from qwen_vl_utils import process_vision_info

                self._processor = AutoProcessor.from_pretrained(self.model_name)
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
            elif "llava" in self.model_name.lower():
                from llava.model.builder import load_pretrained_model
                self._model, _, _ = load_pretrained_model(
                    model_name=self.model_name,
                    load_8bit=False,
                    load_4bit=False,
                    device_map=self.device
                )
            logger.info(f"Loaded VLM: {self.model_name}")

    def analyze_frame(
        self,
        image_path: Union[str, Image.Image],
        prompt: str = None
    ) -> Dict:
        """
        Analyze a single frame and return structured description.

        Returns:
            {
                "characters": [...],
                "setting": "...",
                "actions": [...],
                "emotions": "...",
                "objects": [...],
                "confidence": 0.85
            }
        """
        self._load_model()

        if prompt is None:
            prompt = """Analyze this movie frame carefully. Provide:
1. CHARACTERS: Who is visible? Describe appearance, clothing, expressions.
2. SETTING: Where is this scene? Time of day, location type.
3. ACTIONS: What is happening?
4. EMOTIONS: What is the emotional tone?
5. NOTABLE OBJECTS: What objects are important?
6. CAMERA: How is this shot framed?

Respond as JSON."""

        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path

        # Call VLM
        if "qwen" in self.model_name.lower():
            return self._analyze_qwen(image, prompt)
        elif "llava" in self.model_name.lower():
            return self._analyze_llava(image, prompt)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _analyze_qwen(self, image: Image.Image, prompt: str) -> Dict:
        """Analyze with Qwen2-VL."""
        # Implementation for Qwen2-VL
        # See: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
        pass

    def _analyze_llava(self, image: Image.Image, prompt: str) -> Dict:
        """Analyze with LLaVA."""
        # Implementation for LLaVA
        # See: https://github.com/haotian-liu/LLaVA
        pass

    def analyze_video_frames(
        self,
        frame_paths: List[str],
        video_id: str,
        max_frames: int = 16
    ) -> List[Dict]:
        """
        Analyze multiple frames from a video.

        Samples frames uniformly and describes each.
        """
        # Uniform sampling
        if len(frame_paths) > max_frames:
            indices = [int(i * len(frame_paths) / max_frames) for i in range(max_frames)]
            frame_paths = [frame_paths[i] for i in indices]

        results = []
        for i, frame_path in enumerate(frame_paths):
            try:
                analysis = self.analyze_frame(frame_path)
                analysis["frame_index"] = i
                analysis["frame_path"] = frame_path
                results.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze frame {frame_path}: {e}")

        return results

    def fuse_scene_description(
        self,
        frame_analyses: List[Dict],
        video_id: str
    ) -> Dict:
        """
        Fuse multiple frame analyses into a coherent scene description.

        Uses LLM to merge and reconcile frame-level descriptions.
        """
        # Create fusion prompt
        fusion_prompt = f"""You are analyzing {len(frame_analyses)} frames from the same video scene.
Merge these frame descriptions into a single coherent scene description.

Frames:
"""

        for i, analysis in enumerate(frame_analyses):
            fusion_prompt += f"\n--- Frame {i+1} ---\n"
            fusion_prompt += f"Setting: {analysis.get('setting', 'N/A')}\n"
            fusion_prompt += f"Actions: {', '.join(analysis.get('actions', []))}\n"
            fusion_prompt += f"Characters: {', '.join(analysis.get('characters', []))}\n"
            fusion_prompt += f"Emotions: {analysis.get('emotions', 'N/A')}\n"

        fusion_prompt += """
Task:
1. Identify the MAIN ACTION that spans these frames
2. List all CHARACTERS involved
3. Determine the TIMING of key moments
4. Generate a COHERENT scene description

Respond as JSON with fields:
- scene_description
- main_action
- characters
- key_moments (list of {time_estimate, description})
- continuity_notes
"""

        # Call LLM for fusion
        # (Use existing LLM client from movierag.generation)
        from movierag.generation.universal_client import UniversalLLMClient
        llm = UniversalLLMClient()

        response = llm.models.generate_content(
            model="kimi",
            contents=fusion_prompt
        )

        try:
            return json.loads(response.text)
        except:
            return {
                "scene_description": frame_analyses[0].get("setting", ""),
                "main_action": frame_analyses[0].get("actions", ["Unknown"])[0],
                "characters": frame_analyses[0].get("characters", []),
                "error": "Failed to parse JSON"
            }
```

---

### 2.4 Step 1.4: Temporal Grounding Baseline

**Mục tiêu:** Xây dựng baseline temporal grounding system

**File mới:** `VideoRag/src/movierag/understanding/temporal_grounding.py`

```python
"""
Temporal Grounding Module for Video Understanding
Finds the exact moment in video based on text query
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalSegment:
    """A temporal segment in a video."""
    start: float  # seconds
    end: float    # seconds
    video_id: str
    description: str = ""
    confidence: float = 1.0


class TemporalGroundingEngine:
    """
    Temporal grounding engine for Video Understanding.

    Resolves queries like:
    - "When does Rose first appear?"
    - "Find the scene where Jack draws Rose"
    - "What happens at 1:30:00?"
    """

    def __init__(
        self,
        knowledge_indexer=None,
        visual_indexer=None,
        graph_store=None,
        llm_client=None
    ):
        self.knowledge_indexer = knowledge_indexer
        self.visual_indexer = visual_indexer
        self.graph_store = graph_store
        self.llm_client = llm_client

    def ground(
        self,
        query: str,
        video_id: Optional[str] = None,
        candidates: int = 10
    ) -> TemporalSegment:
        """
        Find temporal segment matching the query.

        Args:
            query: Natural language query
            video_id: Optional filter to specific video
            candidates: Number of candidates to retrieve

        Returns:
            TemporalSegment with best match
        """
        # Step 1: Parse temporal expressions in query
        temporal_expr = self._parse_temporal_expression(query)

        # Step 2: Retrieve candidate segments
        if self.knowledge_indexer:
            candidates_data = self.knowledge_indexer.search(
                query, k=candidates, movie_id=video_id
            )
        else:
            candidates_data = []

        # Step 3: Score candidates
        best_segment = None
        best_score = 0.0

        for candidate in candidates_data:
            segment = self._extract_segment(candidate)

            # Score based on semantic similarity
            score = self._score_segment(query, segment, temporal_expr)

            if score > best_score:
                best_score = score
                best_segment = segment

        if best_segment:
            best_segment.confidence = best_score

        return best_segment

    def _parse_temporal_expression(self, query: str) -> Dict:
        """
        Parse temporal expressions from query.

        Returns:
            {
                "type": "first" | "last" | "during" | "at" | None,
                "anchor": "first_appearance" | "climax" | ...,
                "time_estimate": float or None,
                "duration": float or None
            }
        """
        query_lower = query.lower()

        temporal_info = {
            "type": None,
            "anchor": None,
            "time_estimate": None,
            "duration": None
        }

        # Parse temporal keywords
        first_keywords = ["first", "beginning", "opens", "starts", "introduction"]
        last_keywords = ["last", "final", "ending", "conclusion", "epilogue"]
        during_keywords = ["during", "while", "when", "midst"]

        if any(kw in query_lower for kw in first_keywords):
            temporal_info["type"] = "first"
            temporal_info["anchor"] = "first_occurrence"
        elif any(kw in query_lower for kw in last_keywords):
            temporal_info["type"] = "last"
            temporal_info["anchor"] = "final_occurrence"
        elif any(kw in query_lower for kw in during_keywords):
            temporal_info["type"] = "during"

        # Parse time expressions (e.g., "at 1:30:00")
        import re
        time_match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', query)
        if time_match:
            hours, minutes, seconds = map(int, time_match.groups())
            temporal_info["type"] = "at"
            temporal_info["time_estimate"] = hours * 3600 + minutes * 60 + seconds

        return temporal_info

    def _score_segment(
        self,
        query: str,
        segment: TemporalSegment,
        temporal_expr: Dict
    ) -> float:
        """
        Score a segment against query with temporal constraints.

        Returns:
            Score 0.0 to 1.0
        """
        base_score = 0.5  # Base score if no specific constraints

        # Semantic score (simple keyword overlap)
        query_tokens = set(query.lower().split())
        desc_tokens = set(segment.description.lower().split())
        semantic_overlap = len(query_tokens & desc_tokens) / max(len(query_tokens), 1)

        score = base_score + 0.5 * semantic_overlap

        # Temporal constraint bonus/penalty
        if temporal_expr["type"] == "first":
            # Penalize if segment is late in video (heuristic)
            if segment.start > 3600:  # After 1 hour
                score *= 0.7
        elif temporal_expr["type"] == "last":
            # Penalize if segment is early
            if segment.start < 7200:  # Before 2 hours
                score *= 0.5

        return min(score, 1.0)

    def _extract_segment(self, candidate) -> TemporalSegment:
        """Extract TemporalSegment from retrieval candidate."""
        return TemporalSegment(
            start=candidate.get("start_seconds", 0.0),
            end=candidate.get("end_seconds", 0.0),
            video_id=candidate.get("movie_id", ""),
            description=candidate.get("text", candidate.get("description", ""))
        )
```

---

## 3. Phase 2: Multi-modal Understanding (Tuần 3-4)

### 3.1 Step 2.1: Action Recognition Integration

**File mới:** `VideoRag/src/movierag/preprocessing/action_recognizer.py`

```python
"""
Action Recognition Module using VideoMAE/ActionCLIP
"""

from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ActionRecognizer:
    """
    Action recognition for video segments.
    Uses VideoMAE or ActionCLIP for classification.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base-finetuned-ucf101",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._labels = self._load_labels()

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import VideoMAEForVideoClassification

            self._model = VideoMAEForVideoClassification.from_pretrained(
                self.model_name
            ).to(self.device)

    def _load_labels(self) -> List[str]:
        """Load UCF-101 action labels."""
        return [
            "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling",
            "BalanceBeam", "BandMarching", "BaseballPitch", "Basketball",
            "BasketballDunk", "BenchPress", "Biking", "Billiards",
            # ... (full 101 labels)
        ]

    def recognize_from_frames(
        self,
        frames: List[np.ndarray],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Recognize actions from list of frames.

        Args:
            frames: List of frames as numpy arrays (H, W, C)
            top_k: Number of top predictions to return

        Returns:
            List of {label, score} dicts
        """
        self._load_model()

        import torch
        from transformers import VideoMAEImageProcessor

        # Prepare video tensor (T, H, W, C) -> (B, C, T, H, W)
        video_tensor = np.stack(frames[:16], axis=0)  # Take up to 16 frames
        video_tensor = video_tensor.transpose(3, 0, 1, 2)  # (C, T, H, W)

        processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
        inputs = processor(video_tensor, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0]
        top_indices = probs.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "label": self._labels[idx.item()],
                "score": probs[idx].item()
            })

        return results
```

---

### 3.2 Step 2.2: Face Detection + Character Tracking

**File mới:** `VideoRag/src/movierag/preprocessing/face_tracker.py`

```python
"""
Face Detection and Character Tracking Module
Uses face detection + re-identification for character tracking
"""

from typing import List, Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Face detection and tracking across video shots.

    Pipeline:
    1. Detect faces in each keyframe
    2. Extract face embeddings
    3. Cluster embeddings to track same person
    4. Match with known cast (if available)
    """

    def __init__(
        self,
        detection_model: str = "yolov8n-face",
        embedding_model: str = "buffalo_l",
        device: str = "cuda"
    ):
        self.detection_model = detection_model
        self.embedding_model = embedding_model
        self.device = device
        self._detector = None
        self._embedder = None

    def _load_detector(self):
        """Load face detector."""
        if self._detector is None:
            from facexzoo.detector import RetinaFace
            self._detector = RetinaFace(model_name=self.detection_model)

    def _load_embedder(self):
        """Load face embedder."""
        if self._embedder is None:
            from facexzoo.embedder import InsightFace
            self._embedder = InsightFace(model_name=self.embedding_model)

    def detect_faces(
        self,
        image_path: str
    ) -> List[Dict]:
        """
        Detect faces in an image.

        Returns:
            List of {bbox, embedding, confidence}
        """
        self._load_detector()
        self._load_embedder()

        # Detect
        faces = self._detector.detect(image_path)

        # Embed
        results = []
        for face in faces:
            bbox, landmarks = face["bbox"], face["landmarks"]
            embedding = self._embedder.extract(image_path, bbox)
            results.append({
                "bbox": bbox,
                "embedding": embedding,
                "confidence": face.get("score", 1.0)
            })

        return results

    def track_faces_in_video(
        self,
        keyframe_paths: List[str],
        shot_boundaries: List[int],
        known_cast: Dict[str, List[str]] = None
    ) -> Dict:
        """
        Track faces across video shots.

        Args:
            keyframe_paths: Paths to keyframes
            shot_boundaries: Frame indices where shots change
            known_cast: Optional dict of {character_name: [face_images]}

        Returns:
            {
                "tracks": [
                    {
                        "track_id": 1,
                        "character_name": "Rose",
                        "appearances": [
                            {"frame_idx": 0, "bbox": [...], "confidence": 0.95},
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        # Step 1: Detect faces in all keyframes
        all_detections = []
        for kf_path in keyframe_paths:
            detections = self.detect_faces(kf_path)
            all_detections.append(detections)

        # Step 2: Cluster faces by embedding similarity
        tracks = self._cluster_faces(all_detections)

        # Step 3: Match with known cast (if available)
        if known_cast:
            tracks = self._match_with_cast(tracks, known_cast)

        return {"tracks": tracks}

    def _cluster_faces(
        self,
        detections: List[List[Dict]]
    ) -> List[Dict]:
        """Cluster faces across frames using embedding similarity."""
        from sklearn.cluster import AgglomerativeClustering

        # Collect all embeddings with frame info
        all_embeddings = []
        for frame_idx, faces in enumerate(detections):
            for face in faces:
                all_embeddings.append({
                    "embedding": face["embedding"],
                    "frame_idx": frame_idx,
                    "bbox": face["bbox"]
                })

        if not all_embeddings:
            return []

        # Stack embeddings
        X = np.array([f["embedding"] for f in all_embeddings])

        # Cluster
        n_clusters = min(20, len(X) // 3)  # Heuristic
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(X)

        # Group by cluster
        tracks = []
        for track_id in range(n_clusters):
            indices = np.where(labels == track_id)[0]
            if len(indices) == 0:
                continue

            appearances = []
            for idx in indices:
                appearances.append({
                    "frame_idx": all_embeddings[idx]["frame_idx"],
                    "bbox": all_embeddings[idx]["bbox"]
                })

            tracks.append({
                "track_id": track_id,
                "character_name": None,  # Unknown initially
                "appearances": appearances,
                "num_appearances": len(appearances)
            })

        return tracks

    def _match_with_cast(
        self,
        tracks: List[Dict],
        known_cast: Dict[str, List[str]]
    ) -> List[Dict]:
        """Match tracks to known cast members."""
        # Get embeddings for known cast
        cast_embeddings = {}
        for char_name, face_images in known_cast.items():
            embeddings = []
            for img_path in face_images:
                faces = self.detect_faces(img_path)
                if faces:
                    embeddings.append(faces[0]["embedding"])
            if embeddings:
                cast_embeddings[char_name] = np.mean(embeddings, axis=0)

        # Match each track to cast
        for track in tracks:
            # Average embedding of this track
            track_emb = np.mean([
                all_embeddings[idx]["embedding"]
                for idx in track["appearances"]
            ], axis=0)

            # Find best match
            best_match = None
            best_score = 0.0
            for char_name, cast_emb in cast_embeddings.items():
                score = np.dot(track_emb, cast_emb)  # Cosine similarity
                if score > best_score:
                    best_score = score
                    best_match = char_name

            if best_score > 0.7:  # Threshold
                track["character_name"] = best_match
                track["match_confidence"] = best_score

        return tracks
```

---

### 3.3 Step 2.3: Cross-encoder Reranking

**Thêm vào `VideoRag/src/movierag/indexing/visual_indexer.py`:**

```python
def rerank_by_image(
    self,
    query_image: Union[str, Image.Image],
    candidates: List[SearchResult],
    preferred_movie_id: Optional[str] = None,
    movie_boost: float = 0.05,
) -> List[SearchResult]:
    """
    Cross-encoder re-ranking for image queries.

    Re-computes exact CLIP similarity between query image
    and each candidate's actual image file.
    """
    if not candidates or len(candidates) <= 1:
        return candidates

    try:
        # Encode query image once
        query_emb = self.encoder.encode_image(query_image, normalize=True)
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        reranked = []
        for r in candidates:
            candidate_path = r.path or r.metadata.get("path", "")
            new_score = r.score

            if candidate_path and Path(candidate_path).exists():
                try:
                    cand_emb = self.encoder.encode_image(candidate_path, normalize=True)
                    cand_emb = cand_emb.reshape(1, -1).astype(np.float32)
                    # Exact cosine similarity
                    new_score = float(np.dot(query_emb, cand_emb.T)[0, 0])
                except Exception:
                    pass

            # Movie preference boost
            if preferred_movie_id and r.movie_id == preferred_movie_id:
                new_score += movie_boost

            reranked.append(SearchResult(
                id=r.id,
                path=r.path,
                movie_id=r.movie_id,
                score=new_score,
                metadata=r.metadata,
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    except Exception as e:
        logger.warning(f"Rerank by image failed: {e}")
        return candidates


def rerank_by_text(
    self,
    query_text: str,
    candidates: List[SearchResult],
    preferred_movie_id: Optional[str] = None,
    movie_boost: float = 0.05,
) -> List[SearchResult]:
    """
    Cross-encoder re-ranking for text queries.

    Re-computes exact CLIP text-to-image cosine similarity.
    """
    if not candidates or len(candidates) <= 1:
        return candidates

    try:
        # Encode text query once
        text_emb = self.encoder.encode_text(query_text, normalize=True)
        text_emb = text_emb.reshape(1, -1).astype(np.float32)

        reranked = []
        for r in candidates:
            candidate_path = r.path or r.metadata.get("path", "")
            new_score = r.score

            if candidate_path and Path(candidate_path).exists():
                try:
                    img_emb = self.encoder.encode_image(candidate_path, normalize=True)
                    img_emb = img_emb.reshape(1, -1).astype(np.float32)
                    new_score = float(np.dot(text_emb, img_emb.T)[0, 0])
                except Exception:
                    pass

            if preferred_movie_id and r.movie_id == preferred_movie_id:
                new_score += movie_boost

            reranked.append(SearchResult(
                id=r.id,
                path=r.path,
                movie_id=r.movie_id,
                score=new_score,
                metadata=r.metadata,
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    except Exception as e:
        logger.warning(f"Rerank by text failed: {e}")
        return candidates
```

---

## 4. Phase 3: Knowledge Integration (Tuần 5-6)

### 4.1 Step 3.1: Neo4j GraphRAG Integration

**Cập nhật `VideoRag/src/movierag/indexing/neo4j_graph_store.py`:**

```python
"""
Neo4j Graph Store for Video Understanding
Enhanced with scene graphs and character relationships
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VideoUnderstandingGraphStore:
    """
    Neo4j-backed knowledge graph for Video Understanding.

    Node types:
    - Movie
    - Character
    - Scene
    - Event
    - Action

    Relationship types:
    - APPEARS_IN (Character → Scene)
    - INTERACTS_WITH (Character ↔ Character)
    - FOLLOWS (Scene → Scene)
    - DEPICTS (Scene → Event)
    - CAUSES (Event → Event)
    """

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        except ImportError:
            logger.warning("Neo4j not installed")
            self.driver = None

    def create_schema(self):
        """Create graph schema for Video Understanding."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Movie nodes
            session.run("""
                CREATE CONSTRAINT movie_id IF NOT EXISTS
                FOR (m:Movie) REQUIRE m.movie_id IS UNIQUE
            """)

            # Character nodes
            session.run("""
                CREATE CONSTRAINT character_name IF NOT EXISTS
                FOR (c:Character) REQUIRE c.name IS UNIQUE
            """)

            # Scene nodes
            session.run("""
                CREATE CONSTRAINT scene_id IF NOT EXISTS
                FOR (s:Scene) REQUIRE s.scene_id IS UNIQUE
            """)

            logger.info("Graph schema created")

    def insert_movie(self, movie_data: Dict):
        """Insert a movie node."""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("""
                MERGE (m:Movie {movie_id: $movie_id})
                SET m.title = $title,
                    m.year = $year,
                    m.genres = $genres
            """, **movie_data)

    def insert_scene(self, scene_data: Dict):
        """Insert a scene node with relationships."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Create scene
            session.run("""
                MERGE (s:Scene {scene_id: $scene_id})
                SET s.start_time = $start_time,
                    s.end_time = $end_time,
                    s.situation = $situation,
                    s.description = $description,
                    s.movie_id = $movie_id
            """, **scene_data)

            # Link to movie
            session.run("""
                MATCH (m:Movie {movie_id: $movie_id})
                MATCH (s:Scene {scene_id: $scene_id})
                MERGE (m)-[:HAS_SCENE]->(s)
            """, movie_id=scene_data["movie_id"], scene_id=scene_data["scene_id"])

            # Link characters
            for char in scene_data.get("characters", []):
                session.run("""
                    MERGE (c:Character {name: $char_name})
                    MERGE (s:Scene {scene_id: $scene_id})
                    MERGE (c)-[:APPEARS_IN]->(s)
                    SET c.movie_id = $movie_id
                """, char_name=char, scene_id=scene_data["scene_id"],
                    movie_id=scene_data["movie_id"])

    def query_character_timeline(
        self,
        character_name: str,
        movie_id: str
    ) -> List[Dict]:
        """Query character appearances in a movie."""
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Character {name: $char_name})-[:APPEARS_IN]->(s:Scene)-[:BELONGS_TO]->(m:Movie {movie_id: $movie_id})
                RETURN s.scene_id, s.start_time, s.end_time, s.situation, s.description
                ORDER BY s.start_time
            """, char_name=character_name, movie_id=movie_id)

            return [dict(record) for record in result]

    def query_causal_chain(
        self,
        event_description: str,
        movie_id: str
    ) -> List[Dict]:
        """Query causal chain of events."""
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:Event)-[:CAUSES]->(e2:Event)
                WHERE e1.movie_id = $movie_id
                RETURN e1.description as cause, e2.description as effect
                ORDER BY e1.start_time
            """, movie_id=movie_id)

            return [dict(record) for record in result]

    def query_similar_scenes(
        self,
        scene_description: str,
        movie_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Find scenes with similar descriptions."""
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Scene)-[:BELONGS_TO]->(m:Movie {movie_id: $movie_id})
                WHERE s.description CONTAINS $keyword
                   OR s.situation CONTAINS $keyword
                RETURN s.scene_id, s.start_time, s.end_time, s.description
                LIMIT $limit
            """, movie_id=movie_id, keyword=scene_description, limit=limit)

            return [dict(record) for record in result]

    def build_full_graph(self, movie_data: Dict, scenes: List[Dict]):
        """Build complete graph for a movie."""
        self.insert_movie(movie_data)

        for scene in scenes:
            self.insert_scene(scene)

        logger.info(f"Built graph for movie: {movie_data['movie_id']}")
```

---

### 4.2 Step 3.2: Script-Scene Alignment

**File mới:** `VideoRag/src/movierag/preprocessing/script_aligner.py`

```python
"""
Script-Scene Alignment Module
Aligns screenplay script with detected scenes
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ScriptAligner:
    """
    Align screenplay script with detected scenes.

    Script format:
    INT. TITANIC - DECK - DAY
    Rose walks along the deck, looking at the ocean.

    Action lines are uppercase.
    Dialogue is in quotes.
    Scene headings start with INT./EXT.
    """

    def __init__(self):
        self.scene_headings = []
        self.dialogue_lines = []
        self.action_lines = []

    def parse_script(self, script_path: str) -> Dict:
        """
        Parse screenplay into structured format.

        Returns:
            {
                "scenes": [...],
                "dialogues": [...],
                "metadata": {...}
            }
        """
        with open(script_path, "r", encoding="utf-8") as f:
            script_text = f.read()

        scenes = self._extract_scenes(script_text)
        dialogues = self._extract_dialogues(script_text)
        actions = self._extract_actions(script_text)

        return {
            "scenes": scenes,
            "dialogues": dialogues,
            "actions": actions,
            "num_scenes": len(scenes),
            "num_dialogues": len(dialogues)
        }

    def _extract_scenes(self, text: str) -> List[Dict]:
        """Extract scene headings and descriptions."""
        scenes = []

        # Pattern: INT./EXT. LOCATION - TIME
        scene_pattern = r'(INT\.|EXT\.)([^\n]+?)(?:\s*-\s*)?([^\n]+)?'

        for match in re.finditer(scene_pattern, text, re.IGNORECASE):
            int_ext = match.group(1)
            location = match.group(2).strip()
            time_of_day = match.group(3).strip() if match.group(3) else ""

            # Get scene description (text after heading until next scene)
            start = match.end()
            next_match = match.re.search(scene_pattern, text[start:], re.IGNORECASE)
            if next_match:
                end = start + next_match.start()
            else:
                end = len(text)

            description = text[start:end].strip()[:500]  # First 500 chars

            scenes.append({
                "heading": f"{int_ext} {location}",
                "location": location,
                "int_ext": int_ext.strip("."),
                "time_of_day": time_of_day,
                "description": description
            })

        return scenes

    def _extract_dialogues(self, text: str) -> List[Dict]:
        """Extract dialogue lines."""
        dialogues = []

        # Pattern: CHARACTER NAME
        # "Dialogue text"
        char_pattern = r'^([A-Z][A-Z\s]+)$'
        prev_char = None

        for i, line in enumerate(text.split('\n')):
            line = line.strip()

            # Check if it's a character name (all caps, short)
            if re.match(r'^[A-Z][A-Z\s]{0,20}$', line) and len(line) > 1:
                prev_char = line

            # Check if it's dialogue (in quotes)
            if prev_char and line.startswith('"') and line.endswith('"'):
                dialogues.append({
                    "character": prev_char,
                    "text": line.strip('"'),
                    "line_idx": i
                })

        return dialogues

    def _extract_actions(self, text: str) -> List[str]:
        """Extract action lines (non-dialogue, non-scene-heading)."""
        actions = []

        for line in text.split('\n'):
            line = line.strip()

            # Skip if too short, scene heading, or dialogue
            if len(line) < 10:
                continue
            if line.startswith("INT.") or line.startswith("EXT."):
                continue
            if line.startswith('"') and line.endswith('"'):
                continue

            actions.append(line)

        return actions

    def align_with_scenes(
        self,
        script_data: Dict,
        detected_scenes: List[Dict],
        video_duration: float
    ) -> List[Dict]:
        """
        Align script scenes with detected video scenes.

        Uses:
        - Scene heading (INT./EXT.)
        - Time of day
        - Scene content similarity
        """
        aligned = []

        for i, script_scene in enumerate(script_data["scenes"]):
            # Estimate temporal position
            position = i / len(script_data["scenes"])
            start_time = position * video_duration
            end_time = (i + 1) / len(script_data["scenes"]) * video_duration

            # Match with detected scene (simplified: by index for now)
            detected = detected_scenes[i] if i < len(detected_scenes) else {}

            aligned.append({
                "script_heading": script_scene["heading"],
                "script_location": script_scene["location"],
                "script_time_of_day": script_scene["time_of_day"],
                "script_description": script_scene["description"],
                "start_seconds": start_time,
                "end_seconds": end_time,
                "detected_scene_id": detected.get("scene_id"),
                "detected_situation": detected.get("situation"),
                "dialogues": self._get_scene_dialogues(
                    script_scene,
                    script_data["dialogues"]
                )
            })

        return aligned

    def _get_scene_dialogues(
        self,
        scene: Dict,
        all_dialogues: List[Dict]
    ) -> List[Dict]:
        """Get dialogues for a specific scene."""
        # Simplified: return dialogues with matching character mentions
        scene_chars = set(re.findall(r'[A-Z][a-z]+', scene["heading"]))

        scene_dialogues = []
        for dial in all_dialogues:
            if dial["character"] in scene_chars:
                scene_dialogues.append(dial)

        return scene_dialogues[:5]  # Limit to 5 per scene
```

---

## 5. Phase 4: Advanced Features + Final (Tuần 7-8)

### 5.1 Step 4.1: Video Captioning Module

**File mới:** `VideoRag/src/movierag/understanding/video_captioner.py`

```python
"""
Video Captioning Module
Generates detailed descriptions for video segments
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class VideoCaptioner:
    """
    Generate captions for video segments.

    Combines:
    - VLM frame descriptions
    - Whisper transcripts
    - Action recognition
    - Scene metadata
    """

    def __init__(
        self,
        vlm_analyzer=None,
        whisper_transcriber=None,
        action_recognizer=None,
        llm_client=None
    ):
        self.vlm_analyzer = vlm_analyzer
        self.whisper_transcriber = whisper_transcriber
        self.action_recognizer = action_recognizer
        self.llm_client = llm_client

    def generate_caption(
        self,
        video_segment_path: str,
        keyframe_paths: List[str],
        transcript_segment: Dict = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Generate comprehensive caption for a video segment.

        Returns:
            {
                "short_caption": "...",
                "detailed_caption": "...",
                "actions": [...],
                "characters": [...],
                "setting": "...",
                "emotional_tone": "..."
            }
        """
        # 1. Get VLM frame descriptions
        frame_analyses = []
        if self.vlm_analyzer:
            frame_analyses = self.vlm_analyzer.analyze_video_frames(
                keyframe_paths,
                video_id=metadata.get("video_id", "unknown") if metadata else "unknown",
                max_frames=8
            )

        # 2. Get action recognition
        actions = []
        if self.action_recognizer and len(keyframe_paths) >= 2:
            import cv2
            frames = []
            for kf_path in keyframe_paths[:16]:
                frame = cv2.imread(kf_path)
                if frame is not None:
                    frames.append(frame)
            if frames:
                actions = self.action_recognizer.recognize_from_frames(frames)

        # 3. Get dialogue context
        dialogue_context = ""
        if transcript_segment:
            dialogue_context = transcript_segment.get("text", "")

        # 4. Generate caption via LLM
        caption = self._generate_caption_llm(
            frame_analyses,
            actions,
            dialogue_context,
            metadata
        )

        return caption

    def _generate_caption_llm(
        self,
        frame_analyses: List[Dict],
        actions: List[Dict],
        dialogue_context: str,
        metadata: Dict
    ) -> Dict:
        """Use LLM to synthesize caption from analysis."""

        # Build context prompt
        context = "FRAME ANALYSIS:\n"
        for analysis in frame_analyses[:4]:
            context += f"- Setting: {analysis.get('setting', 'N/A')}\n"
            context += f"- Actions: {', '.join(analysis.get('actions', []))}\n"
            context += f"- Characters: {', '.join(analysis.get('characters', []))}\n"

        context += f"\nACTIONS DETECTED: {', '.join(a['label'] for a in actions[:3])}\n"
        context += f"\nDIALOGUE: {dialogue_context[:200] if dialogue_context else 'No dialogue'}\n"

        prompt = f"""Based on the following video analysis, generate a detailed caption.

{context}

Generate:
1. SHORT_CAPTION: One sentence summary (max 20 words)
2. DETAILED_CAPTION: 2-3 sentence description
3. SETTING: Location and environment
4. EMOTIONAL_TONE: Mood/atmosphere
5. KEY_ACTIONS: Main actions occurring

Respond as JSON."""

        response = self.llm_client.models.generate_content(
            model="kimi",
            contents=prompt
        )

        try:
            import json
            return json.loads(response.text)
        except:
            return {
                "short_caption": frame_analyses[0].get("setting", "Unknown scene"),
                "detailed_caption": context[:200],
                "error": "Failed to parse"
            }
```

---

### 5.2 Step 4.2: Causal Reasoning Module

**File mới:** `VideoRag/src/movierag/understanding/causal_reasoner.py`

```python
"""
Causal Reasoning Module
Answers "Why" and "What happened because" questions
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CausalReasoner:
    """
    Reason about causal relationships in video narratives.

    Capabilities:
    - Extract cause-effect pairs from scenes
    - Answer "Why" questions
    - Answer "What happened because" questions
    - Build narrative causal graphs
    """

    def __init__(
        self,
        graph_store=None,
        llm_client=None,
        scene_indexer=None
    ):
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.scene_indexer = scene_indexer

    def answer_why(
        self,
        question: str,
        video_id: str,
        current_scene: Optional[Dict] = None
    ) -> Dict:
        """
        Answer "Why" questions about video events.

        Example: "Why does Rose decide to let Jack go?"
        """
        # Step 1: Identify the target event
        target_event = self._identify_event(question, video_id)

        # Step 2: Query causal chains
        if self.graph_store:
            causal_antecedents = self.graph_store.query_causal_antecedents(
                target_event, video_id
            )
        else:
            causal_antecedents = []

        # Step 3: Use LLM to synthesize explanation
        explanation = self._synthesize_why_explanation(
            question,
            target_event,
            causal_antecedents,
            current_scene
        )

        return {
            "question": question,
            "target_event": target_event,
            "causes": causal_antecedents,
            "explanation": explanation
        }

    def answer_what_happened_because(
        self,
        question: str,
        video_id: str
    ) -> Dict:
        """
        Answer "What happened because of X" questions.

        Example: "What happened because Rose met Jack?"
        """
        # Step 1: Identify the cause event
        cause_event = self._identify_event(question, video_id)

        # Step 2: Query effects
        if self.graph_store:
            effects = self.graph_store.query_causal_effects(cause_event, video_id)
        else:
            effects = []

        # Step 3: Synthesize explanation
        explanation = self._synthesize_effect_explanation(
            question,
            cause_event,
            effects
        )

        return {
            "question": question,
            "cause_event": cause_event,
            "effects": effects,
            "explanation": explanation
        }

    def _identify_event(self, question: str, video_id: str) -> Dict:
        """Identify the event described in question."""
        # Use LLM to extract event from question
        prompt = f"""
Extract the main event from this question about a movie.

Question: {question}

Return the event as a brief description (1-2 sentences).
"""

        response = self.llm_client.models.generate_content(
            model="kimi",
            contents=prompt
        )

        return {
            "description": response.text.strip(),
            "video_id": video_id
        }

    def _synthesize_why_explanation(
        self,
        question: str,
        target_event: Dict,
        causes: List[Dict],
        current_scene: Optional[Dict]
    ) -> str:
        """Synthesize explanation for 'Why' question."""

        context = f"Target Event: {target_event.get('description', 'Unknown')}\n\n"

        if causes:
            context += "Causal Antecedents:\n"
            for i, cause in enumerate(causes[:3], 1):
                context += f"{i}. {cause.get('description', 'Unknown')}\n"
        else:
            context += "No explicit causal chain found.\n"

        if current_scene:
            context += f"\nCurrent Scene Context:\n{current_scene.get('description', '')}\n"

        prompt = f"""Based on the following context, explain WHY this event occurred.

{context}

Question: {question}

Provide a clear, logical explanation of the causal factors.
"""

        response = self.llm_client.models.generate_content(
            model="kimi",
            contents=prompt
        )

        return response.text.strip()

    def _synthesize_effect_explanation(
        self,
        question: str,
        cause_event: Dict,
        effects: List[Dict]
    ) -> str:
        """Synthesize explanation for effect question."""

        context = f"Cause Event: {cause_event.get('description', 'Unknown')}\n\n"

        if effects:
            context += "Consequences:\n"
            for i, effect in enumerate(effects[:5], 1):
                context += f"{i}. {effect.get('description', 'Unknown')}\n"
        else:
            context += "No explicit consequences found in the graph.\n"

        prompt = f"""Based on the following, explain WHAT HAPPENED as a result.

{context}

Question: {question}

Provide a narrative explanation of the consequences.
"""

        response = self.llm_client.models.generate_content(
            model="kimi",
            contents=prompt
        )

        return response.text.strip()
```

---

## 6. Integration Checklist

### 6.1 Checklist: VideoRag Updates

```
□ visual_indexer.py
  □ _build_scene_index()
  □ _aggregate_scene_metadata()
  □ _compose_scene_text()
  □ search_scene_by_text()
  □ hierarchical_search()
  □ hybrid_search()
  □ rerank_by_image()
  □ rerank_by_text()
  □ extract_video_clip()

□ agentic_pipeline.py
  □ VLM multi-frame analysis (Step 2.7)
  □ VLM-FAISS conflict detection
  □ VLM query distillation
  □ LLM Context Booster
  □ Script Scene retrieval
  □ Graph Context retrieval
  □ 5-layer metadata enrichment
  □ Tool-calling JudgeAgent

□ preprocessing/
  □ whisper_transcriber.py (NEW)
  □ vlm_scene_analyzer.py (NEW)
  □ face_tracker.py (NEW)
  □ script_aligner.py (NEW)
  □ action_recognizer.py (NEW)

□ understanding/
  □ temporal_grounding.py (NEW)
  □ video_captioner.py (NEW)
  □ causal_reasoner.py (NEW)

□ indexing/
  □ neo4j_graph_store.py (update)
  □ knowledge_indexer.py (update)
  □ script_scene_indexer.py (NEW)
```

### 6.2 Testing Protocol

```bash
# Test 1: Scene Index Build
python -c "
from movierag.indexing.visual_indexer import VisualIndexer
idx = VisualIndexer('data/indexes', 'test_scene')
idx.build_index([...])  # sample items
print(idx.get_statistics())
"

# Test 2: VLM Analysis
python -c "
from movierag.preprocessing.vlm_scene_analyzer import VLMAnalyzer
analyzer = VLMAnalyzer()
result = analyzer.analyze_frame('test_frame.jpg')
print(result)
"

# Test 3: Temporal Grounding
python -c "
from movierag.understanding.temporal_grounding import TemporalGroundingEngine
engine = TemporalGroundingEngine(...)
result = engine.ground('When does Jack draw Rose?', 'tt0120338')
print(result)
"

# Test 4: Full Pipeline
python -c "
from movierag.pipeline.agentic_pipeline import AgenticVideoRAGPipeline
pipeline = AgenticVideoRAGPipeline(...)
result = pipeline.respond(
    query='Describe the scene where Jack draws Rose',
    image_path='query_image.jpg'
)
print(result['answer'])
"
```

---

## 7. Configuration Files

### 7.1 Environment Variables

```bash
# .env.example

# Models
CLIP_MODEL=ViT-L/14
VLM_MODEL=Qwen/Qwen2-VL-7B-Instruct
WHISPER_MODEL=medium

# Indexes
INDEX_DIR=data/indexes
FAISS_L0_PATH=data/indexes/frame_index.faiss
FAISS_L1_PATH=data/indexes/scene_index.faiss
FAISS_KNOWLEDGE_PATH=data/indexes/knowledge_index.faiss

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM
LLM_PROVIDER=openai  # or anthropic, groq
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# VLM Settings
VLM_CACHE_DIR=data/vlm_cache
VLM_MAX_FRAMES=16

# Retrieval
MOVIERAG_VISUAL_SEARCH_STRATEGY=hierarchical
MOVIERAG_SCENE_IMAGE_WEIGHT=0.72
MOVIERAG_SCENE_TEXT_WEIGHT=0.28
MOVIERAG_SCENE_CLUSTER_VISUAL_WEIGHT=0.6
MOVIERAG_SCENE_CLUSTER_SCRIPT_WEIGHT=0.4

# Whisper
WHISPER_OUTPUT_DIR=data/transcripts
```

### 7.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  movierag:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/root/.cache
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    depends_on:
      - neo4j
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
    deploy:
      resources:
        limits:
          memory: 4G

volumes:
  neo4j_data:
```
