# 📊 Dataset Documentation: Video Understanding

## 1. Tổng Quan Dataset Portfolio

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATASET PORTFOLIO FOR VIDEO UNDERSTANDING                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    INTERNAL DATASETS (Đã có)                          │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │   │
│  │  │ ActivityNet   │ │   MovieNet    │ │ MovieGraphs   │               │   │
│  │  │  Captions    │ │  (subset)    │ │              │               │   │
│  │  └───────────────┘ └───────────────┘ └───────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EXTERNAL DATASETS (Cần thu thập)                    │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────┐  │   │
│  │  │   MSR-VTT    │ │   LSMDC      │ │  Charades    │ │ CinePile │  │   │
│  │  │              │ │              │ │              │ │          │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    RAW VIDEO SOURCES (Cần download)                   │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │   │
│  │  │ ActivityNet   │ │   YouTube    │ │   IMDb Top   │               │   │
│  │  │   Videos     │ │   Trailers   │ │     250      │               │   │
│  │  │ (787/1000)  │ │  (191 done)  │ │ (226 target) │               │   │
│  │  └───────────────┘ └───────────────┘ └───────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Internal Datasets (Chi Tiết)

### 2.1 ActivityNet Captions

**Mô tả:** Dataset chuẩn cho bài toán dense video captioning và temporal localization.

| Thuộc tính | Giá trị |
|---|---|
| **Số lượng video** | 19,803 |
| **Tổng số segments** | 71,957 |
| **Training** | 37,421 segments |
| **Validation 1** | 17,505 segments |
| **Validation 2** | 17,031 segments |
| **Nguồn** | YouTube (activity videos) |
| **Annotation** | Dense temporal captions với timestamps |

**File Format:**
```json
{
    "v_4Lu8ECLHvK4": {
        "duration": 114.64,
        "sentences": [
            "A woman is seen speaking to camera...",
            "She demonstrates proper form..."
        ],
        "timestamps": [
            [1.15, 64.77],
            [38.4, 105.47]
        ]
    }
}
```

**Vai trò trong hệ thống:**
- ✅ Knowledge Index (189,833 vectors): text embeddings của captions
- ✅ Temporal Grounding evaluation
- ✅ Text-to-video retrieval benchmark

**Location:** `dataset/ActivityNet_Captions/`

---

### 2.2 MovieNet (Subset)

**Mô tả:** Hollywood movie dataset với shot-level annotations.

| Thuộc tính | Giá trị |
|---|---|
| **Số phim** | 19 movies |
| **Số shots** | 14,397 shots |
| **Keyframes** | ~33,021 frames |
| **Annotations** | Shot boundaries, subtitles, metadata |
| **Nguồn** | MovieNet benchmark (ECCV 2020) |

**Vai trò trong hệ thống:**
- ✅ Visual Index (keyframe embeddings)
- ✅ Scene segmentation ground truth
- ✅ Shot boundary detection evaluation

**Data Structure:**
```
movie_data_subset_20/
├── tt0120338/              # Titanic
│   ├── shots/
│   │   ├── shot_0001.jpg
│   │   ├── shot_0002.jpg
│   │   └── ...
│   ├── meta/
│   │   └── metadata.json
│   └── subtitles/
│       └── subtitles.srt
└── ...
```

---

### 2.3 MovieGraphs

**Mô tả:** Scene graph annotations cho 52 phim Hollywood.

| Thuộc tính | Giá trị |
|---|---|
| **Số phim** | 52 movies |
| **Số clips** | 7,761 clips |
| **Annotation type** | Scene graphs (situation, characters, interactions) |
| **Graph entities** | Characters, objects, locations, actions |
| **Relationship types** | `APPEARS_IN`, `INTERACTS_WITH`, `LOCATED_AT`, etc. |

**Scene Graph Example:**
```json
{
    "clip_id": "titanic_001",
    "movie": "Titanic",
    "situation": "arguing",
    "characters": [
        {"name": "Rose", "emotion": "angry"},
        {"name": "Cal", "emotion": "frustrated"}
    ],
    "interactions": [
        {"from": "Rose", "to": "Cal", "type": "arguing"},
        {"from": "Rose", "to": "Jack", "type": "glancing"}
    ],
    "location": "Dining room"
}
```

**Vai trò trong hệ thống:**
- ✅ Knowledge Graph (Neo4j) source
- ✅ Character entity tracking
- ✅ Scene understanding ground truth

---

## 3. External Datasets (Cần Thu Thập)

### 3.1 MSR-VTT (Microsoft Research Video to Text)

**Mô tả:** Video captioning benchmark với 10K video clips.

| Thuộc tính | Giá trị |
|---|---|
| **Số lượng video** | 10,000 |
| **Số captions/video** | 20 captions each |
| **Tổng captions** | 200,000 |
| **Duration/clip** | 10-30 seconds |
| **Nguồn** | Microsoft Research |

**Benchmark Tasks:**
- Text-to-Video Retrieval (R@K, MRR)
- Video Captioning (BLEU, CIDEr, SPICE)

**Download:** `https://github.com/AlexZhang-Blocked/MSR-VTT`

---

### 3.2 LSMDC (Large Scale Movie Description Challenge)

**Mô tả:** Movie captioning từ Hollywood movies.

| Thuộc tính | Giá trị |
|---|---|
| **Số phim** | 202 movies |
| **Số clips** | 118,081 clips |
| **Clip duration** | 2-30 seconds |
| **Source** | 15 Hollywood movies (same as MTor)

**Benchmark Tasks:**
- Video Captioning (CIDEr, SPICE)
- Movie-specific understanding

**Download:** `https://sites.google.com/site-describingmovies/`

---

### 3.3 Charades

**Mô tả:** Indoor activity recognition dataset.

| Thuộc tính | Giá trị |
|---|---|
| **Số video** | 9,848 |
| **Số action classes** | 157 |
| **Annotations** | Multi-label activity classification |
| **Source** | RGB-D indoor videos |

**Use Case:**
- Action recognition training/evaluation
- Indoor scene understanding

**Download:** `https://prior.allenai.org/projects/charades`

---

### 3.4 CinePile

**Mô tả:** Large-scale movie-domain video-language dataset.

| Thuộc tính | Giá trị |
|---|---|
| **Số video clips** | ~300,000 |
| **Domain** | Movie-specific |
| **Annotation** | Video-text pairs (captions, descriptions) |
| **Use** | Video-language pretraining, movie domain adaptation |

**Use Case:**
- Domain-specific pretraining cho CLIP/VLM
- Movie visual-semantic alignment

---

### 3.5 DiDeMo (Distinct Descriptions in Movies)

**Mô tả:** Moment localization benchmark.

| Thuộc tính | Giá trị |
|---|---|
| **Số video** | 10,761 |
| **Số descriptions** | 40,579 |
| **Task** | Localize natural language moment queries |
| **Annotations** | Start/end timestamps for each description |

**Benchmark Tasks:**
- Temporal Grounding (IoU@0.5, R@1)
- Moment retrieval

**Download:** `https://github.com/LisaAnne/LocalizingMoments`

---

### 3.6 YouCook2

**Mô tả:** Instructional cooking video dataset.

| Thuộc tính | Giá trị |
|---|---|
| **Số video** | 89 recipes |
| **Segments** | 22,670 |
| **Task** | Segment captioning, recipe understanding |

**Use Case:**
- Instructional video understanding
- Procedural reasoning

---

## 4. Dataset Collection Pipeline

### 4.1 Raw Video Download Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RAW VIDEO ACQUISITION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Source 1: ActivityNet Videos                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Status: 787/1000 downloaded                                                │
│  Tool: yt-dlp                                                              │
│  Command: yt-dlp --download-sections "*0-300" -f worst ...                 │
│  Target: 213 remaining videos                                               │
│                                                                              │
│  Source 2: MovieNet Raw Videos                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Status: Not downloaded                                                     │
│  Source: Academic licensing required                                        │
│  Alternative: Streaming capture from approved sources                       │
│                                                                              │
│  Source 3: YouTube Movie Trailers                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Status: 191/226 downloaded (IMDb Top 250)                                  │
│  Command: yt-dlp "SEARCH_QUERY" -o "trailer.mp4"                           │
│  Target: 35 remaining trailers                                              │
│                                                                              │
│  Source 4: External Dataset Videos                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  MSR-VTT: Direct download (MP4 links provided)                              │
│  Charades: Direct download (ZIP archives)                                   │
│  LSMDC: Academic request + agreement                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Processing Pipeline

```
DOWNLOADED VIDEO (.mp4)
         │
         ▼
┌─────────────────────────────────┐
│  Step 1: Shot Detection         │  PySceneDetect
│  Output: Shot boundaries         │  ffmpeg -scenechange
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Step 2: Keyframe Extraction    │  FFmpeg
│  Output: 1 FPS keyframes        │  Diverse sampling
└──────────────┬──────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│  CLIP   │ │   VLM   │ │  Face  │
│ Embed   │ │ Scene   │ │ Detect │
│ (L0)    │ │ Desc    │ │        │
└────┬────┘ └────┬────┘ └───┬────┘
     │          │          │
     ▼          ▼          ▼
┌─────────────────────────────────┐
│  Step 3: Whisper STT           │  Whisper (medium)
│  Output: Timestamped transcript │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Step 4: Semantic Scene Seg.   │  LLM-based
│  Output: Scene boundaries       │  VLM + Script
└──────────────┬──────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Script  │ │ Dialog  │ │ Action  │
│ Align   │ │ Extract │ │ Labels  │
└────┬────┘ └────┬────┘ └────┬────┘
     │          │          │
     └──────────┼──────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Step 5: 5-Layer Chunk Builder │
│  Output: Temporal chunks JSON    │
└──────────────┬──────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ FAISS   │ │ FAISS   │ │ FAISS   │
│ L0 Frame│ │ L1 Scene│ │ L3 Know │
└─────────┘ └─────────┘ └─────────┘
```

---

## 5. Dataset Statistics Target

### 5.1 Final Dataset Composition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TARGET DATASET STATISTICS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  VISUAL INDEX                                                                │
│  ───────────────                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Source              │ Count      │ Type         │ Notes            │    │
│  ├────────────────────┼────────────┼──────────────┼─────────────────┤    │
│  │ MovieNet subset    │ 14,397     │ Keyframes    │ Shot-level      │    │
│  │ MovieGraphs frames │  5,888     │ Keyframes    │ Start/End       │    │
│  │ UCF-101 frames     │  7,025     │ Keyframes    │ Action clips    │    │
│  │ ActivityNet frames │ 99,452     │ Keyframes    │ 787 videos      │    │
│  │ Trailer frames     │ 12,657     │ Keyframes    │ 191 trailers    │    │
│  ├────────────────────┼────────────┼──────────────┼─────────────────┤    │
│  │ TOTAL              │ 139,419    │              │                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  KNOWLEDGE INDEX                                                             │
│  ───────────────                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Source              │ Count      │ Type         │ Notes            │    │
│  ├────────────────────┼────────────┼──────────────┼─────────────────┤    │
│  │ ActivityNet caps    │  71,957   │ Text vectors │ Dense captions  │    │
│  │ CMU Movie Summ.    │  42,306   │ Text vectors │ Wikipedia plots  │    │
│  │ Cornell Dialogs    │  55,456   │ Text vectors │ 220K lines      │    │
│  │ MovieNet subtitles │  12,255   │ Text vectors │ 38 SRT files    │    │
│  │ MovieGraphs texts  │   7,761   │ Text vectors │ Scene graphs    │    │
│  │ External (future)  │  TBD      │ Text vectors │ LSMDC, CinePile │    │
│  ├────────────────────┼────────────┼──────────────┼─────────────────┤    │
│  │ TOTAL              │ 189,735+  │              │                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  SCENE INDEX (L1)                                                            │
│  ────────────────                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Source              │ Count      │ Type         │ Notes            │    │
│  ├────────────────────┼────────────┼──────────────┼─────────────────┤    │
│  │ MovieNet scenes     │   3,500   │ Scene vectors│ LLM-segmented  │    │
│  │ MovieGraphs scenes  │   2,000   │ Scene vectors│ Graph-chunked  │    │
│  │ ActivityNet scenes  │  10,000   │ Scene vectors│ Semantic groups│    │
│  ├────────────────────┼────────────┼──────────────┼─────────────────┤    │
│  │ TOTAL              │  15,500   │              │                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  GRAPH DATABASE (Neo4j)                                                      │
│  ─────────────────────                                                       │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Entity/Relation    │ Count      │ Type                            │    │
│  ├────────────────────┼────────────┼─────────────────────────────────┤    │
│  │ Movie nodes        │    242    │ Movies (IMDb Top 250)           │    │
│  │ Character nodes    │  1,200+  │ Characters across movies        │    │
│  │ Scene nodes        │ 15,500+   │ Semantic scenes                  │    │
│  │ Event nodes        │  8,000+   │ Narrative events                 │    │
│  │ APPEARS_IN rels    │ 25,000+   │ Character → Scene               │    │
│  │ INTERACTS_WITH rels│ 10,000+   │ Character ↔ Character           │    │
│  │ FOLLOWS rels      │ 12,000+   │ Scene → Scene (temporal)        │    │
│  │ DEPICTS rels      │  8,000+   │ Scene → Event                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Dataset Preprocessing Specifications

### 6.1 Keyframe Extraction Config

```yaml
# keyframe_extraction.yaml
keyframe_extraction:
  # Sampling strategy
  strategy: "diverse"  # vs "uniform"

  # Uniform sampling
  fps: 1.0

  # Diverse sampling (for visual diversity)
  min_frames_per_scene: 3
  max_frames_per_scene: 10
  diversity_threshold: 0.3  # CLIP cosine distance

  # Quality filtering
  min_resolution: [224, 224]
  max_blur_score: 100  # Laplacian variance
  min_aesthetic_score: 0.5

  # Output
  format: "jpg"
  quality: 95
  resize: [336, 336]  # CLIP ViT-L/14 input size
```

### 6.2 Whisper STT Config

```yaml
# whisper_transcription.yaml
whisper:
  model: "medium"  # vs "small", "base", "large"
  language: null  # Auto-detect
  task: "transcribe"

  # Timestamping
  word_timestamps: true
  prepend_punctuations: ".?!"
  append_punctuations: ".?!,"

  # Chunking
  max_segment_duration: 30.0  # seconds
  segment_overlap: 1.0  # seconds

  # Output format
  output_format: "srt"

  # Alignment (for subtitle sync)
  align_model: "WAV2VEC2"  # or null
```

### 6.3 VLM Scene Understanding Config

```yaml
# vlm_scene_understanding.yaml
vlm:
  model: "Qwen2-VL-7B-Instruct"  # Primary
  fallback: "llava-1.6-mistral-7b"  # Fallback

  # Frame sampling
  frames_per_video: 16  # Max frames to analyze
  sampling_strategy: "uniform"  # or "keyframes_only"

  # Prompt templates
  system_prompt: |
    You are an expert film analyst. Analyze this movie frame carefully.

  output_format: "json"

  # Batching (for efficiency)
  batch_size: 4
  max_tokens: 512

  # Caching
  cache_embeddings: true
  cache_dir: "data/vlm_cache"
```

---

## 7. Dataset Format Specifications

### 7.1 Temporal Chunk JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "chunk_id": {
      "type": "string",
      "description": "Unique chunk identifier (e.g., tt0120338_chunk_0001)"
    },
    "movie_id": {
      "type": "string",
      "description": "IMDb ID or movie identifier"
    },
    "shot_start": {
      "type": "integer",
      "description": "Starting shot number"
    },
    "shot_end": {
      "type": "integer",
      "description": "Ending shot number"
    },
    "start_seconds": {
      "type": "number",
      "description": "Start time in seconds"
    },
    "end_seconds": {
      "type": "number",
      "description": "End time in seconds"
    },
    "timestamp_source": {
      "type": "string",
      "enum": ["annotation_frame", "shot_boundary", "scene_segmentation"],
      "description": "Source of timestamp accuracy"
    },
    "description": {
      "type": "string",
      "description": "VLM-generated scene description"
    },
    "situation": {
      "type": "string",
      "description": "Overall situation label"
    },
    "characters": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Named characters in scene"
    },
    "dialogue_text": {
      "type": "string",
      "description": "Spoken dialogue (Whisper output)"
    },
    "script_primary_heading": {
      "type": "string",
      "description": "INT./EXT. heading from script"
    },
    "script_location": {
      "type": "string",
      "description": "Detailed script location"
    },
    "vision_setting": {
      "type": "string",
      "description": "Visual environment description"
    },
    "vision_actions": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Actions occurring in scene"
    },
    "cast_in_scene": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "actor": {"type": "string"},
          "character": {"type": "string"}
        }
      }
    },
    "keyframe_paths": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Paths to representative keyframes"
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Confidence score for this chunk"
    }
  },
  "required": ["chunk_id", "movie_id", "start_seconds", "end_seconds"]
}
```

### 7.2 Metadata JSON Schema

```json
{
  "movie_id": "tt0120338",
  "title": "Titanic",
  "year": 1997,
  "genres": ["Drama", "Romance"],
  "duration_seconds": 11340,
  "cast": [
    {"name": "Leonardo DiCaprio", "character": "Jack Dawson"},
    {"name": "Kate Winslet", "character": "Rose DeWitt Bukater"}
  ],
  "director": "James Cameron",
  "num_scenes": 150,
  "num_shots": 847,
  "total_chunks": 320,
  "keyframes_dir": "data/movienet_subset_20/tt0120338/shots",
  "subtitles_path": "data/movienet_subset_20/tt0120338/subtitles/subtitles.srt",
  "script_path": "data/scripts/tt0120338.txt",
  "graph_data_path": "data/graphs/tt0120338.json"
}
```

---

## 8. Dataset Versioning

### 8.1 Dataset Manifest

```yaml
# dataset_manifest.yaml
version: "2.0.0"
last_updated: "2026-03-19"

internal_datasets:
  activitynet_captions:
    version: "1.0"
    size_mb: 450
    path: "dataset/ActivityNet_Captions/"
    split:
      train: 10019 videos
      val_1: 4917 videos
      val_2: 4917 videos

  movienet_subset:
    version: "2.0"
    size_mb: 12000
    path: "data/movienet_subset_20/"
    num_movies: 19
    num_keyframes: 33021

  moviegraphs:
    version: "1.2"
    size_mb: 85
    path: "data/moviegraphs/"
    num_movies: 52
    num_clips: 7761

external_datasets:
  msrvtt:
    status: "pending"
    target_size_gb: 20
    download_url: "https://github.com/..."

  charades:
    status: "pending"
    target_size_gb: 5
    download_url: "https://prior.allenai.org/..."

processing:
  clip_model: "ViT-L/14"
  whisper_model: "medium"
  vlm_model: "Qwen2-VL-7B"
  embedding_dim: 768  # ViT-L/14
```

---

## 9. Ethical Considerations

### 9.1 Data Licensing

| Dataset | License | Commercial Use | Attribution Required |
|---|---|---|---|
| ActivityNet | Non-commercial research | No | Yes |
| MovieNet | Research only | No | Yes |
| MovieGraphs | Non-commercial | No | Yes |
| MSR-VTT | Custom (Microsoft) | No | Yes |
| Charades | Research only | No | Yes |
| LSMDC | Academic request | No | Yes |
| CinePile | Apache 2.0 | Yes | Yes |
| YouTube (via yt-dlp) | Platform ToS | Varies | Depends |

### 9.2 Privacy Considerations

- **Face Detection**: Only detect, don't identify or store identities
- **Audio**: Transcribe only speech, don't store raw audio
- **Movie Content**: Use only for academic research
- **User Queries**: Don't log or store user queries externally

---

## 10. Storage Requirements

### 10.1 Estimated Storage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STORAGE ESTIMATES                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RAW VIDEOS                                                                  │
│  ────────────                                                                │
│  ActivityNet Videos (787 x 300MB avg)     = 236 GB                          │
│  MovieNet Raw Videos (19 x 5GB avg)       = 95 GB                          │
│  YouTube Trailers (191 x 200MB avg)       = 38 GB                          │
│  External datasets (MSR-VTT, etc.)       = 50 GB                          │
│                                                                              │
│  Subtotal Raw Videos                     = 419 GB                         │
│                                                                              │
│  PROCESSED DATA                                                              │
│  ────────────────                                                           │
│  Keyframes (139K x 100KB avg)            = 14 GB                          │
│  VLM Embeddings Cache                    = 5 GB                           │
│  CLIP Embeddings                         = 1 GB                           │
│  Temporal Chunks JSON                    = 2 GB                           │
│  Subtitle/SRT files                      = 500 MB                         │
│  Whisper Transcripts                    = 1 GB                           │
│                                                                              │
│  Subtotal Processed                      = 23.5 GB                        │
│                                                                              │
│  INDEXES                                                                        │
│  ──────                                                                     │
│  FAISS L0 Frame Index                   = 500 MB                          │
│  FAISS L1 Scene Index                   = 100 MB                          │
│  FAISS L3 Knowledge Index               = 400 MB                          │
│  Neo4j Graph Database                    = 2 GB                           │
│  SQLite Metadata Cache                  = 500 MB                          │
│                                                                              │
│  Subtotal Indexes                       = 3.5 GB                          │
│                                                                              │
│  ══════════════════════════════════════════════════════════════════════    │
│  TOTAL STORAGE REQUIRED                  = 446 GB                          │
│  Recommended: 500GB SSD (for performance)                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
