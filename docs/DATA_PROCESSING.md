# 📦 Data Processing Pipeline: VideoSceneRAG

> **Mục tiêu:** Từ raw video → structured knowledge → FAISS indexes → ready for inference.
> Phần này hướng dẫn **toàn bộ quy trình** thu thập, xử lý, tổ chức data để chạy thực nghiệm cho VideoSceneRAG.

---

## 1. Tổng Quan Kiến Trúc Data

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│   │ Internal     │  │ External     │  │ Manual               │ │
│   │ MovieNet     │  │ MSR-VTT      │  │ Your own videos      │ │
│   │ MovieGraphs  │  │ LSMDC        │  │ (MP4, MKV, AVI)      │ │
│   │ Subset (20)  │  │ Charades-STA │  │                      │ │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
└──────────┼─────────────────┼──────────────────────┼─────────────┘
           │                 │                      │
           ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING PIPELINE                      │
│                                                                   │
│  RAW VIDEO ──► Shot Detection ──► Keyframe Extract ──► CLIP Emb  │
│      │              │               │                │             │
│      ▼              ▼               ▼                ▼          │
│  Raw Files     Shot Boundary    Frame Images      128K+ Vectors  │
│                                                                   │
│  Whisper STT ──► Chunking 30s ──► Text Embed ──► Knowledge Index │
│                                                                   │
│  VLM Analysis ──► Scene Desc ──► Conflict Check ──► Enrich Metadata│
│                                                                   │
│  Script Align ──► Face Track ──► Action Recog ──► Neo4j Graph     │
└─────────────────────────────────────────────────────────────────┘
           │                 │                      │
           ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INDEXING & STORAGE                            │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐│
│  │ Frame Index  │  │ Scene Index │  │ Knowledge Index          ││
│  │ L0 (128K+)   │  │ L1 (~15.5K) │  │ L3 (189K+)              ││
│  │ FAISS        │  │ FAISS        │  │ FAISS                    ││
│  └──────────────┘  └──────────────┘  └──────────────────────────┘│
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ Neo4j Graph: Movie → Scene → Character → Event → Relationship││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ Chunk JSON (5-Layer Metadata) per Scene                      ││
│  │ Layer 1: Temporal  Layer 2: Semantic  Layer 3: Dialogue     ││
│  │ Layer 4: Cast      Layer 5: Narrative                       ││
│  └──────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Hiện Có & Cần Thu Thập

### 2.1 Internal Data (đã có trong project)

| Dataset | Đường dẫn | Trạng thái | Cần làm gì |
|---------|-----------|-----------|------------|
| `movie_data_subset_20` | `project_ky4/movie_data_subset_20/` | ⚠️ Thư mục rỗng | Thu thập video từng phim |
| MovieNet metadata | `data/movienet/` | ⚠️ Chưa tải | Tải từ OpenDataLab |
| MovieGraphs | `data/MovieGraphs_repo/` | ⚠️ Chưa tải | Tải từ trang chủ |
| Raw videos | `data/raw_videos/` | ✅ Có thể có 1 số | Kiểm tra + bổ sung |
| Subtitles | `movie_data_subset_20/subtitle/` | ⚠️ Có thể có | Copy từ phim có sẵn |
| Scripts | `movie_data_subset_20/script/` | ⚠️ Có thể có | Tìm thêm online |

### 2.2 External Data (cần thu thập thêm)

| Dataset | Dung lượng | Nguồn | Cách lấy |
|---------|-----------|--------|----------|
| **MSR-VTT** | ~7GB video, 10K clips | Microsoft | `yt-dlp` hoặc link trực tiếp |
| **LSMDC** | ~10GB video, 118K clips | Penn State | Đăng ký form, download |
| **Charades-STA** | ~5GB video, 12K clips | Charades Dataset | `yt-dlp` |
| **ActivityNet** | ~400GB video, 20K videos | ActivityNet | `yt-dlp` + Crawlee |
| **YouCook2** | ~4GB video, 89 recipes | YouCook2 | `yt-dlp` |

### 2.3 Self-collected Videos (tự thu thập)

```
📁 data/self_collected/
├── titanic_1997.mp4
├── godfather_1972.mp4
├── inception_2010.mp4
├── ...
└── METADATA.yaml    # Tên phim, năm, thể loại, nguồn
```

---

## 3. Quy Trình Xử Lý Chi Tiết (Pipeline)

### Phase A: Data Acquisition

```
Step A1 ──► A2 ──► A3 ──► A4 ──► A5
```

#### A1. Kiểm tra data hiện có

```bash
# Chạy script kiểm tra
python -m movierag.scripts.download_data

# Output mẫu:
# [1] MovieNet: ❌ MISSING — Cần tải metadata + keyframes
# [2] MovieGraphs: ❌ MISSING — Cần tải graphs
# [3] Raw Videos: ⚠️ PARTIAL — Có 3/20 phim
```

#### A2. Tải MovieNet Metadata + Keyframes

```bash
# 1. Đăng ký tài khoản OpenDataLab
#    https://opendatalab.com/OpenDataLab/MovieNet

# 2. Tải 2 file cần thiết:
#    - annotation.v1.zip (~2GB)
#    - meta.v1.zip (~50MB)

# 3. Giải nén vào thư mục
mkdir -p data/movienet
unzip annotation.v1.zip -d data/movienet/
unzip meta.v1.zip -d data/movienet/

# 4. Kiểm tra cấu trúc
ls data/movienet/
# Cần có: items/  annotations/  keyframes/ (nếu có)
```

#### A3. Tải MovieGraphs

```bash
# 1. Truy cập: http://moviegraphs.cs.toronto.edu/download.html
# 2. Điền form Google → nhận email với link download
# 3. Tải và giải nén
mkdir -p data/MovieGraphs_repo
cp all_movies.pkl data/MovieGraphs_repo/
```

#### A4. Tải phim từ public sources (self-collected)

```bash
# Cài đặt yt-dlp
pip install yt-dlp

# Tải trailer (đủ để demo)
yt-dlp \
  -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
  -o "data/self_collected/%(title)s.%(ext)s" \
  "https://www.youtube.com/watch?v=..."
```

#### A5. Dataset versioning (YAML manifest)

```yaml
# data/datasets.yaml
version: "1.0"
last_updated: "2026-03-20"

datasets:
  movienet:
    source: "https://opendatalab.com/MovieNet"
    downloaded: true
    size_gb: 15.0
    num_movies: 800
    path: "data/movienet/"
    status: "metadata_ready_frames_partial"

  moviegraphs:
    source: "http://moviegraphs.cs.toronto.edu"
    downloaded: false
    size_gb: 2.0
    num_movies: 312
    path: "data/MovieGraphs_repo/"
    status: "pending"

  msr_vtt:
    source: "Microsoft Research"
    downloaded: false
    size_gb: 7.0
    num_clips: 10000
    path: "data/msr_vtt/"
    status: "pending"
    download_cmd: "python tools/download_msrvtt.py"
```

---

### Phase B: Preprocessing Pipeline

```
B1: Shot Detection ──► B2: Keyframe Extract ──► B3: CLIP Encoding
       │                        │                        │
       ▼                        ▼                        ▼
  Shot boundaries          Frame images            128K+ vectors
  (boundary.csv)           (shot_keyf/)           (frame_index.faiss)

B4: Whisper STT ──► B5: Scene Understanding ──► B6: Graph Building
       │                        │                        │
       ▼                        ▼                        ▼
  Transcript chunks       VLM descriptions        Neo4j graph
  (dialogue_index)       (scene_index L1)        (knowledge_graph)
```

#### B1. Shot Detection

```python
# src/preprocess_data/pipeline.py — đã có sẵn
# Dùng PySceneDetect hoặc ffmpeg để detect shot boundaries

from preprocess_data.pipeline import PipelineRunner

runner = PipelineRunner(
    movie_id="tt0120338",
    video_path="data/raw_videos/tt0120338.mp4",
    srt_path="movie_data_subset_20/subtitle/tt0120338.srt",
    force=False,  # Skip nếu đã có
)
runner.run_all()  # Shot detect → Keyframes → CLIP → Scene cluster → Subtitle index
```

**Output:**
```
data/processed/
└── tt0120338/
    ├── boundaries/shot_boundaries.csv   # Frame idx, timestamp
    ├── keyframes/shot_keyf/             # 1 keyframe per shot
    ├── embeddings/clip_embeddings.npy   # N x 512
    └── subtitle_index/                  # Whisper/subtitle chunks
```

#### B2. Keyframe Extraction

```bash
# Cách 1: Dùng ffmpeg (nhanh, đơn giản)
ffmpeg -i input.mp4 \
  -vf "select='eq(pict_type,PICT_TYPE_I)'" \
  -vsync vfr \
  -q:v 2 \
  output_dir/frame_%04d.jpg

# Cách 2: Dùng PySceneDetect (chính xác hơn)
pip install scenedetect
scenedetect detect-content input.mp4 \
  --min-scene-len 15 \
  -o output_dir/

# Cách 3: Dùng script có sẵn
python -m preprocess_data \
  --movie-id tt0120338 \
  --video data/raw_videos/tt0120338.mp4 \
  --mode keyframe \
  --fps 1.0 \
  --output data/processed/tt0120338/keyframes/
```

#### B3. CLIP Encoding

```python
# Encoding tất cả keyframes thành vectors

from movierag.indexing.clip_encoder import CLIPEncoder
from movierag.indexing.visual_indexer import VisualIndexer
import numpy as np
from PIL import Image
from tqdm import tqdm

encoder = CLIPEncoder()
indexer = VisualIndexer(index_dir="data/indexes", index_name="frame_index")

# Encode từng batch
BATCH_SIZE = 32
keyframe_dir = Path("data/processed/tt0120338/keyframes/")

items = []
for kf_path in sorted(keyframe_dir.glob("*.jpg")):
    emb = encoder.encode_image(str(kf_path))
    items.append({
        "keyframe_id": kf_path.stem,
        "keyframe_path": str(kf_path),
        "movie_id": "tt0120338",
        "embedding": emb,
    })

    if len(items) >= BATCH_SIZE:
        indexer.add_batch(items)
        items = []

if items:
    indexer.add_batch(items)

indexer.save()
print(f"✅ Frame index saved: {len(indexer)} vectors")
```

#### B4. Whisper STT Transcription

```python
# Chuyển đổi audio → text với timestamps

from movierag.indexing.whisper_transcriber import WhisperTranscriber

transcriber = WhisperTranscriber(
    model_name="medium",
    device="cuda",
    chunk_length=30,  # 30 giây per chunk
    output_dir="data/transcripts/",
)

# Transcribe từ video
result = transcriber.transcribe(
    video_path="data/raw_videos/tt0120338.mp4",
    movie_id="tt0120338",
    word_timestamps=True,
)

print(f"Language: {result['language']}")
print(f"Full text: {result['full_text'][:200]}...")
print(f"Chunks: {len(result['chunks'])}")

# Convert to index documents
docs = transcriber.to_index_documents(result)
print(f"Index docs: {len(docs)}")

# Save
import json
with open(f"data/transcripts/{result['movie_id']}_transcript.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
```

**Output schema (per chunk):**
```json
{
  "chunk_id": "tt0120338_chunk_0000",
  "movie_id": "tt0120338",
  "start_seconds": 0.0,
  "end_seconds": 29.8,
  "text": "You jump, I jump right? That's the deal.",
  "language": "en",
  "speaker": null,
  "audio_events": [],
  "background_music": false,
  "avg_logprob": -0.12,
}
```

#### B5. VLM Scene Understanding

```python
# Phân tích scene bằng Vision Language Model

from movierag.indexing.vlm_scene_analyzer import VLMSceneAnalyzer

analyzer = VLMSceneAnalyzer(
    llm_client=llm_client,
    max_frames=8,
    batch_size=4,
)

# Analyze từng scene
scene_result = analyzer.analyze_scene(
    video_path="data/raw_videos/tt0120338.mp4",
    scene_start=120.0,
    scene_end=150.0,
    scene_id="tt0120338_scene_001",
    movie_id="tt0120338",
    expected_movie="Titanic",
    expected_characters=["Jack", "Rose"],
)

# Kết quả
print(f"Situation: {scene_result.situation}")        # "romantic"
print(f"Setting: {scene_result.vision_setting}")     # "deck at sunset"
print(f"Actions: {scene_result.vision_actions}")     # ["kissing", "talking"]
print(f"Characters: {scene_result.characters_detected}")
print(f"Keywords: {scene_result.distilled_keywords}")
print(f"Conflict: {scene_result.vlm_conflict_detected}")

# Convert to 5-Layer metadata
metadata = scene_result.to_metadata_dict()
```

#### B6. Neo4j Graph Construction

```python
# Xây dựng knowledge graph từ scene metadata

from movierag.indexing.neo4j_graph_store import Neo4jGraphStore

store = Neo4jGraphStore()

# Load MovieGraphs data
import pickle
with open("data/MovieGraphs_repo/all_movies.pkl", "rb") as f:
    moviegraphs = pickle.load(f)

# Sync vào Neo4j
for movie_id, graph_data in moviegraphs.items():
    result = store.sync_movie(movie_id, graph_data)
    print(f"Synced {movie_id}: {result['nodes']} nodes, {result['edges']} edges")

# Query example: tìm character relationships
hits = store.search("Rose", filters={"movie_id": "tt0120338"}, top_k=10)
print(f"Found {len(hits)} hits for 'Rose'")
```

---

### Phase C: Index Building

#### C1. Build Frame Index (L0)

```python
# Index tất cả frame vectors bằng FAISS

from movierag.indexing.visual_indexer import VisualIndexer
import numpy as np

indexer = VisualIndexer(
    index_dir="data/indexes",
    index_name="movie_frame",
)

# Build from MovieNet
items = []
for movie_dir in Path("data/movienet/keyframes/").iterdir():
    if not movie_dir.is_dir():
        continue
    movie_id = movie_dir.name
    for kf in movie_dir.glob("*.jpg"):
        items.append({
            "keyframe_id": kf.stem,
            "keyframe_path": str(kf),
            "movie_id": movie_id,
            "shot_id": kf.stem.split("_")[0],
        })

print(f"Building index for {len(items)} frames...")
indexer.build_index(items)
indexer.save()

stats = indexer.get_statistics()
print(f"✅ Frame Index: {stats['num_vectors']} vectors, dim={stats['embedding_dim']}")
```

#### C2. Build Scene Index (L1) — Fusion 72% image + 28% text

```python
# Xây dựng scene-level index từ frame-level

# Điều kiện: cần có scene boundaries từ B1 và scene metadata từ B5
# visual_indexer.py đã có _build_scene_index() — gọi nó:

scene_items = []
for movie_dir in Path("data/processed/").iterdir():
    scene_meta_path = movie_dir / "scene_metadata.json"
    if not scene_meta_path.exists():
        continue

    import json
    with open(scene_meta_path) as f:
        scenes = json.load(f)

    for scene in scenes:
        scene_items.append({
            "scene_id": scene["scene_id"],
            "movie_id": scene["movie_id"],
            "start_seconds": scene["start_seconds"],
            "end_seconds": scene["end_seconds"],
            # Layer 2: Semantic
            "situation": scene.get("situation", ""),
            "description": scene.get("vlm_description", ""),
            "vision_setting": scene.get("vision_setting", ""),
            "vision_actions": scene.get("vision_actions", []),
            # Layer 3: Dialogue
            "dialogue_text": scene.get("dialogue_text", ""),
            # Layer 4: Characters
            "characters": scene.get("characters", []),
            "character_emotions": scene.get("character_emotions", {}),
            # Layer 5: Narrative
            "script_heading": scene.get("script_heading", ""),
            "narrative_arc": scene.get("narrative_arc", ""),
        })

# Build scene index với visual fusion
# (visual_indexer._build_scene_index() tự động fusion 72/28)
scene_indexer.build_index(scene_items)
scene_indexer.save()

print(f"✅ Scene Index (L1): {len(scene_items)} scene vectors")
```

#### C3. Build Knowledge Index (L3)

```python
# FAISS index cho text-based knowledge retrieval

from movierag.indexing.knowledge_indexer import KnowledgeIndexer

indexer = KnowledgeIndexer(index_dir="data/indexes", index_name="movie_knowledge")

# Nguồn 1: Whisper transcripts (Layer 3)
for transcript_file in Path("data/transcripts/").glob("*.json"):
    with open(transcript_file) as f:
        result = json.load(f)
    docs = transcriber.to_index_documents(result)
    indexer.add_documents(docs)

# Nguồn 2: MovieGraphs knowledge
with open("data/MovieGraphs_repo/all_movies.pkl", "rb") as f:
    moviegraphs = pickle.load(f)
kg_docs = []
for movie_id, data in moviegraphs.items():
    for scene in data.get("scenes", []):
        kg_docs.append({
            "text": scene.get("description", ""),
            "movie_id": movie_id,
            "type": "scene_graph",
            "source": "moviegraphs",
        })
indexer.add_documents(kg_docs)

# Nguồn 3: ActivityNet / external captions
for cap_file in Path("data/activitynet_captions/").glob("*.json"):
    with open(cap_file) as f:
        caps = json.load(f)
    indexer.add_documents([{
        "text": c["sentence"],
        "video_id": c["video_id"],
        "timestamp": c["timestamp"],
        "type": "caption",
        "source": "activitynet",
    } for c in caps])

indexer.save()
print(f"✅ Knowledge Index (L3): {indexer.get_count()} documents")
```

#### C4. Build Neo4j Graph

```python
# Đẩy tất cả dữ liệu vào Neo4j

from movierag.indexing.neo4j_graph_store import Neo4jGraphStore

store = Neo4jGraphStore()

# Batch sync tất cả movies
movie_dirs = list(Path("data/processed/").iterdir())
for movie_dir in tqdm(movie_dirs, desc="Syncing to Neo4j"):
    movie_id = movie_dir.name
    graph_file = movie_dir / "knowledge_graph.json"
    if graph_file.exists():
        with open(graph_file) as f:
            graph_data = json.load(f)
        store.sync_movie(movie_id, graph_data)
    else:
        # Build from 5-layer metadata
        scene_file = movie_dir / "scene_metadata.json"
        if scene_file.exists():
            store.sync_movie(movie_id, _build_graph_from_scenes(scene_file))

print(f"✅ Neo4j graph populated")
```

---

### Phase D: Storage Schema (CHUẨN HÓA)

> **Cấu trúc này được enforce bởi `verify_pipeline_output.py`**
> Chạy: `python -m movierag.scripts.verify_pipeline_output` để kiểm tra.

```
📁 data/                           ← Root data directory
│
├── .dataset_config.yaml           ← Dataset manifest (quản lý version)
│
├── msr_vtt/                       ← Raw dataset: MSR-VTT
│   ├── MSR_VTT_All.json           ← Annotations (200K captions)
│   ├── videourlist.txt            ← Video URLs
│   └── RawVideoAll/               ← Downloaded videos
│
├── didemo/                        ← Raw dataset: DiDeMo
│   ├── didemo_captions.json       ← 40,579 moment descriptions
│   ├── video_urls.json            ← Video URLs
│   └── videos/                    ← Downloaded videos
│
├── MovieGraphs_repo/             ← Raw dataset: MovieGraphs
│   ├── all_movies.pkl            ← Scene graphs (52 phim)
│   ├── py3loader_new/            ← Python 3 loader
│   └── DOWNLOAD_INSTRUCTIONS.txt
│
├── ActivityNet_Captions/          ← Raw dataset: ActivityNet
│   ├── captions.json              ← 71,957 dense captions
│   └── entities.json
│
├── charades_sta/                  ← Raw dataset: Charades-STA
│   ├── Charades_Sta.csv          ← Temporal grounding GT
│   └── videos/
│
├── movie_data_subset_20/         ← Internal: 19 phim Hollywood
│   ├── meta/                     ← Movie metadata JSON
│   ├── subtitle/                 ← SRT subtitles
│   ├── script/                  ← Screenplays
│   ├── annotation/              ← Shot/scene annotations
│   └── shots/                    ← Keyframes (nếu có)
│
├── raw_videos/                    ← Self-collected video files
│   ├── tt0120338.mp4            ← Titanic
│   └── tt0068646.mp4            ← Godfather
│
├── self_collected/               ← User-uploaded videos
│   ├── videos/
│   ├── subtitles/
│   ├── scripts/
│   └── metadata.yaml
│
└── pipeline_output/               ← ⭐ SAU KHI CHẠY PIPELINE
    │
    ├── indexes/                   ← FAISS Indexes
    │   ├── movie_frame_index.faiss       # L0: 128K+ frame vectors
    │   ├── movie_frame_index_map.json     # Metadata map
    │   ├── movie_scene_index.faiss        # L1: ~15.5K scene vectors (72/28 fusion)
    │   ├── movie_scene_index_map.json
    │   ├── movie_knowledge_index.faiss     # L3: text embeddings
    │   ├── movie_knowledge_index_map.json
    │   ├── knowledge_msr_vtt.faiss          # Dataset-specific indexes
    │   ├── knowledge_didemo.faiss
    │   └── knowledge_moviegraphs.faiss
    │
    ├── graphs/                   ← Knowledge Graphs
    │   └── movie_graph_index.graphml
    │
    ├── temporal_chunks/         ← Whisper/SRT chunks (30s)
    │   └── {movie_id}_chunks.json
    │
    ├── transcripts/              ← Whisper outputs
    │   └── {movie_id}_transcript.json
    │
    ├── msr_vtt_chunks/          ← MSR-VTT processed
    │   ├── all_chunks.json       # 5-Layer chunks
    │   └── grounding_gt.json     # Temporal GT
    │
    ├── didemo_chunks/           ← DiDeMo processed
    │   ├── all_chunks.json
    │   └── grounding_gt.json
    │
    ├── moviegraphs_chunks/       ← MovieGraphs processed
    │   ├── all_chunks.json
    │   └── knowledge_graph.json  # Neo4j-ready graph
    │
    ├── activitynet_chunks/       ← ActivityNet processed
    │   └── all_chunks.json
    │
    └── {movie_id}/              ← Per-movie processed output
        ├── boundaries/
        │   └── shot_boundaries.csv
        ├── keyframes/
        │   └── shot_keyf/       # 1 keyframe per shot
        ├── embeddings/
        │   ├── frame_embeddings.npy
        │   └── scene_embeddings.npy
        ├── scene_metadata.json   # ⭐ 5-Layer Scene Metadata
        ├── transcript.json       # Whisper STT
        ├── vlm_analysis.json    # VLM scene analysis
        ├── face_tracks.json      # Face detection + tracking
        ├── action_labels.json    # VideoMAE outputs
        └── knowledge_graph.json  # Neo4j-ready
```

---

## 3.5. Automated Scripts (QUAN TRỌNG)

### 3.5.1 Shell Script — download_process_datasets.sh

```bash
# Chạy toàn bộ quy trình tự động
cd /home/hiwe/project/DAP303m/project_ky4
chmod +x scripts/download_process_datasets.sh

# Tải Tier 1 (MSR-VTT + DiDeMo + MovieGraphs)
./scripts/download_process_datasets.sh --tier1

# Verify datasets
./scripts/download_process_datasets.sh --verify

# Process → chunks + FAISS indexes
./scripts/download_process_datasets.sh --process

# Full run: tải + verify + process + pipeline
./scripts/download_process_datasets.sh --full

# Chỉ tải 1 dataset
./scripts/download_process_datasets.sh --msrvtt
./scripts/download_process_datasets.sh --didemo
./scripts/download_process_datasets.sh --moviegraphs
```

### 3.5.2 Python CLI — manage_datasets.py

```bash
# Liệt kê tất cả datasets
python -m movierag.scripts.manage_datasets --list

# Trạng thái chi tiết
python -m movierag.scripts.manage_datasets --status

# Tải datasets
python -m movierag.scripts.manage_datasets --download msr_vtt didemo

# Verify
python -m movierag.scripts.manage_datasets --verify msr_vtt didemo moviegraphs

# Process (chuyển → 5-Layer chunks + indexes)
python -m movierag.scripts.manage_datasets --process msr_vtt didemo moviegraphs

# Tải + verify + process trong 1 lệnh
python -m movierag.scripts.manage_datasets --download-and-process msr_vtt didemo

# Tải + process tất cả Tier 1
python -m movierag.scripts.manage_datasets --all-tier1
```

### 3.5.3 Verification — verify_pipeline_output.py

```bash
# Verify toàn bộ pipeline output
python -m movierag.scripts.verify_pipeline_output

# Verify + check FAISS indexes
python -m movierag.scripts.verify_pipeline_output --check-indexes

# Verify 1 phim cụ thể
python -m movierag.scripts.verify_pipeline_output --movie-id tt0120338
```

### 3.5.4 Dataset Manifest — data/.dataset_config.yaml

File YAML quản lý version và trạng thái tất cả datasets:

```bash
# Mở file
cat data/.dataset_config.yaml

# Cập nhật status sau khi tải
# Sửa: downloaded: true, status: "ready"
```

---

## 3.6. 5-Layer Chunk JSON Schema (Output chuẩn)

```json
{
  "chunk_id": "tt0120338_chunk_0001",
  "movie_id": "tt0120338",
  "video_id": "tt0120338",

  "layer_1_temporal": {
    "start_seconds": 120.5,
    "end_seconds": 150.3,
    "timestamp_source": "shot_boundary",
    "shot_id": "shot_0042"
  },

  "layer_2_semantic": {
    "situation": "romantic",
    "description": "Rose stands at the bow of the Titanic...",
    "vision_setting": "deck at sunset",
    "vision_actions": ["standing", "looking", "smiling"],
    "emotional_tone": "romantic",
    "vlm_description": "A young woman stands at the bow..."
  },

  "layer_3_dialogue": {
    "dialogue_text": "I'm flying Jack!",
    "speaker": "Rose",
    "audio_events": ["wind", "ocean sounds"],
    "background_music": true
  },

  "layer_4_cast": {
    "characters": ["Rose", "Jack"],
    "character_emotions": {"Rose": "euphoric", "Jack": "joyful"},
    "face_tracking_ids": {"Rose": 1, "Jack": 2},
    "action_labels": ["standing", "looking over ocean"]
  },

  "layer_5_narrative": {
    "script_heading": "INT. TITANIC - BOW - DUSK",
    "screenplay_context": "Rose takes Jack to the bow...",
    "narrative_arc": "rising_action",
    "causal_relations": [],
    "scene_graph": {}
  },

  "metadata": {
    "dataset": "msr_vtt",
    "source": "whisper_transcription",
    "chunk_duration": 29.8,
    "language": "en",
    "avg_logprob": 0.87
  }
}
```

---

## 4. Hướng Dẫn Từng Bước: Chạy Thực Nghiệm

### Bước 0: Môi trường

```bash
# Clone/copy project
cd /home/hiwe/project/DAP303m/project_ky4

# Tạo virtual environment
python -m venv venv && source venv/bin/activate  # Linux
# python -m venv venv && venv\Scripts\activate   # Windows

# Cài dependencies
pip install torch torchvision                   # PyTorch (CUDA nếu có GPU)
pip install transformers faiss-cpu faiss-gpu    # FAISS + transformers
pip install openai-whisper                     # Whisper
pip install openai tiktoken                    # LLM client
pip install scenedetect opencv-python           # Video processing
pip install mediapipe insightface               # Face detection
pip install neo4j                               # Graph DB
pip install yt-dlp                              # Video download
pip install python-dotenv pydantic loguru      # Utils

# Cài MovieNet dependencies
pip install scenedetect pandas numpy pillow

# Kiểm tra GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Bước 1: Thu thập data

```bash
# 1.1 Tải MovieNet từ OpenDataLab (đăng ký + tải 2 file)
# 1.2 Tải MovieGraphs từ trang chủ
# 1.3 Copy video vào data/raw_videos/

# Verify
python -m movierag.scripts.download_data
```

### Bước 2: Chạy preprocessing pipeline cho 1 phim thử

```bash
# Chạy với phim đã có video
python run_full_pipeline_tt0167404.py

# Hoặc chạy từ đầu cho phim mới
python -m preprocess_data \
  --movie-id tt0120338 \
  --video data/raw_videos/tt0120338.mp4 \
  --srt movie_data_subset_20/subtitle/tt0120338.srt \
  --force

# Output: data/processed/tt0120338/
```

### Bước 3: Build indexes

```bash
# Build frame index
python -m movierag.scripts.build_index \
  --data-dir data/movienet \
  --index-dir data/indexes \
  --sample  # Thử với sample data trước

# Build scene + knowledge index (tự động sau khi có scene_metadata)
python -m movierag.scripts.build_scene_index \
  --processed-dir data/processed \
  --index-dir data/indexes
```

### Bước 4: Sync Neo4j

```bash
# Khởi động Neo4j (Docker)
docker-compose -f docker-compose.neo4j.yml up -d

# Chờ Neo4j ready
sleep 10

# Sync graph data
python -c "
from movierag.indexing.neo4j_graph_store import Neo4jGraphStore
import json
from pathlib import Path

store = Neo4jGraphStore()
for movie_dir in Path('data/processed/').iterdir():
    gf = movie_dir / 'knowledge_graph.json'
    if gf.exists():
        with open(gf) as f:
            store.sync_movie(movie_dir.name, json.load(f))
        print(f'Synced: {movie_dir.name}')
print('Done!')
"
```

### Bước 5: Chạy demo/query

```bash
# Chạy FastAPI server
python -m uvicorn movierag.app:app --reload --port 8000

# Hoặc chạy script test
python -c "
from movierag.pipeline.agentic_pipeline import AgenticVideoRAGPipeline

pipeline = AgenticVideoRAGPipeline()

# Temporal query
result = pipeline.respond(
    query='When does Jack first appear in Titanic?'
)
print(result['answer'])
print(result.get('temporal_grounding'))

# Narrative query
result = pipeline.respond(
    query='Why does Rose let Jack go?'
)
print(result['answer'])
print(result.get('causal_explanation'))
"
```

---

## 5. Thứ Tự Ưu Tiên & Milestones

### Milestone 1: Minimal Viable Pipeline (1-2 ngày)
**Mục tiêu:** Demo chạy được với 1 phim

```
□ Có video của 1 phim (tt0120338 Titanic)
□ Chạy shot detection + keyframe extraction
□ CLIP encoding → frame index (L0)
□ Whisper transcription → knowledge index (L3)
□ Demo query: "Find scene where Jack draws Rose"
□ Demo query: "When does Rose first appear?"
```

### Milestone 2: Multi-Movie Index (3-5 ngày)
**Mục tiêu:** 5-10 phim, scene index (L1) hoạt động

```
□ Xử lý 5 phim
□ Build scene index (L1) với 72/28 fusion
□ Build knowledge index (L3)
□ Neo4j sync cho character tracking
□ Demo: Cross-movie character queries
```

### Milestone 3: Full Dataset (1-2 tuần)
**Mục tiêu:** MovieNet + MovieGraphs + external datasets

```
□ Tải và xử lý MovieNet (800 phim)
□ Load MovieGraphs knowledge graph
□ Tải MSR-VTT (10K clips)
□ Build all 3 FAISS indexes
□ Build full Neo4j graph
□ Full benchmark evaluation
```

---

## 6. Troubleshooting

### Lỗi thường gặp

```bash
# 1. FFmpeg not found
which ffmpeg || echo "❌ Install ffmpeg: apt install ffmpeg"

# 2. CUDA OOM (Out of Memory)
export CUDA_VISIBLE_DEVICES=0
# Giảm batch_size trong config

# 3. Neo4j connection refused
docker ps | grep neo4j
docker logs neo4j --tail 20

# 4. Whisper model not found
pip install --upgrade openai-whisper
python -c "import whisper; whisper.load_model('medium')"

# 5. FAISS index corrupted
# Xóa và rebuild
rm data/indexes/*.faiss
python -m movierag.scripts.build_index --force
```

### Resource Estimates

| Dataset | Video Size | Processing Time | Storage |
|---------|-----------|----------------|---------|
| 1 movie (2h) | ~2GB MP4 | 30-60 phút | 500MB processed |
| 10 movies | ~20GB | 5-10 giờ | 5GB |
| MovieNet (800) | ~1TB | 3-7 ngày | 200GB |
| MSR-VTT (10K) | ~7GB | 2-3 giờ | 15GB |
| ActivityNet (20K) | ~400GB | 2-3 tuần | 80GB |

### GPU Memory Guidelines

```python
# Batch size theo GPU VRAM
GPU_8GB:   CLIP batch=16,  VideoMAE batch=4
GPU_16GB:  CLIP batch=32,  VideoMAE batch=8
GPU_40GB:  CLIP batch=64,  VideoMAE batch=16
CPU only:  CLIP batch=4,   Sequential frame processing
```

---

*End of Data Processing Guide*
