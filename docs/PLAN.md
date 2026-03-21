# 📋 Kế Hoạch Chuyển Đổi: Retrieval → Video Understanding

## 1. Vấn Đề Hiện Tại

### 1.1 VideoRag — Chỉ là Retrieval thuần túy

**Hạn chế hiện tại:**
- CLIP FAISS vector search chỉ tìm frame gần nhất về mặt visual similarity
- Không hiểu **nội dung hành động, cảm xúc, mối quan hệ nhân vật**
- Không có **Temporal Reasoning** (sự kiện xảy ra trước/sau)
- Không có **Scene-level understanding** (cảnh vs shot)
- Không có **Audio/Speech understanding**
- Không có **Causal reasoning** (tại sao nhân vật làm vậy)
- Không có **Character tracking** (theo dõi nhân vật qua phim)

### 1.2 project_ky4 — Đã có nền tảng Video Understanding

**Điểm mạnh đã có:**
- ✅ Multi-layer metadata (5 tầng: Temporal → Semantic → Dialogue → Cast → Script)
- ✅ Scene Index (L1) — gộp frame thành scene có ngữ nghĩa
- ✅ Cross-encoder reranking
- ✅ Temporal modeling (hybrid search)
- ✅ Script alignment
- ✅ GraphRAG (Neo4j) cho entity relationships

**Điểm còn thiếu:**
- ❌ VLM multi-frame analysis (hiểu nội dung video qua vision model)
- ❌ Audio/STT integration (Whisper transcription)
- ❌ Temporal Grounding thực sự (localize sự kiện trong video)
- ❌ Video captioning / description generation
- ❌ Character face tracking
- ❌ Action recognition
- ❌ Evaluation benchmark đầy đủ

---

## 2. Mục Tiêu: Video Understanding Pipeline

### 2.1 Định Nghĩa Video Understanding

**Video Understanding ≠ Video Retrieval**

| Khía cạnh | Video Retrieval | Video Understanding |
|---|---|---|
| **Input** | Query (text/image) → Frame | Video → Full comprehension |
| **Output** | Top-k similar frames | Narrative summary, causal chains, character arcs |
| **Kiến thức** | "Tìm frame nào khớp" | "Cảnh này nói về cái gì, tại sao, hệ quả là gì" |
| **Temporal** | Single frame matching | Event ordering, timeline reasoning |
| **Reasoning** | Similarity search | Causal + Temporal + Spatial reasoning |
| **Models** | CLIP, FAISS | CLIP + VLM + VLM + Whisper + Action Models |
| **Use Case** | "Tìm cảnh Titanic chìm tàu" | "Phân tích tại sao Jack chết trong Titanic" |

### 2.2 Sản Phẩm Mục Tiêu

**Sau khi chuyển đổi, hệ thống sẽ trả lời được:**

1. **Temporal Grounding**: "Cảnh Jack vẽ Rose ở thời điểm nào trong phim?"
2. **Causal Reasoning**: "Tại sao Rose quyết định cứu Jack?"
3. **Character Analysis**: "Phân tích hành trình thay đổi của nhân vật Rose"
4. **Scene Understanding**: "Mô tả chi tiết cảnh phòng ăn tối cuối cùng"
5. **Multi-modal QA**: "Cảnh này có âm thanh gì đặc biệt?"
6. **Narrative QA**: "So sánh cách hai đạo diễn xử lý cảnh chết trong Titanic và Inception"

---

## 3. Dataset Strategy

### 3.1 Internal Datasets (đã có sẵn)

| Dataset | Nội dung | Kích thước | Vai trò |
|---|---|---|---|
| **ActivityNet Captions** | YouTube video captions | 19,803 videos, 71,957 segments | Knowledge index (text), evaluation |
| **ActivityNet Videos** | Raw YouTube videos | ~787 downloaded | Visual index, keyframe extraction |
| **MovieNet (subset)** | Hollywood movie shots | 19 movies, 14,397 shots | Visual index (keyframes) |
| **MovieGraphs** | Scene graphs annotations | 52 movies, 7,761 clips | Scene graph knowledge, entity tracking |

### 3.2 External Datasets (thu thập thêm)

| Dataset | Nguồn | Kích thước | Lý do |
|---|---|---|---|
| **MSR-VTT** | Microsoft Research | 10K video clips, 200K captions | Video-text alignment benchmark |
| **CinePile** | Academic dataset | ~300K video-caption pairs | Movie-domain pretraining |
| **Charades** | RGB-D video dataset | 9,848 videos, 157 classes | Action recognition |
| **LSMDC** | Movie descriptions | 118,081 clips from 202 movies | Movie captioning benchmark |
| **DiDeMo** | Moment localization | 10K videos, 40K descriptions | Temporal grounding |
| **YouCook2** | Cooking videos | 89 cooking recipes | Instructional video understanding |

### 3.3 Dataset Collection Pipeline

```
Bước 1: Xác định gap
├── Gap 1: Không có raw video cho movie understanding → Cần download thêm
├── Gap 2: Không có action labels → Cần Charades/AVA
├── Gap 3: Không có face tracking → Cần MovieNet face annotations
└── Gap 4: Không có audio understanding → Cần Whisper trên raw video

Bước 2: Ưu tiên thu thập
├── Priority 1: Raw video từ YouTube (yt-dlp) cho MovieNet subset
├── Priority 2: ActivityNet videos (787/1000 đã có)
├── Priority 3: External movie dataset (LSMDC, CinePile)
└── Priority 4: Action recognition dataset (Charades subset)

Bước 3: Annotation pipeline
├── Whisper STT → Subtitle alignment → Dialogue extraction
├── CLIP embeddings → Scene clustering → Semantic scene boundaries
├── VLM Scene Understanding → Description generation → Knowledge graph
└── Face detection → Character tracking → Cast alignment
```

---

## 4. Hệ Thống Đánh Giá (Evaluation)

### 4.1 Video Understanding Benchmarks

**Cần xây dựng internal benchmark + sử dụng external benchmarks:**

| Benchmark | Task | Metrics | Dataset |
|---|---|---|---|
| **VQAv2-style** | Visual QA | Accuracy | Internal + MSRVTT-QA |
| **Temporal Grounding** | Find moment | R@1, R@5, IoU | Charades-STA, DiDeMo |
| **Movie Narrative QA** | Story comprehension | Accuracy, F1 | Internal (MovieGraphs) |
| **Scene Description** | Generate description | BLEU, CIDEr, SPICE | LSMDC, CinePile |
| **Action Recognition** | Classify action | Top-1, Top-5 Accuracy | Charades |
| **Multi-modal Retrieval** | Cross-modal match | R@K, MRR | MSR-VTT |

### 4.2 Internal Evaluation Protocol

```
Dataset Split:
├── Train: 60% videos (MovieNet train + ActivityNet train)
├── Val:   20% videos (MovieNet val + ActivityNet val_1)
└── Test:  20% videos (MovieNet test + ActivityNet val_2)

Evaluation Scenarios:
├── Scenario 1: Temporal Grounding (Text → Video Moment)
├── Scenario 2: Visual QA (Image + Question → Answer)
├── Scenario 3: Narrative Reasoning (Multi-step QA)
├── Scenario 4: Scene Description Generation
└── Scenario 5: Cross-video Movie Identification
```

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Tuần 1-2)
1. Hoàn thiện Scene Index (L1) trong `visual_indexer.py`
2. Tích hợp Whisper STT pipeline
3. Xây dựng VLM Scene Understanding module
4. Đánh giá: Temporal Grounding baseline

### Phase 2: Multi-modal Understanding (Tuần 3-4)
1. Tích hợp Action Recognition (Charades)
2. Face Detection + Character Tracking
3. Cross-encoder reranking nâng cao
4. Đánh giá: VQAv2-style benchmarks

### Phase 3: Knowledge Integration (Tuần 5-6)
1. Neo4j GraphRAG hoàn chỉnh
2. Script-Scene alignment
3. Multi-hop reasoning pipeline
4. Đánh giá: Narrative QA, Movie comprehension

### Phase 4: Advanced Features (Tuần 7-8)
1. Video captioning / description generation
2. Causal reasoning module
3. Evaluation benchmark đầy đủ
4. Report và demo

---

## 6. Technical Dependencies

### 6.1 Models Required

| Model | Purpose | Library |
|---|---|---|
| **CLIP ViT-L/14** | Frame embedding (thay ViT-B/32) | open-clip |
| **Whisper (medium)** | Audio transcription | openai-whisper |
| **LLaVA / GPT-4V** | VLM scene understanding | transformers / API |
| **VideoMAE** | Action recognition | timm |
| **SAM** | Segment anything (face/motion) | segment-anything |
| **Claude-V3** | LLM reasoning | API |
| **Qwen2-VL** | Video understanding | modelscope |

### 6.2 Infrastructure

| Component | Requirement |
|---|---|
| GPU | NVIDIA A100 (40GB) hoặc tốt hơn |
| Storage | 500GB+ SSD cho video + indexes |
| RAM | 64GB+ |
| Neo4j | 4GB heap |
| FAISS | CPU + GPU indexes |

---

## 7. Success Criteria

### 7.1 Functional Requirements

- [ ] Trả lời được Temporal Grounding question với IoU > 0.5
- [ ] Generate được scene description có CIDEr > 0.5
- [ ] Hoàn thành character tracking qua >80% phim
- [ ] Temporal reasoning (trước/sau) chính xác > 85%
- [ ] Cross-modal retrieval R@5 > 80%

### 7.2 Research Contributions

1. **5-Layer Scene Understanding Architecture** — Đầu tiên kết hợp temporal, semantic, dialogue, cast, script trong 1 unified framework
2. **VLM-Guided Scene Segmentation** — Dùng VLM để detect scene boundaries thay vì chỉ dùng visual features
3. **Hierarchical Video Indexing** — Frame → Shot → Scene → Event → Narrative
4. **Agentic Video Understanding Pipeline** — Multi-step reasoning với tool calling

---

## 8. Risks và Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| VLM API cost quá cao | High | High | Dùng local VLM (LLaVA) + cache embeddings |
| Raw video download chậm | Medium | Medium | yt-dlp với parallel workers |
| Neo4j memory overflow | Low | High | Partition by movie, incremental queries |
| Whisper transcription quality | Medium | Medium | Manual spot-check trên 5% samples |
| CLIP bias với movie domain | High | Medium | Fine-tune trên movie subset |
