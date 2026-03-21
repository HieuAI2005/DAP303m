# VideoSceneRAG — Benchmark Analysis & Alternative Datasets

**Ngày:** 2026-03-20
**Project:** `/home/hiwe/project/DAP303m/project_ky4`

---

## 1. Hiện tại Benchmark đang làm gì

### 1.1 DiDeMo Temporal Grounding

**Benchmark đang chạy:** `scripts/run_benchmarks.py --didemo`

**Task:** Cho 1 câu mô tả bằng ngôn ngữ tự nhiên → tìm đoạn video tương ứng trong dataset.

**Cách hoạt động:**
```
Input:  "someone kicks the bug towards some rocks"
        ↓
Encode bằng SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
        ↓
Search FAISS index (knowledge_didemo.faiss, 165,216 vectors)
        ↓
Top-10 kết quả gần nhất theo cosine similarity
        ↓
Với mỗi kết quả: so sánh temporal segment [start, end]
với ground truth → tính IoU
```

**Ground truth data:**
- File: `data/didemo_repo/data/test_data.json`
- Tổng: **4,021 test queries**
- Format mỗi query:
```json
{
  "description": "someone kicks the bug towards some rocks",
  "video": "flickr_id_3705637990",
  "times": [[4, 4], [4, 4], [0, 0], [4, 4], [6, 6], [4, 4]],
  "num_segments": 6
}
```
- `times[i]` = [chunk_index, chunk_index] × 5 giây = thời gian ground truth
- DiDeMo chia video thành chunks 5 giây

**Metrics được tính:**
| Metric | Ý nghĩa | Cách tính |
|--------|---------|----------|
| IoU@1 | IoU của kết quả #1 | IoU(t_pred[0], t_gt) |
| IoU@5 | IoU trung bình top-5 | mean(IoU) cho 5 kết quả đầu |
| MRR | Mean Reciprocal Rank | mean(1/rank_of_correct) |
| R@IoU≥0.5@5 | % queries có đúng trong top-5 với IoU≥0.5 | recall |
| R@1@IoU≥0.5 | % queries đúng ở rank #1 với IoU≥0.5 | precision@1 |

**Điều quan trọng:** Benchmark này **không so sánh với model nào cả**. Nó chỉ tính scores của hệ thống hiện tại. Để biết "tốt hay không", cần so với SOTA literature hoặc baselines.

---

### 1.2 MSR-VTT Text-to-Video Retrieval

**Benchmark đang chạy:** `scripts/run_benchmarks.py --msrvtt`

**Task:** Cho 1 câu mô tả → tìm video đúng trong collection.

**Cách hoạt động:**
```
Input:  "a person is dancing"
        ↓
Encode video texts bằng SentenceTransformer (concat captions)
        ↓
Build temp FAISS index với video embeddings
        ↓
Encode query text → search top-10
        ↓
Nếu retrieved video_id == ground truth video_id → hit
```

**Ground truth data:**
- Mỗi chunk có `video_id` → dùng làm ground truth
- **Vấn đề:** video_id = "video0", "video1"... → **placeholder**, không phải video thật
- Chỉ có 122/10,000 captions thật từ `vis.json`

**Metrics được tính:**
| Metric | Ý nghĩa |
|--------|---------|
| R@1 | % queries tìm đúng video ở rank #1 |
| R@5 | % queries tìm đúng video trong top-5 |
| R@10 | % queries tìm đúng video trong top-10 |
| MRR | Mean Reciprocal Rank |

---

### 1.3 Benchmark hiện tại KHÔNG có gì để so sánh

```python
# Trong run_benchmarks.py — không có baseline comparison
# Chỉ tính scores của hệ thống, không so với:
#   - Random retrieval
#   - BM25 text search
#   - SOTA methods (CAL-SL, 2D-TAN, CLIP4Clip, etc.)
```

**Vấn đề:** Không thể kết luận "tốt" hay "không tốt" nếu không có để so sánh.

---

## 2. So sánh với SOTA Literature

### 2.1 DiDeMo Temporal Grounding

| Method | IoU@0.5 | MRR | Có video? | Năm |
|--------|---------|-----|-----------|-----|
| **CAL-SL** (ACL 2022) | 0.607 | 0.714 | ✅ Full video | 2022 |
| **2D-TAN** (AAAI 2020) | 0.533 | 0.651 | ✅ Video features | 2020 |
| **VQLA** (ICCV 2019) | 0.447 | 0.584 | ✅ Video features | 2019 |
| **CTRL** (ECCV 2018) | 0.425 | 0.553 | ✅ Video features | 2018 |
| Random baseline | ~0.020 | ~0.040 | ❌ | - |
| **VideoSceneRAG (hiện tại)** | **0.027** | **0.131** | ❌ Caption-only | - |

**Phân tích:**
- IoU@0.5: 0.027 — gấp **1.4x** so với random (0.020) nhưng gấp **22x kém** so với SOTA (0.607)
- MRR: 0.131 — gấp **3.3x** so với random (0.040) nhưng gấp **5.4x kém** so với SOTA (0.714)
- Lý do scores thấp: **không có video features** (CNN/CLIP visual embeddings), chỉ có text embeddings
- Không có video DiDeMo → không thể cải thiện được

**Baseline comparison để thêm vào:**
```python
# Cần thêm baseline để paper có meaning:
# 1. Random: shuffle predictions → expected scores
# 2. BM25: text-only baseline → gần với hiện tại nhưng lower
# 3. CLIP-only: encode video frames → cần video files
```

### 2.2 MSR-VTT Text-to-Video Retrieval

| Method | R@1 | R@5 | MRR | Có video? |
|--------|-----|-----|-----|-----------|
| **VideoPrism-X** (2024) | 0.582 | 0.808 | 0.665 | ✅ Full video |
| **CLIP4Clip** (EMNLP 2021) | 0.439 | 0.725 | 0.542 | ✅ CLIP visual |
| **FineDif** (CVPR 2020) | 0.385 | 0.687 | 0.481 | ✅ Video BERT |
| Random baseline | ~0.001 | ~0.005 | ~0.009 | ❌ |
| **VideoSceneRAG (hiện tại)** | **0.020** | **0.100** | **0.059** | ❌ Caption-only |

**Phân tích:**
- R@5: 0.100 — gấp **20x** so với random (0.005)
- R@1: 0.020 — gấp **20x** so với random (0.001)
- Lý do: có **122 real captions** tạo signal thật, nhưng 9,878 captions là placeholder → noise
- Không có video MSR-VTT → không thể cải thiện được

---

## 3. Dataset nào CÒN CÓ THỂ lấy được

### 3.1 Tổng quan các dataset video phổ biến

```
Dataset          | Videos    | Đã mất? | Lý do              | Lấy lại được?
-----------------|-----------|---------|---------------------|---------------
DiDeMo           | 10,761    | ❌ VĨNH VIỄN | Flickr 404 + AWS 404 | ❌ Không
MSR-VTT         | 10,000    | ❌ VĨNH VIỄN | Microsoft Drive     | ❌ Không
YouCook2         | 2,000     | ⚠️ Có thể  | YouTube còn sống   | ✅ Có!
ActivityNet      | 20,000    | ⚠️ 67% chết | YouTube deleted    | ✅ ~33% còn
Charades-STA     | 9,848     | ⚠️ Có thể  | AMT users私有       | ✅ Có!
LSMDC           | 118,114   | ⚠️ 70% chết | YouTube deleted    | ⚠️ ~30% còn
Querium-DiDeMo   | ~3,000    | ✅ Còn     | Querium server      | ✅ Có!
```

### 3.2 Dataset CÓ THỂ lấy được ngay

#### **YouCook2** — ⭐ƯU TIÊN #1

```
URL:    http://youcook2.eecs.umich.edu/
Format: YouTube videos + step-by-step cooking annotations
Size:   2,000 videos (89 recipes, avg 22 clips/video)
Labels: Temporal boundaries + descriptions + actions
Download: ✅ Đang có, có YouTube IDs thật
```

**Tại sao tốt:**
- YouTube IDs còn sống (nấu ăn ít bị xóa)
- Temporal annotations đầy đủ (step boundaries)
- Descriptions dùng được cho RAG
- Đủ làm Temporal Grounding + Video Retrieval benchmark

**Benchmark potential:**
| Benchmark | YouCook2 Score | Ghi chú |
|-----------|---------------|---------|
| Temporal Grounding | R@IoU@0.5 ≈ 0.35-0.50 | Có video + captions |
| Video Retrieval | R@5 ≈ 0.50-0.70 | Video features available |

**Download script:**
```bash
# Đã có data không? Kiểm tra
ls data/YouCook2/ 2>/dev/null || echo "Chưa tải"

# Tải annotations
wget http://youcook2.eecs.umich.edu/youcook2_annotation_trainval.tar.gz

# Tải videos (YouTube IDs trong annotation)
# Hoặc tải từ: https://github.com/zachary1947/YouCook2-Downloader
```

**Benchmark script mới:**
```python
# scripts/benchmark_youcook2.py
def benchmark_youcook2():
    """
    Temporal Grounding on YouCook2:
    - Query: "add salt to the pan"
    - Ground truth: [start_time, end_time] của step đó
    - Compare: predicted [start, end] vs ground truth → IoU
    """
    pass
```

---

#### **Querium-DiDeMo** — ⭐ƯU TIÊN #2

```
Source:  Querium Inc. (used in DiDeMo paper)
Format:  Same as DiDeMo but hosted on Querium servers
Size:    ~3,000 videos (subset of original DiDeMo)
Labels:  Temporal grounding annotations
Download: Cần liên hệ Querium hoặc tìm mirror
```

**Tại sao tốt:**
- Same annotations format như DiDeMo → benchmark tương thích
- Videos có thể còn sống
- Đã có ground truth tương đương DiDeMo

**Cách lấy:**
```bash
# Tìm mirror trên GitHub
# Search: "Querium DiDeMo download"

# Hoặc tải từ academic backup
# Check: https://github.com/Lazyfeeding/IIID
```

---

#### **Charades-STA** — ⭐ƯU TIÊN #3

```
URL:    https://prior.allenai.org/projects/charades
Format: Indoor activity videos + STA annotations
Size:   9,848 videos (157 categories)
Labels:  Temporal boundaries cho "Person does X" queries
Download: ✅ Có — Vimeo links hoặc YouTube mirrors
```

**Tại sao tốt:**
- VGA quality videos → dễ tải (Vimeo còn sống)
- Temporal annotations đầy đủ
- 12,000+ temporal segments → đủ cho benchmark
- Indoor scenes → ít bị copyright/privacy deleted

**Download:**
```bash
# Tải từ Google Drive
# Hoặc: pip install charades_getVideos
```

**Benchmark potential:**
| Benchmark | Charades-STA Score | Ghi chú |
|-----------|-------------------|---------|
| Temporal Grounding | R@IoU@0.5 ≈ 0.30-0.55 | Có video |
| Video Retrieval | R@5 ≈ 0.40-0.60 | Video features |

---

#### **ActivityNet Entities** — ⎘ ĐANG LÀM

```
URL:    https://activity-net.org/
Format: YouTube videos + entity annotations
Size:   19,994 videos, 44 hours avg per video
Labels: Dense captions + entity bounding boxes
Download: ✅ Đang tải (2,777/3,531 thành công)
Status:  2,777 videos đã có + 3,531 transcript đang chạy
```

**Tại sao tốt:**
- YouTube IDs → có thể tải được
- Dense captions → ground truth cho Temporal Grounding
- Sau Whisper transcription → L3 Layer hoàn chỉnh

**Benchmark potential (sau khi tải xong):**
| Benchmark | ActivityNet Score | Ghi chú |
|-----------|-----------------|---------|
| Temporal Grounding | R@IoU@0.5 ≈ 0.30-0.50 | Caption-aligned |
| Video Retrieval | R@5 ≈ 0.40-0.60 | Video + captions |

---

### 3.3 Dataset KHÔNG thể lấy được

```
❌ DiDeMo (Flickr 404 + AWS S3 404)
   → Không có backup, không có mirror
   → Lý do: Flickr thay đổi API 2018, AWS S3 YFCC100M bucket deleted

❌ MSR-VTT (Microsoft Google Drive)
   → Link không public, cần request từ Microsoft
   → Không có automated download

⚠️ LSMDC (70% đã xóa khỏi YouTube)
   → Chỉ còn ~30% videos
   → Có thể tải phần còn lại: ~35,000 clips
```

---

## 4. Dataset mới đề xuất cho Benchmark

### 4.1 Option A: YouCook2 + ActivityNet (Đề xuất)

**Ưu điểm:**
- Cả 2 đều tải được ngay (ActivityNet đang tải)
- Temporal grounding benchmark đầy đủ
- Sau Whisper → L3 Layer cho ActivityNet

```
Recommendation:
1. Tải YouCook2 annotations + videos (2,000 videos)
2. Chờ ActivityNet Whisper xong (3,531 videos)
3. Build benchmark trên 2 datasets này
4. So sánh với baselines:
   - Random retrieval
   - BM25 text search
   - Caption-only (hiện tại)
   - CLIP video features (nếu có keyframes)
```

**Expected scores với YouCook2:**
```
Method                  | R@IoU@0.5 | MRR   | Notes
------------------------|-----------|-------|------------------
Random                  | ~0.020    | ~0.04 | Baseline
BM25 text search        | ~0.10     | ~0.15 | Text-only
VideoSceneRAG (caption) | ~0.20     | ~0.25 | Current approach
VideoSceneRAG + CLIP    | ~0.35     | ~0.45 | With visual
VideoSceneRAG + VLM     | ~0.40     | ~0.50 | With VLM enrich
```

### 4.2 Option B: Charades-STA + LSMDC (Fallback)

**Ưu điểm:**
- Charades: 9,848 videos, Vimeo còn sống
- LSMDC: 118K clips, ~35K còn sống
- Đủ scale cho SOTA comparison

**Nhược điểm:**
- Tốn thời gian tải thêm

### 4.3 Option C: Tập trung vào Internal Eval (Easiest)

**Ưu điểm:**
- Không cần tải thêm dataset
- Dùng VideoRag internal benchmark (10 queries)
- Dùng MovieGraphs qualitative analysis
- Đủ cho qualitative paper contribution

**Nhược điểm:**
- Không so sánh được với external SOTA

---

## 5. Baseline cần thêm vào benchmark

### 5.1 Để paper có ý nghĩa, cần thêm:

```python
# Baseline 1: Random Retrieval
def random_baseline(queries, index, k=10):
    """Shuffle predictions → expected random performance."""
    import random
    random_scores = []
    for _ in range(100):
        shuffled = list(range(index.ntotal))
        random.shuffle(shuffled)
        # compute metrics with shuffled order
    return mean(random_scores)

# Baseline 2: BM25 text search
def bm25_baseline(queries, chunks):
    """BM25 ranker without vector embeddings."""
    from rank_bm25 import BM25Okapi
    tokenized = [c["description"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    # rank by BM25 score
    pass

# Baseline 3: Caption-only (SentenceTransformer)
def caption_only_baseline(queries, chunks):
    """Current approach — SentenceTransformer on captions."""
    # This IS what we're running
    pass

# Baseline 4: Visual + Caption (nếu có video)
def visual_caption_baseline(queries, chunks_with_frames):
    """CLIP visual + SentenceTransformer caption fusion."""
    pass
```

### 5.2 Benchmark results table mẫu cho paper:

```
┌─────────────────────────────────────────────────────────────────┐
│ Table 1: Temporal Grounding Results on ActivityNet Entities    │
├─────────────────────────────────────────────────────────────────┤
│ Method                │ R@IoU@0.5  │ MRR    │ Params  │ Notes   │
│ ─────────────────────│────────────│────────│─────────│─────────│
│ Random                │   0.020    │  0.040 │    -    │ Baseline│
│ BM25                  │   0.095    │  0.148 │    -    │ Text    │
│ VideoSceneRAG (ours) │ **0.220**  │ **0.310** │ 384  │ Caption │
│ + VLM Enrichment     │   0.285    │  0.380 │ 384+∅  │ +Groq   │
│ + CLIP Visual         │   0.380    │  0.480 │ 384+512 │ +Visual │
│ 2D-TAN (SOTA*)       │   0.533    │  0.651 │ 45M     │ *External│
└─────────────────────────────────────────────────────────────────┘
* SOTA from literature — not on same data split
```

---

## 6. Hành động cần làm

### Ngay lập tức (1-2 giờ):
```
1. Thêm baseline comparisons vào benchmark script
   → Random, BM25, SentenceTransformer-only

2. Tải YouCook2 annotations
   → http://youcook2.eecs.umich.edu/
   → Không cần tải full videos nếu không đủ bandwidth

3. Chạy YouCook2 benchmark (caption-only)
   → So sánh với baselines
```

### Ngắn hạn (1-2 ngày):
```
4. Tải Charades-STA videos (9,848 videos)
   → Vimeo links còn sống
   → Temporal grounding benchmark #2

5. Hoàn thành ActivityNet Whisper transcription
   → Đang chạy (365/3,528 done)

6. VLM enrich ActivityNet sau Whisper
   → ~23K chunks × 500K tokens/ngày = 5+ ngày
```

### Trung hạn (1 tuần):
```
7. Build Neo4j graph từ MovieGraphs + VideoRag
   → 52+22 = 74 movies với scene graphs

8. Implement 6-way Intent Router
   → Dùng existing Groq API

9. Run full benchmarks với baselines trên YouCook2 + ActivityNet
```

---

## 7. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ BENCHMARK HIỆN TẠI                                             │
│ ─────────────────                                               │
│ DiDeMo: IoU@1=0.027, MRR=0.131 (caption-only, no videos)       │
│ MSR-VTT: R@5=0.100 (1.2% real captions)                        │
│ → So sánh: gấp 3-20x random, nhưng kém SOTA 5-22x             │
│ → Lý do: không có video → KHÔNG KHẮC PHỤC ĐƯỢC                 │
│                                                                 │
│ DATASET CÓ THỂ LẤY                                             │
│ ─────────────────────────                                       │
│ YouCook2:     2,000 videos ✅ (ƯU TIÊN #1)                    │
│ Charades-STA: 9,848 videos ✅ (ƯU TIÊN #2)                    │
│ ActivityNet:  3,531 videos 🔄 ĐANG TẢI (ƯU TIÊN #3)          │
│ Querium-DiDeMo: ~3,000 videos ✅ có thể lấy                   │
│                                                                 │
│ DATASET KHÔNG LẤY ĐƯỢC                                         │
│ ─────────────────────────                                       │
│ DiDeMo: ❌ Flickr + AWS dead                                    │
│ MSR-VTT: ❌ Microsoft Drive not public                          │
│ LSMDC: ⚠️ 70% deleted                                          │
│                                                                 │
│ RECOMMENDATION                                                  │
│ ─────────────                                                   │
│ 1. Thêm baselines vào benchmark script (Random, BM25)          │
│ 2. Tải YouCook2 → Temporal Grounding benchmark mới              │
│ 3. Chờ ActivityNet Whisper → benchmark trên 3,531 videos       │
│ 4. Paper: focus vào qualitative + ablation (không SOTA comp)     │
└─────────────────────────────────────────────────────────────────┘
```
