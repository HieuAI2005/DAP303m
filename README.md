# MovieRAG – Multimodal Movie Retrieval & QA

MovieRAG là hệ thống truy xuất và hiểu phim đa phương thức. Hệ thống nhận **ảnh/video + câu hỏi**, trả về **phim đích + timestamp + câu trả lời**, có thể kèm bằng chứng hình ảnh và clip ngắn.

---

## Highlights

- Ingest pipeline đầy đủ: shot detection → semantic segmentation → keyframes → temporal chunks → FAISS + Graph.
- Visual Retrieval: CLIP + FAISS cho ảnh/video.
- Knowledge RAG: metadata, subtitle, script, graph.
- Agentic pipeline: route → retrieve → grade → generate.
- Demo UI: Gradio chat + gallery + clip.

---

## Repository Structure (rút gọn)

- `src/preprocess_data/` – Ingest pipeline (step 1–8) và indexing.
- `src/preprocess_data/preprocessing/` – Visual pruning pipeline tích hợp.
- `src/movierag/` – Runtime retrieval + agentic pipeline + UI.
- `data/` – Dữ liệu, indexes, temporal chunks, logs.
- `docs/` – Tài liệu nghiên cứu và báo cáo.
- `notebooks/` – Notebook chạy full ingest.

---

## Prerequisites

- Python 3.10+ (khuyến nghị dùng môi trường riêng).
- FFmpeg (để trích keyframe/clip).
- CUDA nếu chạy trên GPU (không bắt buộc).

Các thư viện thường dùng:
- `torch`, `faiss`, `opencv-python`, `scenedetect`, `facenet-pytorch`, `scikit-learn`.
- LLM clients: Gemini / Groq (tùy chọn).

---

## Environment Variables

Đặt trong `src/.env` hoặc `.env` ở project root.

- `GEMINI_API_KEY=...`
- `GROQ_API_KEY=...`
- `TMDB_API_KEY=...`
- `NEO4J_URI=bolt://localhost:7688`
- `NEO4J_USER=neo4j`
- `NEO4J_PASS=...`
- `MOVIERAG_DEVICE=cuda|cpu`

---

## Quick Start – Full Ingest Pipeline

Script đầy đủ chạy ingest nằm ở:
- `run_full_pipeline_tt0167404.py`

Notebook dễ kiểm tra tại:
- `notebooks/movierag_full_ingest_tt0167404.ipynb`

Flow ingest đầy đủ:
1. Copy video
2. Fetch metadata
3. Shot detection
4. Subtitle/STT
5. Semantic segmentation
6. Clip video
7. Keyframe extraction
8. Visual pruning (tích hợp)
9. VLM + Fusion + Chunk
10. Index (FAISS + Graph)

---

## Data Layout (quan trọng)

- Raw videos: `data/raw_videos/`
- Subtitle: `data/movienet_subset/subtitle/`
- Annotation: `data/movienet_subset/annotation/`
- Output ingest mặc định: `data/pipeline_output/`
- Output test: `data/pipeline_full_test/` hoặc `data/temp_pipeline/`
- Temporal chunks: `data/pipeline_output/temporal_chunks/`
- Indexes: `data/pipeline_output/indexes/`

---

## Evaluation

Eval dataset hiện tại:
- `data/eval_queries.json`

Eval runner:
- `python -m movierag.main eval --dataset data/eval_queries.json`

Metrics chính:
- Acc@K
- MAE
- IoU
- LLM Judge: accuracy/context/detail

---

## Demo UI

Chạy Gradio demo:
- `python -m movierag.main demo`

---

## Troubleshooting (pipeline bị kẹt)

Nguyên nhân phổ biến:
- LLM API bị 403/429 → step 4b (semantic segmentation) retry nhiều lần.
- Thiếu subtitle trong OUTPUT_DIR → build chunks = 0.
- Thiếu `keyframe_index.json` → step 5b/6a fail.

Xem log:
- `pipeline_full_run_v3.log`

---

## Documentation

- Task definition: `docs/task_definition.md`
- System design: `docs/system_desgin.md`
- Research summary: `docs/integrated_research_summary.md`
- Báo cáo kỹ thuật: `docs/report/report.md`
- Outline slide: `docs/report/slide_outline.md`

---

## Notes

- `src/preprocess_data/preprocessing/` là phần pruning tích hợp vào flow ingest.
- Agentic pipeline runtime nằm ở `src/movierag/pipeline/agentic_pipeline.py`.

---

## License

N/A (project nội bộ).
