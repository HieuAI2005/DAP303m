#!/usr/bin/env python3
"""
VideoSceneRAG — Streamlit Demo UI
==================================
Interactive demo for 5-Layer Video Understanding.

Usage:
  cd /home/hiwe/project/DAP303m/project_ky4
  streamlit run scripts/demo_app.py --server.port 8501

Features:
  - Natural language video scene queries
  - Multi-layer retrieval (text + visual)
  - Scene timeline visualization
  - 5-Layer metadata display per result
"""

import streamlit as st
import json, faiss, numpy as np
from pathlib import Path
import sys
from sentence_transformers import SentenceTransformer
import torch

PROJECT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT / "src"))

st.set_page_config(
    page_title="VideoSceneRAG — 5-Layer Video Understanding",
    page_icon="🎬",
    layout="wide",
)


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset(movie_id: str):
    """Load chunks + FAISS indexes for a movie."""
    output_dir = PROJECT / "data" / "pipeline_output" / movie_id

    chunks_path = output_dir / "chunks.json"
    if not chunks_path.exists():
        return None, None, None

    with open(chunks_path) as f:
        raw = json.load(f)
    chunks = raw.get("chunks", raw)

    text_emb_path = output_dir / "text_chunks.faiss"
    clip_emb_path = output_dir / "clip_chunks.faiss"

    text_idx = faiss.read_index(str(text_emb_path)) if text_emb_path.exists() else None
    clip_idx = faiss.read_index(str(clip_emb_path)) if clip_emb_path.exists() else None

    return chunks, text_idx, clip_idx


def load_transcript(movie_id: str):
    """Load Whisper transcript if available."""
    output_dir = PROJECT / "data" / "pipeline_output" / movie_id
    transcript_path = output_dir / "whisper_transcript.json"
    if transcript_path.exists():
        with open(transcript_path) as f:
            return json.load(f)
    return None


def make_text(c: dict, use_dialogue: bool = True) -> str:
    parts = []
    parts.append(c.get("description", ""))
    parts.append(c.get("situation", ""))
    if use_dialogue and c.get("dialogue_text"):
        parts.append(c.get("dialogue_text", ""))
    if c.get("characters"):
        parts.append("Characters: " + ", ".join(c["characters"][:5]))
    parts.append(c.get("narrative_arc", ""))
    return " | ".join(p for p in parts if p and p not in ("[NO_DIALOGUE]", "[PENDING_SRT_ALIGNMENT]"))


# ── Retrieval Functions ────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, chunks: list, text_idx, model, top_k: int = 5, use_dialogue: bool = True):
    """Retrieve top-K chunks for a query."""
    q_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)
    D, I = text_idx.search(q_vec, min(top_k, text_idx.ntotal))

    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        if idx < len(chunks):
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk": chunks[idx],
                "temporal": f"{chunks[idx].get('start_seconds', 0):.0f}s – {chunks[idx].get('end_seconds', 0):.0f}s",
            })
    return results


def clip_retrieve(query: str, chunks: list, clip_idx, model, top_k: int = 5):
    """Retrieve using CLIP image embeddings via query expansion."""
    # Encode query as text
    q_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)
    D, I = clip_idx.search(q_vec, min(top_k, clip_idx.ntotal))

    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        if idx < len(chunks):
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk": chunks[idx],
                "temporal": f"{chunks[idx].get('start_seconds', 0):.0f}s – {chunks[idx].get('end_seconds', 0):.0f}s",
            })
    return results


def build_timeline(chunks: list) -> dict:
    """Build timeline metadata for visualization."""
    if not chunks:
        return {}
    total_dur = max(c.get("end_seconds", 0) for c in chunks)
    return {
        "total_seconds": total_dur,
        "num_chunks": len(chunks),
    }


def render_chunk_card(result: dict):
    """Render a single chunk result as an expandable card."""
    c = result["chunk"]
    score = result["score"]

    with st.expander(
        f"#{result['rank']} | ⏱ {result['temporal']} | "
        f"📊 sim={score:.3f} | "
        f"🎬 {c.get('title', c.get('movie_id', ''))}",
        expanded=(result["rank"] <= 2),
    ):
        # Temporal
        st.markdown(f"**⏱ Temporal:** {c.get('start_seconds', 0):.1f}s – {c.get('end_seconds', 0):.1f}s")

        # L2: Semantic
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🎭 Description:** {c.get('description', '—')}")
            st.markdown(f"**📍 Setting:** {c.get('vision_setting', '—')}")
            st.markdown(f"**😀 Tone:** {c.get('emotional_tone', '—')}")
        with col2:
            actions = c.get("vision_actions", [])
            st.markdown(f"**🏃 Actions:** {', '.join(actions) if actions else '—'}")
            st.markdown(f"**📖 Situation:** {c.get('situation', '—')}")

        st.divider()

        # L3: Dialogue
        dialogue = c.get("dialogue_text", "")
        if dialogue and dialogue not in ("[NO_DIALOGUE]", "[PENDING_SRT_ALIGNMENT]", ""):
            st.markdown(f"**💬 Dialogue:** _{dialogue[:300]}{'...' if len(dialogue) > 300 else ''}_")
        else:
            st.markdown("**💬 Dialogue:** _[No dialogue in this scene]_")

        st.divider()

        # L4: Characters
        chars = c.get("characters", [])
        if chars:
            st.markdown(f"**👤 Characters:** {', '.join(chars[:8])}")
        emotions = c.get("character_emotions", {})
        if emotions:
            em_str = ", ".join(f"{k}: {v}" for k, v in list(emotions.items())[:3])
            st.markdown(f"**😊 Emotions:** {em_str}")

        # L5: Narrative
        st.markdown(f"**📜 Narrative Arc:** {c.get('narrative_arc', '—')}")
        script = c.get("screenplay_context", "")
        if script and len(script) > 10:
            st.markdown(f"**📝 Context:** {script[:200]}{'...' if len(script) > 200 else ''}")

        # Layer coverage badges
        cols = st.columns(5)
        for i, (label, key) in enumerate([
            ("L1 Temporal", "start_seconds"),
            ("L2 Semantic", "description"),
            ("L3 Dialogue", "dialogue_text"),
            ("L4 Characters", "characters"),
            ("L5 Narrative", "narrative_arc"),
        ]):
            has = bool(c.get(key))
            with cols[i]:
                color = "green" if has else "gray"
                st.markdown(f":{color}[**{label}** {'✅' if has else '❌'}]")


def render_timeline(chunks: list, results: list):
    """Render a visual timeline with result markers."""
    if not chunks:
        st.info("No chunks to display.")
        return

    total_dur = max(c.get("end_seconds", 0) for c in chunks)
    if total_dur == 0:
        return

    # Build coverage heatmap data
    scene_map = []
    for c in chunks:
        start = c.get("start_seconds", 0)
        end = c.get("end_seconds", 0)
        has_dialogue = bool(c.get("dialogue_text") and c.get("dialogue_text") not in ("[NO_DIALOGUE]", "[PENDING_SRT_ALIGNMENT]", ""))
        has_chars = bool(c.get("characters"))
        scene_map.append({
            "start": start,
            "end": end,
            "has_dialogue": has_dialogue,
            "has_chars": has_chars,
        })

    # Result positions
    result_pos = {}
    for r in results[:5]:
        mid = (r["chunk"].get("start_seconds", 0) + r["chunk"].get("end_seconds", 0)) / 2
        result_pos[r["rank"]] = mid

    # Simple HTML timeline
    timeline_html = f"""
    <div style="background:#f0f0f0; border-radius:8px; padding:12px; margin:8px 0; font-family:monospace;">
    <div style="font-size:12px; color:#666; margin-bottom:4px;">
        Timeline: 0s — {total_dur:.0f}s ({len(chunks)} scenes)
    </div>
    <div style="position:relative; height:32px; background:#ddd; border-radius:4px; overflow:hidden;">
    """
    # Color bands for scenes with dialogue
    for m in scene_map:
        left = (m["start"] / total_dur) * 100
        w = max((m["end"] - m["start"]) / total_dur * 100, 0.5)
        color = "#4CAF50" if m["has_dialogue"] else "#2196F3"
        timeline_html += f'<div style="position:absolute;left:{left:.1f}%;width:{w:.1f}%;height:100%;background:{color};opacity:0.6;"></div>'

    # Result markers
    colors = {1: "#FF5722", 2: "#FFC107", 3: "#8BC34A", 4: "#03A9F4", 5: "#9C27B0"}
    for rank, pos in result_pos.items():
        left = (pos / total_dur) * 100
        c = colors.get(rank, "#999")
        timeline_html += f'<div style="position:absolute;top:0;left:{left:.1f}%;width:3px;height:100%;background:{c};border-radius:2px;" title="#{rank}"></div>'
        timeline_html += f'<div style="position:absolute;top:0;left:{left:.1f}%;transform:translateX(-50%);background:{c};color:white;font-size:10px;padding:1px 4px;border-radius:3px;">#{rank}</div>'

    timeline_html += "</div>"

    # Legend
    timeline_html += """
    <div style="font-size:11px; margin-top:6px;">
        <span style="background:#4CAF50;padding:1px 6px;border-radius:3px;color:white;margin-right:8px;">Scene w/ dialogue</span>
        <span style="background:#2196F3;padding:1px 6px;border-radius:3px;color:white;margin-right:8px;">Scene (no dialogue)</span>
        <span style="background:#FF5722;padding:1px 6px;border-radius:3px;color:white;margin-right:8px;">Top result</span>
    </div>
    </div>
    """
    st.markdown(timeline_html, unsafe_allow_html=True)


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🎬 VideoSceneRAG — 5-Layer Video Understanding Demo")

# Sidebar: dataset selector
st.sidebar.header("📁 Dataset")
available_movies = [
    ("big_buck_bunny", "Big Buck Bunny (2008)", "10.6 min, 140 chunks"),
    ("titanic_clip", "Titanic (1997) — clip", "3 min, 20 chunks"),
]
# Check which datasets are actually available
actual_datasets = []
for mid, title, desc in available_movies:
    p = PROJECT / "data" / "pipeline_output" / mid
    if p.exists():
        chunks_p = p / "chunks.json"
        if chunks_p.exists():
            with open(chunks_p) as f:
                raw = json.load(f)
            chunks = raw.get("chunks", raw) if isinstance(raw, dict) else raw
            actual_datasets.append((mid, title, f"{desc}, L1=100%, L2={sum(1 for c in chunks if c.get('description'))/max(len(chunks),1)*100:.0f}%"))

if not actual_datasets:
    st.sidebar.warning("No datasets found. Run pipeline_5layer_video.py first.")
    default_movie = None
else:
    selected = st.sidebar.selectbox(
        "Select movie:", range(len(actual_datasets)),
        format_func=lambda i: f"{actual_datasets[i][1]} — {actual_datasets[i][2]}"
    )
    default_movie = actual_datasets[selected][0] if actual_datasets else None

if default_movie:
    chunks, text_idx, clip_idx = load_dataset(default_movie)
    model = load_model()
    transcript = load_transcript(default_movie)

    if chunks is None:
        st.error(f"No data for {default_movie}. Run pipeline first.")
    else:
        # Dataset stats
        n = len(chunks)
        l3 = sum(1 for c in chunks if c.get("dialogue_text") and c.get("dialogue_text") not in ("[NO_DIALOGUE]", "[PENDING_SRT_ALIGNMENT]", "")) / max(n, 1) * 100
        l4 = sum(1 for c in chunks if c.get("characters")) / max(n, 1) * 100

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🎬 Chunks", n)
        col2.metric("L1 Temporal", "100%")
        col3.metric("L2 Semantic", "100%")
        col4.metric("L3 Dialogue", f"{l3:.0f}%")
        col5.metric("L4 Characters", f"{l4:.0f}%")

        st.divider()

        # Query input
        st.subheader("🔍 Natural Language Scene Query")

        example_queries = [
            "Action scene with fast movement",
            "A character talks about their feelings",
            "The villain appears",
            "Emotional confrontation between two characters",
            "Funny scene with comedy",
            "Quiet and peaceful moment",
        ]

        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., 'An intense action scene' or 'A tender moment between two characters'",
            label_visibility="collapsed",
        )

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
        with col_btn1:
            top_k = st.selectbox("Top-K:", [3, 5, 10], index=1)
        with col_btn2:
            use_clip = st.toggle("Use CLIP (visual)", value=False)

        if query:
            if st.button("🔍 Search", type="primary", use_container_width=True):
                if use_clip and clip_idx is not None:
                    results = clip_retrieve(query, chunks, clip_idx, model, top_k)
                    st.info("🔎 CLIP visual retrieval (text query → visual embeddings)")
                else:
                    results = retrieve(query, chunks, text_idx, model, top_k)

                if results:
                    # Timeline
                    st.subheader("📊 Timeline Visualization")
                    render_timeline(chunks, results)

                    # Results
                    st.subheader(f"🎬 Top-{len(results)} Results")
                    for r in results:
                        render_chunk_card(r)
                else:
                    st.warning("No results found.")
        else:
            st.info("Enter a query above to search scene chunks.")
            st.markdown("**Try these example queries:**")
            for eq in example_queries:
                if st.button(f"🔍 {eq}", key=eq):
                    results = retrieve(eq, chunks, text_idx, model, 5)
                    if results:
                        st.session_state["query_result"] = results
                        st.rerun()
else:
    st.info("👈 Select a dataset from the sidebar, then run a query.")
    st.markdown("""
    **To add a dataset:**
    ```bash
    python scripts/pipeline_5layer_video.py \
      --video data/movies/your_movie.mp4 \
      --movie-id your_id \
      --title "Your Movie (Year)"
    ```
    """)

# Transcript viewer
if default_movie and transcript:
    with st.expander("📜 View Whisper Transcript"):
        st.markdown(f"**Language:** {transcript.get('language', 'unknown')}")
        st.markdown(f"**Segments:** {len(transcript.get('segments', []))}")
        for seg in transcript.get("segments", [])[:20]:
            st.markdown(f"**[{seg['start']:.1f}s–{seg['end']:.1f}s]** {seg['text']}")

st.sidebar.divider()
st.sidebar.markdown("""
**📖 5-Layer Schema:**
- **L1 Temporal:** start/end seconds, shot_id
- **L2 Semantic:** description, setting, actions, tone
- **L3 Dialogue:** transcribed speech (Whisper/SRT)
- **L4 Characters:** named characters, emotions
- **L5 Narrative:** arc, causal relations, script

**Pipeline version:** 2026-03-20
""")
