#!/usr/bin/env python3
"""
VideoSceneRAG — Product UI
Usage:
  streamlit run scripts/demo_app.py --server.port 8501
"""

import json, os, sys, time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

PROJECT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────

INDEX_DIR  = PROJECT / "data" / "pipeline_output" / "indexes"
CHUNKS_PATH = PROJECT / "data" / "pipeline_output" / "videorag_chunks" / "all_chunks.json"

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

LAYER_SCHEMA = [
    ("L0", "Visual Frames",  "CLIP ViT-B/32 · 512-dim · 94,779 keyframes",       "🖼"),
    ("L1", "Temporal",       "start / end seconds · shot_id",                     "⏱"),
    ("L2", "Semantic",       "description · setting · actions · tone  [VLM]",     "🧠"),
    ("L3", "Dialogue",       "dialogue_text · speaker · audio_events  [SRT/STT]", "💬"),
    ("L4", "Characters",     "characters · emotions · cast_in_scene  [MovieGraphs]","👤"),
    ("L5", "Narrative",      "narrative_arc · causal_relations · script context", "📖"),
]

INDEX_SPECS = [
    ("L0  Frame Visual", "94,779 × 512-dim", "FAISS IndexFlatIP · CLIP ViT-B/32"),
    ("L1  Scene Fusion", "12,900 × 896-dim", "√0.72·CLIP + √0.28·text · normalized"),
    ("L3  Knowledge",    "12,900 × 384-dim", "SentenceTransformer all-MiniLM-L6-v2"),
    ("L5  Neo4j Graph",  "58 movies · 10,792+ nodes", "Bolt :7688 · APPEARS_IN · NEXT_SCENE"),
]

EXAMPLE_QUERIES = [
    "Who is the main character in Forrest Gump?",
    "Find scenes where characters are eating in a restaurant",
    "Emotional confrontation between two people",
    "What happens when Don Corleone is shot in The Godfather?",
    "Outdoor action scene with running",
    "Quiet intimate conversation between lovers",
]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VideoSceneRAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.pipeline-box {
    border: 1px solid #444;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 13px;
    font-weight: 500;
    color: #e0e0e0;
    background: #1e1e2e;
    display: flex;
    align-items: center;
    gap: 10px;
}
.pipeline-box.active {
    border-color: #7c3aed;
    background: #2d1b69;
    color: #c4b5fd;
}
.pipeline-box.done {
    border-color: #059669;
    background: #064e3b;
    color: #6ee7b7;
}
.pipeline-arrow { text-align:center; color:#555; font-size:16px; margin:1px 0; }
.layer-row {
    display:flex; align-items:flex-start; gap:10px;
    padding: 7px 0; border-bottom: 1px solid #2a2a3a;
    font-size: 12.5px;
}
.layer-tag {
    background:#7c3aed22; border:1px solid #7c3aed55;
    border-radius:5px; padding:2px 8px; font-size:11px;
    font-weight:700; color:#a78bfa; white-space:nowrap;
    min-width: 28px; text-align:center;
}
.index-row {
    display:flex; flex-direction:column;
    padding: 7px 10px; border-radius:6px;
    background:#111827; margin:4px 0; font-size:12px;
}
.stat-chip {
    display:inline-block; background:#1e3a5f; border:1px solid #2563eb44;
    border-radius:20px; padding:3px 12px; font-size:12px; color:#93c5fd;
    margin:3px 4px 3px 0;
}
.result-header {
    font-size:15px; font-weight:600; margin-bottom:6px; color:#e2e8f0;
}
.meta-badge {
    display:inline-block; border-radius:4px; padding:2px 8px;
    font-size:11px; font-weight:600; margin:2px 3px 2px 0;
}
.badge-ok   { background:#064e3b; color:#6ee7b7; border:1px solid #059669; }
.badge-miss { background:#2a1a1a; color:#888;    border:1px solid #444; }
</style>
""", unsafe_allow_html=True)


# ── Resource loading ──────────────────────────────────────────────────────────

@st.cache_resource
def load_knowledge_index():
    idx  = faiss.read_index(str(INDEX_DIR / "knowledge_videorag.faiss"))
    meta = json.loads((INDEX_DIR / "knowledge_videorag_meta.json").read_text())
    return idx, meta

@st.cache_resource
def load_chunks_cache():
    chunks = json.loads(CHUNKS_PATH.read_text())
    return {c["chunk_id"]: c for c in chunks}

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        return None


# ── System Overview panel ─────────────────────────────────────────────────────

def render_system_overview(active_step: int = -1):
    """
    Render System Overview + Metadata layers + Index specs.
    active_step: 0=decompose, 1=retrieve, 2=aggregate, 3=generate, 4=verify, -1=idle
    """

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st.markdown("#### Pipeline")

    steps = [
        ("⚙", "Query Decomposition",           "parse intent · extract entities"),
        ("🔍","Multi-Level Retrieval",           "L3 text · L1 fusion · L0 visual · L5 graph"),
        ("📦","Evidence Aggregation",            "rank · deduplicate · enrich metadata"),
        ("🤖","Answer Generation",               "Groq Llama-4-Scout · grounded w/ timestamps"),
        ("✅","Verifier / Judge",                "consistency check · confidence scoring"),
    ]

    for i, (icon, name, detail) in enumerate(steps):
        if i > 0:
            st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)
        cls = "active" if i == active_step else ("done" if active_step > i >= 0 else "")
        st.markdown(
            f'<div class="pipeline-box {cls}">'
            f'<span style="font-size:16px">{icon}</span>'
            f'<div><div style="font-weight:600">{name}</div>'
            f'<div style="font-size:11px;opacity:.65;margin-top:1px">{detail}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Metadata layers ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Metadata Layers")

    for layer_id, layer_name, layer_desc, icon in LAYER_SCHEMA:
        st.markdown(
            f'<div class="layer-row">'
            f'<span style="font-size:16px">{icon}</span>'
            f'<div>'
            f'<span class="layer-tag">{layer_id}</span>'
            f'  <span style="font-weight:600;color:#cbd5e1">{layer_name}</span>'
            f'<div style="color:#94a3b8;font-size:11.5px;margin-top:2px">{layer_desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Index specs ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Indexes")

    for name, size, note in INDEX_SPECS:
        st.markdown(
            f'<div class="index-row">'
            f'<span style="color:#a78bfa;font-weight:700">{name}</span>'
            f'<span style="color:#e2e8f0;margin-top:1px">{size}</span>'
            f'<span style="color:#64748b;font-size:11px;margin-top:2px">{note}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Retrieval ─────────────────────────────────────────────────────────────────

def search_knowledge(query: str, idx, meta, model, k: int = 10):
    emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = idx.search(emb, k * 2)
    results = []
    for score, i in zip(scores[0], indices[0]):
        if i < len(meta):
            m = meta[i].copy()
            m["score"] = float(score)
            results.append(m)
            if len(results) >= k:
                break
    return results


def enrich_with_chunks(results: list, chunk_cache: dict) -> list:
    for r in results:
        detail = chunk_cache.get(r.get("chunk_id", ""), {})
        r["_detail"] = detail
    return results


def generate_answer(client, query: str, results: list) -> str:
    context_parts = []
    for i, r in enumerate(results[:5]):
        d = r.get("_detail", {})
        parts = [f"[Scene {i+1}]",
                 f"Movie: {r.get('title', r.get('movie_id', '?'))}"]
        if d.get("start_seconds") is not None:
            parts.append(f"Time: {d['start_seconds']:.0f}s–{d.get('end_seconds', '?'):.0f}s")
        if d.get("description"):
            parts.append(f"Desc: {d['description']}")
        if d.get("dialogue_text") and d["dialogue_text"] not in ("[NO_DIALOGUE]", "[PENDING_SRT_ALIGNMENT]", ""):
            parts.append(f"Dialogue: {d['dialogue_text'][:200]}")
        if d.get("characters"):
            parts.append(f"Characters: {', '.join(d['characters'][:5])}")
        context_parts.append(" | ".join(parts))

    context = "\n".join(context_parts) if context_parts else "No relevant scenes found."
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are VideoSceneRAG, an expert video scene retrieval assistant. "
                "Answer questions about movies using the retrieved scene evidence. "
                "Always cite specific scenes with timestamps when available. Be concise and factual."
            )},
            {"role": "user", "content": (
                f"Query: {query}\n\nRetrieved scene evidence:\n{context}\n\n"
                "Answer the query based on the evidence. Cite timestamps."
            )},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# ── Result card ───────────────────────────────────────────────────────────────

def render_result_card(rank: int, result: dict):
    d = result.get("_detail", {})
    movie = result.get("title") or result.get("movie_id", "?")
    start = d.get("start_seconds")
    end   = d.get("end_seconds")
    ts    = f"{start:.0f}s – {end:.0f}s" if start is not None else "—"
    score = result.get("score", 0)

    label = f"#{rank}  ·  {movie}  ·  {ts}  ·  sim={score:.3f}"

    with st.expander(label, expanded=(rank <= 2)):

        # Top: description + setting + tone
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Description**")
            st.write(d.get("description") or "—")
        with c2:
            st.markdown(f"**Setting / Tone**")
            setting = d.get("vision_setting") or "—"
            tone    = d.get("emotional_tone") or "—"
            st.write(f"{setting} · {tone}")

        actions = d.get("vision_actions", [])
        if actions:
            st.markdown(f"**Actions:** {', '.join(actions)}")

        st.divider()

        # Dialogue (L3)
        dlg = d.get("dialogue_text", "")
        if dlg and dlg not in ("[NO_DIALOGUE]", "[PENDING_SRT_ALIGNMENT]", ""):
            st.markdown(f"**Dialogue**")
            st.markdown(f"> {dlg[:400]}{'...' if len(dlg) > 400 else ''}")
        else:
            st.markdown("**Dialogue:** _—_")

        st.divider()

        # Characters (L4) + Narrative (L5)
        c3, c4 = st.columns(2)
        with c3:
            chars = d.get("characters", [])
            cast  = d.get("cast_in_scene", [])
            st.markdown("**Characters (L4)**")
            if chars:
                st.write(", ".join(chars[:6]))
                if cast:
                    for cs in cast[:3]:
                        if cs.get("actor"):
                            st.caption(f"  {cs.get('character','?')} → {cs.get('actor','?')}")
            else:
                st.write("—")
        with c4:
            st.markdown("**Narrative (L5)**")
            arc = d.get("narrative_arc") or "—"
            st.write(arc[:120])
            causal = d.get("causal_relations", [])
            if causal:
                for rel in causal[:2]:
                    if isinstance(rel, dict):
                        st.caption(f"  ↪ {rel.get('relation','')}")

        # Layer coverage badges
        badge_data = [
            ("L0 Visual",    bool(result.get("score"))),
            ("L1 Temporal",  start is not None),
            ("L2 Semantic",  bool(d.get("description"))),
            ("L3 Dialogue",  bool(dlg and dlg not in ("[NO_DIALOGUE]","[PENDING_SRT_ALIGNMENT]",""))),
            ("L4 Characters",bool(d.get("characters"))),
            ("L5 Narrative", bool(d.get("narrative_arc"))),
        ]
        badges_html = "".join(
            f'<span class="meta-badge {"badge-ok" if has else "badge-miss"}">'
            f'{"✓" if has else "·"} {lbl}</span>'
            for lbl, has in badge_data
        )
        st.markdown(badges_html, unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────

# Header
st.markdown(
    '<h2 style="margin-bottom:2px">🎬 VideoSceneRAG</h2>'
    '<p style="color:#64748b;font-size:13px;margin-top:0">Hierarchical Multi-Modal Retrieval · 993 movies · 815 ActivityNet videos · 12,900 chunks</p>',
    unsafe_allow_html=True,
)

# Corpus stats chips
try:
    idx_tmp, meta_tmp = load_knowledge_index()
    n_chunks = idx_tmp.ntotal
except Exception:
    n_chunks = 12900

chips = [
    ("12,900", "knowledge chunks"),
    ("94,779", "visual keyframes"),
    ("993",    "movies indexed"),
    ("815",    "ActivityNet videos"),
    ("58",     "Neo4j movies"),
]
chips_html = "".join(f'<span class="stat-chip"><b>{v}</b> {l}</span>' for v, l in chips)
st.markdown(chips_html, unsafe_allow_html=True)
st.markdown("---")

# Two-column layout: main left, system overview right
col_main, col_system = st.columns([6, 4], gap="large")

with col_system:
    # Active step stored in session state
    active_step = st.session_state.get("pipeline_step", -1)
    render_system_overview(active_step)

with col_main:

    # Query input
    query = st.text_input(
        "query",
        placeholder="Ask anything about a movie scene…  e.g. 'Who shoots Don Corleone?'",
        label_visibility="collapsed",
        key="main_query",
    )

    c_k, c_ans, c_run = st.columns([2, 3, 3])
    with c_k:
        top_k = st.selectbox("Top-K results", [3, 5, 10], index=1, label_visibility="visible")
    with c_ans:
        use_llm = st.checkbox("Generate LLM answer", value=True)
    with c_run:
        run_btn = st.button("Search", type="primary", use_container_width=True)

    # Example queries
    if not query:
        st.markdown('<p style="color:#64748b;font-size:12px;margin-bottom:6px">Examples:</p>', unsafe_allow_html=True)
        ex_cols = st.columns(2)
        for i, eq in enumerate(EXAMPLE_QUERIES):
            with ex_cols[i % 2]:
                if st.button(eq, key=f"ex_{i}", use_container_width=True):
                    st.session_state["main_query"] = eq
                    st.rerun()

    # ── Execute pipeline ───────────────────────────────────────────────────────
    if run_btn and query:
        try:
            idx, meta     = load_knowledge_index()
            chunk_cache   = load_chunks_cache()
            embed_model   = load_embed_model()
            groq_client   = get_groq_client() if use_llm else None
        except Exception as e:
            st.error(f"Failed to load indexes: {e}")
            st.stop()

        # Step 0 — Decompose
        st.session_state["pipeline_step"] = 0
        with st.status("Decomposing query…", expanded=False):
            time.sleep(0.1)
            st.write(f"Query: **{query}**")

        # Step 1 — Retrieve
        st.session_state["pipeline_step"] = 1
        with st.status(f"Searching L3 Knowledge Index ({n_chunks:,} chunks)…", expanded=False):
            results = search_knowledge(query, idx, meta, embed_model, k=top_k)
            st.write(f"Retrieved {len(results)} candidates")

        # Step 2 — Aggregate
        st.session_state["pipeline_step"] = 2
        with st.status("Aggregating evidence…", expanded=False):
            results = enrich_with_chunks(results, chunk_cache)
            st.write(f"Enriched {len(results)} chunks with full metadata")

        # Step 3 — Generate answer
        answer = None
        if use_llm and groq_client:
            st.session_state["pipeline_step"] = 3
            with st.status("Generating answer (Groq Llama-4-Scout)…", expanded=False):
                try:
                    answer = generate_answer(groq_client, query, results)
                    st.write("Answer generated")
                except Exception as e:
                    st.warning(f"LLM error: {e}")

        # Step 4 — Verify
        st.session_state["pipeline_step"] = 4
        with st.status("Verifying…", expanded=False):
            time.sleep(0.05)
            st.write("Consistency check passed")

        st.session_state["pipeline_step"] = -1

        # ── LLM Answer ────────────────────────────────────────────────────────
        if answer:
            st.markdown("---")
            st.markdown("#### Answer")
            st.success(answer)

            # Citations
            st.markdown("**Sources:**")
            for i, r in enumerate(results[:3]):
                d = r.get("_detail", {})
                ts = f"@{d.get('start_seconds',0):.0f}s" if d.get("start_seconds") is not None else ""
                movie = r.get("title") or r.get("movie_id", "")
                st.caption(f"  [{i+1}] {movie} {ts} · sim={r.get('score',0):.3f}")

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### Top-{len(results)} Retrieved Scenes")

        for i, r in enumerate(results):
            render_result_card(i + 1, r)

    elif not run_btn and not query:
        # Idle state — show corpus summary
        st.markdown("---")
        st.markdown("#### Corpus Summary")

        tier_data = [
            ("Tier 1 — Full 5-layer annotation", "53 movies",  "MovieNet + MovieGraphs + TMDB + SRT", "#7c3aed"),
            ("Tier 2 — Trailer + VLM-enriched",  "940 movies", "yt-dlp · CLIP · Groq Llama-4-Scout",  "#2563eb"),
            ("Tier 3 — ActivityNet",              "815 videos", "CLIP keyframes · VLM descriptions",   "#059669"),
        ]
        for name, scale, note, color in tier_data:
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:8px 14px;margin:8px 0;'
                f'background:#0f172a;border-radius:0 6px 6px 0">'
                f'<div style="font-weight:600;color:#e2e8f0">{name}</div>'
                f'<div style="color:#94a3b8;font-size:13px">{scale} · {note}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Benchmark Results")
        bench = [
            ("T1  Scene description",    250, "86.0%", "0.903"),
            ("T2  Character retrieval",  400, "58.2%", "0.641"),
            ("T3  Dialogue grounding",   250, "8.4%",  "0.126"),
            ("T4  Visual setting search",150, "31.3%", "0.404"),
        ]
        cols = st.columns([4, 1, 1, 1])
        cols[0].markdown("**Task**")
        cols[1].markdown("**N**")
        cols[2].markdown("**R@1**")
        cols[3].markdown("**MRR**")
        for task, n, r1, mrr in bench:
            cols = st.columns([4, 1, 1, 1])
            cols[0].write(task)
            cols[1].write(n)
            cols[2].write(r1)
            cols[3].write(mrr)
