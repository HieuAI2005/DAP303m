import { useEffect, useState, useRef } from 'react'

// ── Config ────────────────────────────────────────────────────────────────────

const LAYER_COLORS = {
  L0: '#7c3aed', L1: '#2563eb', L2: '#0891b2',
  L3: '#059669', L4: '#d97706', L5: '#c65d2f',
}

const LAYERS = [
  { id: 'L0', name: 'Visual Frames',  icon: '🖼', desc: 'CLIP ViT-B/32 · 94,779 keyframes · 512-dim L2-norm' },
  { id: 'L1', name: 'Temporal',       icon: '⏱', desc: 'start / end seconds · shot_id · duration' },
  { id: 'L2', name: 'Semantic',       icon: '🧠', desc: 'VLM description · setting · actions · emotional tone' },
  { id: 'L3', name: 'Dialogue',       icon: '💬', desc: 'dialogue_text · speaker · audio events  [SRT / Whisper]' },
  { id: 'L4', name: 'Characters',     icon: '👤', desc: 'characters · emotions · cast_in_scene  [MovieGraphs]' },
  { id: 'L5', name: 'Narrative',      icon: '📖', desc: 'narrative_arc · causal_relations · screenplay_context' },
]

const INDEXES = [
  { label: 'L0  Frame Visual', spec: '94,779 × 512', note: 'FAISS FlatIP · CLIP ViT-B/32', color: LAYER_COLORS.L0 },
  { label: 'L1  Scene Fusion', spec: '12,900 × 896', note: '√0.72·CLIP + √0.28·text · normalized', color: LAYER_COLORS.L1 },
  { label: 'L3  Knowledge',    spec: '12,900 × 384', note: 'SentenceTransformer all-MiniLM-L6-v2', color: LAYER_COLORS.L3 },
  { label: 'L5  Neo4j Graph',  spec: '58 movies · 10,792+ nodes', note: 'Bolt :7688', color: LAYER_COLORS.L5 },
]

const EXAMPLES = [
  'Who is the main character in Forrest Gump?',
  'Find scenes where characters are eating together',
  'What happens when Don Corleone is shot in The Godfather?',
  'Emotional confrontation between two characters',
  'Outdoor action scene with running or chase',
  'Quiet intimate conversation between lovers',
]

const BENCHMARK = [
  { task: 'T1  Scene description',    n: 250, r1: '86.0%', r5: '96.0%', mrr: '0.903' },
  { task: 'T2  Character retrieval',  n: 400, r1: '58.2%', r5: '69.8%', mrr: '0.641' },
  { task: 'T3  Dialogue grounding',   n: 250, r1: '8.4%',  r5: '17.6%', mrr: '0.126' },
  { task: 'T4  Visual setting search',n: 150, r1: '31.3%', r5: '52.0%', mrr: '0.404' },
]

const PROCESS_STEPS = [
  { id: 'extract',  label: 'Keyframe Extraction',  icon: '🎞', desc: 'ffmpeg → 1 frame/3s · 224×224' },
  { id: 'clip',     label: 'CLIP Embedding',        icon: '🔭', desc: 'ViT-B/32 · 512-dim vectors' },
  { id: 'vlm',      label: 'VLM Annotation',        icon: '🧠', desc: 'Llama-4-Scout · L2 semantic metadata' },
  { id: 'srt',      label: 'SRT / Whisper STT',     icon: '💬', desc: 'dialogue alignment · L3 layer' },
  { id: 'index',    label: 'Index Update',           icon: '📦', desc: 'FAISS rebuild · knowledge + visual' },
]

const STEP_LABELS = {
  step_1_copy_video:              { label: 'Copy video',              layer: null },
  step_2_fetch_metadata:          { label: 'Fetch metadata',          layer: null },
  step_3_auto_annotate:           { label: 'Auto annotate',           layer: null },
  step_4_stt:                     { label: 'Speech-to-Text (STT)',    layer: 'L3' },
  step_4b_semantic_segmentation:  { label: 'Scene segmentation',      layer: 'L1' },
  step_4c_clip_video:             { label: 'CLIP video encoding',     layer: 'L0' },
  step_5_extract_keyframes:       { label: 'Extract keyframes',       layer: 'L0' },
  step_5b_cv_face_extraction:     { label: 'Face extraction',         layer: 'L4' },
  step_5c_identity_mapping:       { label: 'Character identity',      layer: 'L4' },
  step_6a_vlm_vision_extraction:  { label: 'VLM visual enrichment',   layer: 'L2' },
  step_6aa_script_scene_vlm_extraction: { label: 'Script VLM enrichment', layer: 'L2' },
  step_6b_fusion_graphing:        { label: 'Fusion graph build',      layer: 'L5' },
  step_6c_backfill_script_mapping:{ label: 'Script alignment',        layer: 'L3' },
  step_7_build_chunks:            { label: 'Build temporal chunks',   layer: 'L1' },
  step_7b_build_script_subscenes: { label: 'Build script sub-scenes', layer: 'L5' },
  step_8_index:                   { label: 'FAISS index update',      layer: null },
  step_8a_script_scene_index:     { label: 'Script scene index',      layer: null },
  step_8b_knowledge_graph:        { label: 'Knowledge graph build',   layer: 'L5' },
  step_8c_sync_graph:             { label: 'Neo4j graph sync',        layer: 'L5' },
}

const ARTIFACT_LAYERS = {
  keyframes:       { layer: 'L0', label: 'Keyframes',        icon: '🖼' },
  temporal_chunks: { layer: 'L1', label: 'Temporal chunks',  icon: '⏱' },
  annotation:      { layer: 'L2', label: 'VLM annotation',   icon: '🧠' },
  subtitle:        { layer: 'L3', label: 'Subtitle / SRT',   icon: '💬' },
  scene_graph:     { layer: 'L4', label: 'Scene graph',      icon: '👤' },
  script_subscenes:{ layer: 'L5', label: 'Script sub-scenes',icon: '📖' },
  raw_video:       { layer: null, label: 'Video',             icon: '🎬' },
  metadata:        { layer: null, label: 'Metadata',          icon: '📋' },
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function apiFetch(path, options) {
  const r = await fetch(path, options)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

function analyzeQuery(q) {
  const lower = q.toLowerCase()
  let intent = 'SCENE_SEARCH'
  if (/\bwho\b|\bcharacter\b|\bactor\b|\bcast\b/.test(lower))           intent = 'CHARACTER_RETRIEVAL'
  else if (/dialogue|says|said|quote|\bwhat.*say/.test(lower))           intent = 'DIALOGUE_GROUNDING'
  else if (/setting|location|\bwhere\b|outdoor|indoor/.test(lower))      intent = 'VISUAL_SETTING'
  else if (/\bwhat happens\b|\bwhat is\b|\bexplain\b|\bplot\b/.test(lower)) intent = 'NARRATIVE_QA'
  const STOP = new Set(['The','A','An','In','On','At','For','Find','What','Who','Where','When','Why','How','Is','Are','Was','Were','Does','Did','Can','Could','From'])
  const nouns = [...new Set(q.split(/\s+/).filter(w => /^[A-Z][a-z]{1,}/.test(w) && !STOP.has(w)))]
  return { intent, entities: nouns }
}

function Typewriter({ text, speed = 6 }) {
  const [shown, setShown] = useState('')
  useEffect(() => {
    setShown('')
    if (!text) return
    let i = 0
    const t = setInterval(() => { i++; setShown(text.slice(0, i)); if (i >= text.length) clearInterval(t) }, speed)
    return () => clearInterval(t)
  }, [text])
  return <span>{shown}<span className={shown.length < (text?.length || 0) ? 'cursor' : ''} /></span>
}

// ── System Overview Panel ─────────────────────────────────────────────────────

function ArchDiagram({ activeStep }) {
  const nodes = [
    { label: 'User Query',            cls: 'arch-query' },
    { label: 'Query Decomposition',   cls: 'arch-decompose',  step: 0 },
    { label: 'Multi-Level Retrieval', cls: 'arch-retrieval',  step: 1, sub: ['L3 Text', 'L1 Fusion', 'L0 Visual', 'L5 Graph'] },
    { label: 'Evidence Aggregation',  cls: 'arch-aggregate',  step: 4 },
    { label: 'Answer Generation',     cls: 'arch-generate',   step: 5, sub: ['Groq Llama-4-Scout'] },
    { label: 'Verifier / Judge',      cls: 'arch-verify',     step: 6 },
  ]
  return (
    <div className="arch-diagram">
      {nodes.map((node, i) => (
        <div key={node.cls}>
          {i > 0 && <div className="arch-arrow">↓</div>}
          <div className={`arch-node ${node.cls} ${node.step === activeStep ? 'arch-active' : activeStep > node.step ? 'arch-done' : ''}`}>
            <div className="arch-node-label">{node.label}</div>
            {node.sub && (
              <div className="arch-sub-row">
                {node.sub.map(s => <span key={s} className="arch-chip">{s}</span>)}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function SystemOverview({ activeStep }) {
  return (
    <aside className="overview-panel">
      <div className="ov-block">
        <div className="ov-title">System Architecture</div>
        <div className="ov-sub">Fig 1 — Agentic retrieval pipeline</div>
        <ArchDiagram activeStep={activeStep} />
      </div>
      <div className="ov-block">
        <div className="ov-title">Metadata Layers</div>
        <div className="ov-sub">5-layer scene schema (L0 – L5)</div>
        <div className="layer-schema">
          {LAYERS.map(l => (
            <div key={l.id} className="layer-band" style={{ '--lc': LAYER_COLORS[l.id] }}>
              <span className="layer-swatch" />
              <span className="layer-icon">{l.icon}</span>
              <div className="layer-text">
                <span className="layer-id">{l.id}</span>
                <span className="layer-name">{l.name}</span>
                <span className="layer-desc">{l.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="ov-block">
        <div className="ov-title">Retrieval Indexes</div>
        {INDEXES.map(idx => (
          <div key={idx.label} className="index-row" style={{ '--ic': idx.color }}>
            <div className="index-left"><span className="index-dot" />{idx.label}</div>
            <div className="index-right">
              <span className="index-spec">{idx.spec}</span>
              <span className="index-note">{idx.note}</span>
            </div>
          </div>
        ))}
      </div>
    </aside>
  )
}

// ── Pipeline StepCard ─────────────────────────────────────────────────────────

function StepCard({ step, status, data, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen || false)
  useEffect(() => { if (status === 'done') setOpen(true) }, [status])
  const icons = { idle: '○', running: '◌', done: '✓', skip: '—' }
  const stateClass = { idle: '', running: 'step-running', done: 'step-done', skip: 'step-skip' }
  return (
    <div className={`step-card ${stateClass[status] || ''}`} style={{ '--sc': step.color }}>
      <div className="step-header" onClick={() => status === 'done' && setOpen(o => !o)}>
        <span className={`step-icon-status ${status}`}>{icons[status]}</span>
        <div className="step-header-main">
          <span className="step-icon-big">{step.icon}</span>
          <div>
            <div className="step-title">{step.label}</div>
            <div className="step-subtitle">{step.desc}</div>
          </div>
        </div>
        {status === 'done' && data?.elapsed && <span className="step-elapsed">{data.elapsed}ms</span>}
        {status === 'running' && <span className="step-spinner" />}
        {status === 'done' && <span className="step-toggle">{open ? '▲' : '▼'}</span>}
      </div>
      {status === 'done' && open && data?.content && (
        <div className="step-content">{data.content}</div>
      )}
    </div>
  )
}

// ── Knowledge mini-card ───────────────────────────────────────────────────────

function KnowledgeCard({ result, rank }) {
  const m = result.metadata || {}
  const movie = m.title || result.movie_id || '—'
  const start = m.start_seconds
  const desc  = m.description || m.dialogue_text || ''
  const chars = m.characters
  return (
    <div className="k-card">
      <div className="k-card-top">
        <span className="k-rank">#{rank}</span>
        <span className="k-movie">{movie}</span>
        {start != null && <span className="k-ts">@{Math.round(start)}s</span>}
        <span className="k-score">sim={result.score?.toFixed(3) || '—'}</span>
      </div>
      {desc && <div className="k-desc">{desc.slice(0, 140)}{desc.length > 140 ? '…' : ''}</div>}
      {chars?.length > 0 && <div className="k-chars">{chars.slice(0, 4).join(', ')}</div>}
    </div>
  )
}

// ── Scene expandable card ─────────────────────────────────────────────────────

function SceneCard({ result, rank }) {
  const m = result.metadata || {}
  const movie  = m.title || result.movie_id || '—'
  const start  = m.start_seconds
  const end    = m.end_seconds
  const ts     = start != null ? `${Math.round(start)}s – ${Math.round(end)}s` : '—'
  const dlgRaw = m.dialogue_text || ''
  const hasDlg = dlgRaw && !['[NO_DIALOGUE]','[PENDING_SRT_ALIGNMENT]',''].includes(dlgRaw)
  const chars  = m.characters || []
  const cov    = { L0: true, L1: start != null, L2: !!m.description, L3: hasDlg, L4: chars.length > 0, L5: !!m.narrative_arc }
  return (
    <details className="scene-card" open={rank <= 2}>
      <summary className="scene-summary">
        <span className="sc-rank">#{rank}</span>
        <span className="sc-movie">{movie}</span>
        <span className="sc-ts">{ts}</span>
        <span className="sc-score">sim={result.score?.toFixed(3) || '—'}</span>
      </summary>
      <div className="scene-body">
        <div className="sb-grid">
          <div>
            <div className="sb-label">L2  Description</div>
            <p className="sb-val">{m.description || '—'}</p>
            <div className="sb-label">Setting · Tone</div>
            <p className="sb-val">{[m.vision_setting, m.emotional_tone].filter(Boolean).join(' · ') || '—'}</p>
            {m.vision_actions?.length > 0 && <>
              <div className="sb-label">Actions</div>
              <p className="sb-val">{m.vision_actions.join(', ')}</p>
            </>}
          </div>
          <div>
            <div className="sb-label">L3  Dialogue</div>
            <p className={`sb-val ${hasDlg ? 'sb-dlg' : ''}`}>
              {hasDlg ? `"${dlgRaw.slice(0, 300)}${dlgRaw.length > 300 ? '…' : ''}"` : '—'}
            </p>
          </div>
        </div>
        <div className="sb-grid">
          <div>
            <div className="sb-label">L4  Characters</div>
            <p className="sb-val">{chars.slice(0, 6).join(', ') || '—'}</p>
            {(m.cast_in_scene || []).slice(0, 2).map(cs =>
              cs.actor ? <div key={cs.character} className="sb-cast">{cs.character} → {cs.actor}</div> : null
            )}
          </div>
          <div>
            <div className="sb-label">L5  Narrative Arc</div>
            <p className="sb-val">{(m.narrative_arc || '—').slice(0, 120)}</p>
            {(m.causal_relations || []).slice(0, 2).map((r, i) => (
              <div key={i} className="sb-causal">↪ {typeof r === 'object' ? r.relation : r}</div>
            ))}
          </div>
        </div>
        <div className="layer-badges">
          {Object.entries(cov).map(([l, ok]) => (
            <span key={l} className={`lb ${ok ? 'lb-ok' : 'lb-miss'}`}
                  style={ok ? { '--lbg': LAYER_COLORS[l] + '22', '--lbc': LAYER_COLORS[l] } : {}}>
              {ok ? '✓' : '·'} {l}
            </span>
          ))}
        </div>
      </div>
    </details>
  )
}

// ── STEPS config ──────────────────────────────────────────────────────────────

const STEPS = [
  { id: 'decompose',   label: 'Query Decomposition',    icon: '⚙',  desc: 'Parse intent · extract entities',      color: '#7c3aed' },
  { id: 'retrieve_l3', label: 'L3 Knowledge Retrieval', icon: '📚', desc: 'SentenceTransformer · 12,900 chunks',   color: '#2563eb' },
  { id: 'retrieve_l0', label: 'L0 Visual Retrieval',    icon: '🖼', desc: 'CLIP ViT-B/32 · 94,779 keyframes',     color: '#0891b2' },
  { id: 'retrieve_l5', label: 'L5 Script / Graph',      icon: '📖', desc: 'Neo4j narrative graph · 58 movies',    color: '#059669' },
  { id: 'aggregate',   label: 'Evidence Aggregation',   icon: '📦', desc: 'Rank · deduplicate · enrich metadata', color: '#d97706' },
  { id: 'generate',    label: 'Answer Generation',      icon: '🤖', desc: 'Groq Llama-4-Scout · grounded answer', color: '#c65d2f' },
]

// ── Query Page ────────────────────────────────────────────────────────────────

function QueryPage() {
  const [query,     setQuery]     = useState('')
  const [topK,      setTopK]      = useState(5)
  const [movieId,   setMovieId]   = useState('')
  const [loading,   setLoading]   = useState(false)
  const [activeIdx, setActiveIdx] = useState(-1)
  const [stepData,  setStepData]  = useState({})
  const [apiResult, setApiResult] = useState(null)
  const [error,     setError]     = useState(null)
  const startTimes = useRef({})

  function resetState() { setStepData({}); setApiResult(null); setError(null); setActiveIdx(-1); startTimes.current = {} }

  function markStep(id, status, content = null) {
    const elapsed = startTimes.current[id] ? Date.now() - startTimes.current[id] : null
    setStepData(prev => ({ ...prev, [id]: { status, data: { content, elapsed } } }))
  }

  function startStep(id) {
    startTimes.current[id] = Date.now()
    setActiveIdx(STEPS.findIndex(s => s.id === id))
    setStepData(prev => ({ ...prev, [id]: { status: 'running', data: {} } }))
  }

  async function runQuery(q) {
    if (!q.trim() || loading) return
    resetState()
    setLoading(true)
    try {
      startStep('decompose')
      const { intent, entities } = analyzeQuery(q)
      await new Promise(r => setTimeout(r, 120))
      markStep('decompose', 'done',
        <div className="step-data">
          <div className="sd-row"><span className="sd-key">Intent</span><span className="sd-val intent-chip">{intent.replace(/_/g,' ')}</span></div>
          {entities.length > 0 && (
            <div className="sd-row">
              <span className="sd-key">Entities</span>
              <span className="sd-val">{entities.map(e => <span key={e} className="entity-chip">{e}</span>)}</span>
            </div>
          )}
          <div className="sd-row"><span className="sd-key">Query</span><span className="sd-val sd-query">{q}</span></div>
        </div>
      )

      startStep('retrieve_l3'); startStep('retrieve_l0')

      const form = new FormData()
      form.append('query', q)
      form.append('history_json', '[]')
      if (movieId) form.append('movie_id', movieId)

      const t0     = Date.now()
      const result = await apiFetch('/api/qa', { method: 'POST', body: form })
      setApiResult(result)

      const status  = result.status || {}
      const kDocs   = result.knowledge_results || []
      const vFrames = result.visual_results    || []
      const sDocs   = result.script_results    || []

      startTimes.current['retrieve_l3'] = t0
      markStep('retrieve_l3', kDocs.length > 0 ? 'done' : 'skip',
        kDocs.length > 0 ? (
          <div className="step-data">
            <div className="sd-summary">{status.knowledge_docs || kDocs.length} chunks retrieved</div>
            <div className="k-list">{kDocs.slice(0, 3).map((r, i) => <KnowledgeCard key={i} result={r} rank={i + 1} />)}</div>
          </div>
        ) : <div className="sd-empty">No knowledge results</div>
      )

      await new Promise(r => setTimeout(r, 80))
      startTimes.current['retrieve_l0'] = t0
      markStep('retrieve_l0', vFrames.length > 0 ? 'done' : 'skip',
        vFrames.length > 0 ? (
          <div className="step-data">
            <div className="sd-summary">{status.visual_frames || vFrames.length} keyframes retrieved</div>
            <div className="visual-frames-grid">
              {vFrames.slice(0, 6).map((r, i) => (
                r.url ? (
                  <div key={i} className="visual-frame-card">
                    <a href={r.url} target="_blank" rel="noreferrer">
                      <img className="visual-frame-img" src={r.url} alt={`frame-${i+1}`} loading="lazy" />
                    </a>
                    <div className="visual-frame-meta">
                      <span>{r.movie_id || '—'}</span>
                      <span>sim={r.score?.toFixed(3) || '—'}</span>
                    </div>
                  </div>
                ) : (
                  <div key={i} className="visual-chip">
                    <span>{r.movie_id || '—'}</span>
                    <span>sim={r.score?.toFixed(3) || '—'}</span>
                  </div>
                )
              ))}
            </div>
          </div>
        ) : <div className="sd-empty">Visual retrieval skipped</div>
      )

      await new Promise(r => setTimeout(r, 80))
      startTimes.current['retrieve_l5'] = t0
      markStep('retrieve_l5', sDocs.length > 0 ? 'done' : 'skip',
        sDocs.length > 0 ? (
          <div className="step-data"><div className="sd-summary">{status.script_scenes || sDocs.length} script scenes</div></div>
        ) : <div className="sd-empty">Script / graph retrieval skipped</div>
      )

      await new Promise(r => setTimeout(r, 80))
      startStep('aggregate')
      const total = kDocs.length + vFrames.length + sDocs.length
      const tg    = result.temporal_grounding
      await new Promise(r => setTimeout(r, 100))
      markStep('aggregate', 'done',
        <div className="step-data">
          <div className="sd-summary">{total} evidence items combined</div>
          <div className="sd-row"><span className="sd-key">Knowledge</span><span className="sd-val">{kDocs.length} docs</span></div>
          <div className="sd-row"><span className="sd-key">Visual</span><span className="sd-val">{vFrames.length} frames</span></div>
          <div className="sd-row"><span className="sd-key">Script</span><span className="sd-val">{sDocs.length} scenes</span></div>
          {tg && <div className="sd-row"><span className="sd-key">Temporal</span><span className="sd-val">{tg.start_time || '—'} → {tg.end_time || '—'}</span></div>}
          {result.thoughts?.length > 0 && (
            <div className="thoughts-list">
              {result.thoughts.map((th, i) => <div key={i} className="thought-item">💭 {th}</div>)}
            </div>
          )}
        </div>
      )

      await new Promise(r => setTimeout(r, 80))
      startStep('generate')
      await new Promise(r => setTimeout(r, 120))
      markStep('generate', 'done',
        <div className="step-data">
          <div className="sd-row"><span className="sd-key">Model</span><span className="sd-val">Groq Llama-4-Scout</span></div>
          <div className="sd-row"><span className="sd-key">Sources</span><span className="sd-val">{kDocs.length} cited</span></div>
          <div className="sd-row"><span className="sd-key">Intent</span><span className="sd-val intent-chip">{(result.intent || intent).replace(/_/g,' ')}</span></div>
        </div>
      )

      setActiveIdx(-1)
    } catch (e) {
      setError(e.message)
      setActiveIdx(-1)
    } finally {
      setLoading(false)
    }
  }

  const getStatus = id => stepData[id]?.status || 'idle'
  const getData   = id => stepData[id]?.data   || null
  const scenes    = apiResult?.knowledge_results || []
  const answer    = apiResult?.answer || ''

  return (
    <div className="query-layout">
      <div className="query-main">
        <MovieSelector value={movieId} onChange={setMovieId} />
        <form className="query-bar" onSubmit={e => { e.preventDefault(); runQuery(query) }}>
          <input className="query-input" value={query} onChange={e => setQuery(e.target.value)}
            placeholder="Ask anything about a movie scene…" autoFocus />
          <div className="query-controls">
            <select className="topk-sel" value={topK} onChange={e => setTopK(+e.target.value)}>
              {[3, 5, 10].map(k => <option key={k} value={k}>Top {k}</option>)}
            </select>
            <button className="search-btn" type="submit" disabled={loading}>
              {loading ? <span className="btn-spinner" /> : null}
              {loading ? 'Running…' : 'Search'}
            </button>
          </div>
        </form>

        {!loading && !apiResult && !error && (
          <>
            <div className="examples-wrap">
              <p className="section-label">Example queries</p>
              <div className="examples-grid">
                {EXAMPLES.map(eq => (
                  <button key={eq} className="ex-btn" onClick={() => { setQuery(eq); runQuery(eq) }}>{eq}</button>
                ))}
              </div>
            </div>
            <div className="idle-stats">
              <div className="corpus-grid">
                {[
                  { n: '993',    l: 'movies indexed',     sub: '53 full · 940 trailer' },
                  { n: '12,900', l: 'knowledge chunks',   sub: '5-layer schema' },
                  { n: '94,779', l: 'visual keyframes',   sub: 'CLIP ViT-B/32' },
                  { n: '815',    l: 'ActivityNet videos', sub: 'Tier 3 corpus' },
                ].map(c => (
                  <div key={c.n} className="corpus-card">
                    <div className="corpus-n">{c.n}</div>
                    <div className="corpus-l">{c.l}</div>
                    <div className="corpus-s">{c.sub}</div>
                  </div>
                ))}
              </div>
              <div className="bench-wrap">
                <p className="section-label">QA Benchmark (12,900-chunk index)</p>
                <table className="bench-table">
                  <thead><tr><th>Task</th><th>N</th><th>R@1</th><th>R@5</th><th>MRR</th></tr></thead>
                  <tbody>
                    {BENCHMARK.map(b => (
                      <tr key={b.task}>
                        <td>{b.task}</td><td>{b.n}</td>
                        <td><strong>{b.r1}</strong></td><td>{b.r5}</td><td>{b.mrr}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {error && <div className="error-box">⚠ {error}</div>}

        {(loading || apiResult) && (
          <>
            {answer && (
              <div className="answer-block">
                <div className="answer-label"><span className="answer-dot" />Answer</div>
                <div className="answer-text"><Typewriter text={answer} speed={6} /></div>
                {scenes.length > 0 && (
                  <div className="answer-sources">
                    {scenes.slice(0, 3).map((r, i) => {
                      const m = r.metadata || {}
                      const ts = m.start_seconds != null ? `@${Math.round(m.start_seconds)}s` : ''
                      return <span key={i} className="source-chip">[{i+1}] {m.title || r.movie_id} {ts} sim={r.score?.toFixed(3)}</span>
                    })}
                  </div>
                )}
              </div>
            )}
            <div className="pipeline-exec">
              <p className="section-label">Agentic Pipeline — step-by-step execution</p>
              {STEPS.map(step => (
                <StepCard key={step.id} step={step} status={getStatus(step.id)} data={getData(step.id)} />
              ))}
            </div>
            {scenes.length > 0 && (
              <div className="results-wrap">
                <p className="section-label">Retrieved Scenes — L0–L5 Metadata</p>
                {scenes.slice(0, topK).map((r, i) => <SceneCard key={i} result={r} rank={i + 1} />)}
              </div>
            )}
          </>
        )}
      </div>
      <SystemOverview activeStep={activeIdx} />
    </div>
  )
}

// ── Movie Selector ────────────────────────────────────────────────────────────

function MovieSelector({ value, onChange }) {
  const [movies, setMovies] = useState([])
  useEffect(() => {
    fetch('/api/library').then(r => r.ok ? r.json() : null).then(d => {
      if (d?.rows) setMovies(d.rows.filter(r => r.chunks > 0))
    }).catch(() => {})
  }, [])
  return (
    <div className="movie-selector-row">
      <span className="movie-selector-label">Movie filter:</span>
      <select className="movie-selector-select" value={value} onChange={e => onChange(e.target.value)}>
        <option value="">All movies</option>
        {movies.map(r => (
          <option key={r.movie_id} value={r.movie_id}>
            {r.title && r.title !== r.movie_id ? `${r.title} (${r.movie_id})` : r.movie_id}
          </option>
        ))}
      </select>
      {value && (
        <button className="clear-btn" type="button" onClick={() => onChange('')}>✕ Clear</button>
      )}
    </div>
  )
}

// ── Chat Page ─────────────────────────────────────────────────────────────────

function ChatPage({ initialMovieId, onMovieIdChange }) {
  const [messages,  setMessages]  = useState([])
  const [input,     setInput]     = useState('')
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)
  const [movieId,   setMovieId]   = useState(initialMovieId || '')
  const bottomRef = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  // Sync external initialMovieId when it changes (e.g. after process)
  useEffect(() => { if (initialMovieId) setMovieId(initialMovieId) }, [initialMovieId])

  function handleMovieChange(id) {
    setMovieId(id)
    onMovieIdChange?.(id)
  }

  async function sendMessage(e, overrideText) {
    e?.preventDefault()
    const text = (overrideText || input).trim()
    if (!text || loading) return
    setInput('')
    setError(null)

    const userMsg   = { role: 'user', content: text }
    const thinking  = { role: 'assistant', content: null, loading: true }
    setMessages(prev => [...prev, userMsg, thinking])
    setLoading(true)

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content || '' }))
      const form = new FormData()
      form.append('message', text)
      form.append('history_json', JSON.stringify(history))
      if (movieId) form.append('movie_id', movieId)

      const result = await apiFetch('/api/chat', { method: 'POST', body: form })
      const answer  = result.answer || '(no answer)'
      const sources = result.knowledge_results || []
      const gallery = result.gallery_items || []

      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: 'assistant', content: answer, sources, gallery, intent: result.intent, thoughts: result.thoughts }
      ])
    } catch (e) {
      setMessages(prev => prev.slice(0, -1))
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const CHAT_STARTERS = [
    'Tell me about the opening scene of The Godfather',
    'Which movies have car chase scenes?',
    'Who plays Forrest Gump and what is his personality?',
    'Describe an emotional scene between two characters',
  ]

  return (
    <div className="chat-layout">
      <div className="chat-main">
        <MovieSelector value={movieId} onChange={handleMovieChange} />
        {messages.length === 0 ? (
          <div className="chat-idle">
            <div className="chat-idle-title">VideoSceneRAG Chat</div>
            <div className="chat-idle-sub">
              {movieId
                ? `Chatting about: ${movieId} — ask anything about this movie.`
                : 'Ask the LLM anything about the video corpus — it answers using retrieved scene metadata.'}
            </div>
            <div className="chat-starters">
              {CHAT_STARTERS.map(s => (
                <button key={s} className="ex-btn" onClick={() => { setInput(s); sendMessage({ preventDefault: () => {} }, s) }}>{s}</button>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-msg chat-msg-${msg.role}`}>
                <div className="chat-bubble">
                  {msg.loading ? (
                    <span className="chat-thinking">
                      <span className="chat-dot" /><span className="chat-dot" /><span className="chat-dot" />
                    </span>
                  ) : (
                    <>
                      <div className="chat-text">
                        {msg.role === 'assistant'
                          ? <Typewriter text={msg.content} speed={5} />
                          : msg.content}
                      </div>
                      {msg.sources?.length > 0 && (
                        <div className="chat-sources">
                          {msg.sources.slice(0, 3).map((r, j) => {
                            const m = r.metadata || {}
                            return (
                              <span key={j} className="source-chip">
                                [{j+1}] {m.title || r.movie_id}
                                {m.start_seconds != null ? ` @${Math.round(m.start_seconds)}s` : ''}
                              </span>
                            )
                          })}
                        </div>
                      )}
                      {msg.gallery?.length > 0 && (
                        <div className="chat-gallery">
                          {msg.gallery.map((img, j) => (
                            <a key={j} href={img.url} target="_blank" rel="noreferrer">
                              <img className="chat-gallery-img" src={img.url} alt={`frame-${j+1}`} loading="lazy" />
                            </a>
                          ))}
                        </div>
                      )}
                      {msg.intent && (
                        <div className="chat-meta">
                          <span className="intent-chip">{msg.intent.replace(/_/g,' ')}</span>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        )}

        {error && <div className="error-box">⚠ {error}</div>}

        <form className="chat-input-bar" onSubmit={sendMessage}>
          <input
            className="query-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={movieId ? `Ask about ${movieId}…` : 'Ask about a movie scene, character, dialogue…'}
            disabled={loading}
          />
          <button className="search-btn" type="submit" disabled={loading || !input.trim()}>
            {loading ? <span className="btn-spinner" /> : null}
            {loading ? '' : 'Send'}
          </button>
          {messages.length > 0 && (
            <button type="button" className="clear-btn" onClick={() => setMessages([])}>Clear</button>
          )}
        </form>
      </div>

      <SystemOverview activeStep={-1} />
    </div>
  )
}

// ── Process Page ──────────────────────────────────────────────────────────────

function ProcessPage({ onChatAbout }) {
  const [movieId,  setMovieId]  = useState('')
  const [video,    setVideo]    = useState(null)
  const [sub,      setSub]      = useState(null)
  const [force,    setForce]    = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [result,   setResult]   = useState(null)
  const [stepIdx,  setStepIdx]  = useState(-1)
  const [error,    setError]    = useState(null)

  async function submit(e) {
    e.preventDefault()
    if (!movieId || !video) { setError('movie_id and video file are required'); return }
    setError(null); setResult(null); setLoading(true)

    // Animate steps while uploading / processing
    let s = 0
    setStepIdx(0)
    const advance = setInterval(() => {
      s++
      if (s < PROCESS_STEPS.length) setStepIdx(s)
    }, 3000)

    const form = new FormData()
    form.append('movie_id', movieId)
    form.append('video', video)
    form.append('force', String(force))
    if (sub) form.append('subtitle', sub)

    try {
      const data = await apiFetch('/api/ingest', { method: 'POST', body: form })
      clearInterval(advance)
      setStepIdx(PROCESS_STEPS.length)  // all done
      setResult(data)
    } catch (e) {
      clearInterval(advance)
      setStepIdx(-1)
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const allDone = stepIdx >= PROCESS_STEPS.length

  return (
    <div className="tab-page">
      <div className="grid two-col">
        <div className="panel">
          <h2 className="panel-title">Process New Video</h2>
          <p className="muted" style={{ marginBottom: 16, fontSize: 13 }}>
            Upload a video to run the full pipeline: keyframe extraction → CLIP embedding →
            VLM annotation → SRT alignment → index update.
          </p>
          <form className="form-stack" onSubmit={submit}>
            <label><span>Movie / Video ID</span>
              <input value={movieId} onChange={e => setMovieId(e.target.value)} placeholder="tt0120338 or my_video" />
            </label>
            <label><span>Video file (.mp4 / .mkv)</span>
              <input type="file" accept="video/*" onChange={e => setVideo(e.target.files?.[0] ?? null)} />
            </label>
            <label><span>Subtitle (.srt) — optional</span>
              <input type="file" accept=".srt" onChange={e => setSub(e.target.files?.[0] ?? null)} />
            </label>
            <label className="checkbox-row">
              <input type="checkbox" checked={force} onChange={e => setForce(e.target.checked)} />
              <span>Force rebuild (overwrite existing)</span>
            </label>
            <button className="search-btn" type="submit" disabled={loading}>
              {loading ? <span className="btn-spinner" /> : null}
              {loading ? 'Processing…' : 'Run pipeline'}
            </button>
            {error && <div className="error-box">⚠ {error}</div>}
          </form>
        </div>

        <div className="panel">
          <h2 className="panel-title">Pipeline Status</h2>
          <div className="process-steps">
            {PROCESS_STEPS.map((s, i) => {
              const status = loading
                ? (i < stepIdx ? 'done' : i === stepIdx ? 'running' : 'idle')
                : (allDone ? 'done' : 'idle')
              return (
                <StepCard key={s.id} step={{ ...s, color: Object.values(LAYER_COLORS)[i] }}
                          status={status} data={null} />
              )
            })}
          </div>

          {result && (
            <div style={{ marginTop: 16 }}>
              <p className="section-label">Result</p>

              {/* Status header */}
              <div className={result.success ? 'answer-block' : 'error-box'} style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
                  {result.success ? '✓ Pipeline completed' : '✗ Pipeline failed'}
                  {result.movie_id && <code style={{ marginLeft: 10, fontWeight: 400, fontSize: 13, background: 'rgba(0,0,0,0.06)', padding: '2px 6px', borderRadius: 4 }}>{result.movie_id}</code>}
                </div>
                {result.last_error && (
                  <div style={{ fontSize: 13, color: '#c65d2f', marginTop: 4 }}>
                    <strong>Error:</strong> {result.last_error}
                  </div>
                )}
                {result.failed_step && (
                  <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 2 }}>
                    Failed at: <code>{STEP_LABELS[result.failed_step]?.label || result.failed_step}</code>
                  </div>
                )}
                {result.reload_messages?.length > 0 && (
                  <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>
                    Index: {result.reload_messages.join(' · ')}
                  </div>
                )}
              </div>

              {/* L-layer artifact grid */}
              {result.storage?.artifacts?.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                  <p className="section-label" style={{ marginBottom: 6 }}>Output — L-Layer Artifacts</p>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6 }}>
                    {result.storage.artifacts.map(a => {
                      const meta = ARTIFACT_LAYERS[a.key] || { layer: null, label: a.label, icon: '📁' }
                      const layerColor = meta.layer ? LAYER_COLORS[meta.layer] : 'var(--muted)'
                      return (
                        <div key={a.key} style={{
                          display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px',
                          background: a.exists ? 'rgba(16,185,129,0.07)' : 'rgba(0,0,0,0.03)',
                          border: `1px solid ${a.exists ? 'rgba(16,185,129,0.2)' : 'var(--line)'}`,
                          borderRadius: 8, fontSize: 12,
                        }}>
                          <span style={{ fontSize: 15 }}>{a.exists ? meta.icon : '○'}</span>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontWeight: 600, color: a.exists ? 'var(--ink)' : 'var(--muted)' }}>
                              {meta.label}
                              {a.count != null && <span style={{ fontWeight: 400, color: 'var(--muted)', marginLeft: 4 }}>({a.count})</span>}
                            </div>
                          </div>
                          {meta.layer && (
                            <span style={{ fontSize: 10, fontWeight: 700, color: layerColor, border: `1px solid ${layerColor}`, borderRadius: 4, padding: '1px 5px' }}>
                              {meta.layer}
                            </span>
                          )}
                          <span style={{ color: a.exists ? '#10b981' : 'var(--muted)', fontWeight: 700 }}>
                            {a.exists ? '✓' : '—'}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Completed steps as layer badges */}
              {result.completed_steps?.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                  <p className="section-label" style={{ marginBottom: 6 }}>Pipeline Steps ({result.completed_steps.length})</p>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {result.completed_steps.map(s => {
                      const info = STEP_LABELS[s] || { label: s, layer: null }
                      const color = info.layer ? LAYER_COLORS[info.layer] : 'var(--muted)'
                      return (
                        <span key={s} style={{
                          fontSize: 11, padding: '3px 8px', borderRadius: 999,
                          background: info.layer ? `${color}18` : 'var(--bg)',
                          border: `1px solid ${info.layer ? color + '44' : 'var(--line)'}`,
                          color: info.layer ? color : 'var(--muted)',
                        }}>
                          {info.layer && <span style={{ fontWeight: 700, marginRight: 3 }}>{info.layer}</span>}
                          {info.label}
                        </span>
                      )
                    })}
                  </div>
                </div>
              )}

              {result.success && onChatAbout && (
                <button
                  type="button"
                  className="search-btn"
                  style={{ marginTop: 6, width: '100%' }}
                  onClick={() => onChatAbout(result.movie_id)}
                >
                  💬 Chat about this movie
                </button>
              )}
              <details style={{ marginTop: 10 }}>
                <summary style={{ fontSize: 12, cursor: 'pointer', color: 'var(--muted)' }}>Raw JSON</summary>
                <pre className="json-block">{JSON.stringify(result, null, 2)}</pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Library Page ──────────────────────────────────────────────────────────────

function LibraryPage() {
  const [lib, setLib] = useState({ rows: [], summary: {} })
  useEffect(() => {
    fetch('/api/library').then(r => r.ok ? r.json() : null).then(d => d && setLib(d))
  }, [])
  const rows = lib.rows    || []
  const sum  = lib.summary || {}
  return (
    <div className="tab-page">
      <div className="lib-stats">
        {[['Movies', sum.movie_count ?? rows.length], ['Indexes', sum.index_file_count ?? '—'],
          ['Knowledge', sum.knowledge_ready ? 'Ready' : '—'], ['Visual', sum.visual_ready ? 'Ready' : '—']
        ].map(([l, v]) => (
          <div key={l} className="lib-stat"><span className="ls-n">{v}</span><span className="ls-l">{l}</span></div>
        ))}
      </div>
      <div className="panel">
        <div className="table-wrap">
          <table>
            <thead><tr><th>movie_id</th><th>artifacts</th><th>video</th><th>subtitle</th><th>chunks</th><th>scene graph</th><th>script</th></tr></thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.movie_id}>
                  <td><code>{r.movie_id}</code></td><td>{r.artifact_ratio}</td>
                  <td>{r.video ? '✓' : '—'}</td><td>{r.subtitle ? '✓' : '—'}</td>
                  <td>{r.chunks}</td><td>{r.scene_graph ? '✓' : '—'}</td><td>{r.script_subscenes ? '✓' : '—'}</td>
                </tr>
              ))}
              {!rows.length && <tr><td colSpan={7} className="muted">No data</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Root ──────────────────────────────────────────────────────────────────────

const TABS = [
  { id: 'query',   label: 'Query'   },
  { id: 'chat',    label: 'Chat'    },
  { id: 'process', label: 'Process' },
  { id: 'library', label: 'Library' },
]

export default function App() {
  const [tab,         setTab]         = useState('query')
  const [chatMovieId, setChatMovieId] = useState('')

  function handleChatAbout(movieId) {
    setChatMovieId(movieId)
    setTab('chat')
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <span className="brand">🎬 VideoSceneRAG</span>
        <div className="topbar-stats">
          {[['12,900','chunks'],['94,779','keyframes'],['993','movies'],['815','ActivityNet']].map(([v,l]) => (
            <span key={l} className="topbar-chip"><b>{v}</b> {l}</span>
          ))}
        </div>
        <nav className="tabbar">
          {TABS.map(t => (
            <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {tab === 'query'   && <QueryPage />}
        {tab === 'chat'    && <ChatPage initialMovieId={chatMovieId} onMovieIdChange={setChatMovieId} />}
        {tab === 'process' && <ProcessPage onChatAbout={handleChatAbout} />}
        {tab === 'library' && <LibraryPage />}
      </main>
    </div>
  )
}
