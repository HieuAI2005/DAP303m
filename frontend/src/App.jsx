import { useEffect, useState } from 'react'

const tabList = [
  { id: 'system', label: 'System Story' },
  { id: 'pipeline', label: 'Data Pipeline' },
  { id: 'flows', label: 'Runtime Flows' },
  { id: 'workspace', label: 'Live Workspace' },
]

const datasetCards = [
  {
    id: 'videorag',
    eyebrow: 'Primary corpus',
    title: 'VideoRag 41-movie scene store',
    stats: ['6,077 chunks', '41 movies', '5-layer retrieval-ready'],
    body:
      'Merged corpus built from the 22 original MovieNet-style temporal chunks plus 19 Tier-2 movies converted from unified_dataset and aligned with SRT.',
    highlights: [
      'L3 dialogue coverage ~90%',
      'L4 characters ~92%',
      'knowledge_videorag.faiss is the main scene-text index',
    ],
  },
  {
    id: 'activitynet',
    eyebrow: 'Secondary corpus',
    title: 'ActivityNet event descriptions',
    stats: ['23,064 chunks', 'caption-driven', 'broad domain'],
    body:
      'Useful for scaling text retrieval and testing non-movie video coverage. Dialogue is currently caption proxy text, not true speech transcripts.',
    highlights: [
      'Knowledge index already built',
      'Characters unavailable',
      'Whisper enrichment remains pending',
    ],
  },
  {
    id: 'youcook2',
    eyebrow: 'Benchmark corpus',
    title: 'YouCook2 procedural benchmark',
    stats: ['3,179 chunks', '414 videos', 'temporal benchmark'],
    body:
      'Used to stress-test text grounding on instructional video. Strong for benchmarking retrieval behavior, weaker as a narrative scene corpus.',
    highlights: [
      'knowledge_youcook2.faiss is available',
      'No character layer',
      'Useful for cross-domain evaluation',
    ],
  },
]

const layerCards = [
  {
    id: 'l1',
    title: 'L1 Temporal Anchor',
    tag: 'Where in the video',
    body: 'Chunk identity, movie id, timestamps, shot ranges, and duration let the system ground every result back to a segment.',
    fields: ['chunk_id', 'movie_id', 'start_seconds', 'end_seconds', 'duration'],
  },
  {
    id: 'l2',
    title: 'L2 Semantic Scene',
    tag: 'What is happening',
    body: 'Descriptions, situation labels, visual setting, actions, and tone carry the scene-level meaning used by the text index.',
    fields: ['description', 'situation', 'vision_setting', 'vision_actions', 'emotional_tone'],
  },
  {
    id: 'l3',
    title: 'L3 Dialogue and Audio',
    tag: 'What is being said',
    body: 'Dialogue comes from aligned SRT or Whisper-style transcription. This layer supports quote lookup, speech evidence, and scene disambiguation.',
    fields: ['dialogue_text', 'speaker', 'audio_events', 'background_music'],
  },
  {
    id: 'l4',
    title: 'L4 Cast and Characters',
    tag: 'Who is present',
    body: 'Character names and scene-level cast cues make movie and scene retrieval much more discriminative than pure generic descriptions.',
    fields: ['characters', 'cast_in_scene', 'character_emotions'],
  },
  {
    id: 'l5',
    title: 'L5 Script and Narrative',
    tag: 'Why the scene matters',
    body: 'Narrative arc, causal relations, screenplay context, and script headings turn raw segments into reasoning-ready evidence.',
    fields: ['narrative_arc', 'causal_relations', 'screenplay_context', 'script_primary_heading'],
  },
]

const pipelineStages = [
  {
    id: 'source',
    step: '01',
    title: 'Source acquisition',
    summary: 'Bring raw or semi-structured assets into one workspace.',
    bullets: [
      'MovieNet temporal chunks and scene annotations',
      'unified_dataset clips with interactions and character metadata',
      'SRT subtitles, TMDB metadata, and optional local videos',
    ],
  },
  {
    id: 'convert',
    step: '02',
    title: 'Chunk normalization',
    summary: 'Convert heterogeneous records into the common 5-layer chunk schema.',
    bullets: [
      'Map shot ranges to seconds',
      'Build Tier-2 chunks from unified_dataset',
      'Infer placeholder narrative and emotion fields where needed',
    ],
  },
  {
    id: 'align',
    step: '03',
    title: 'Dialogue alignment',
    summary: 'Fill L3 with subtitle or transcription evidence.',
    bullets: [
      'Overlap SRT windows with chunk timestamps',
      'Use fallback windows when direct overlap is weak',
      'Preserve dialogue text as searchable evidence',
    ],
  },
  {
    id: 'merge',
    step: '04',
    title: 'Merge and coverage audit',
    summary: 'Deduplicate chunks and compute readiness of each layer.',
    bullets: [
      'Merge original 22-movie set with Tier-2 additions',
      'Track L3, L4, and L5 completeness',
      'Persist a single all_chunks.json corpus for retrieval',
    ],
  },
  {
    id: 'index',
    step: '05',
    title: 'Embedding and indexing',
    summary: 'Encode retrieval text and persist FAISS indexes.',
    bullets: [
      'SentenceTransformer all-MiniLM-L6-v2',
      'IndexFlatIP on normalized vectors for cosine similarity',
      'Separate knowledge indexes for VideoRag, ActivityNet, and YouCook2',
    ],
  },
  {
    id: 'runtime',
    step: '06',
    title: 'Runtime evidence loop',
    summary: 'Query, retrieve, inspect evidence, and answer.',
    bullets: [
      'FastAPI runtime exposes overview, library, ingest, and QA endpoints',
      'Knowledge and visual retrieval can both feed the answer path',
      'Evidence is returned with scene metadata, timestamps, and media paths',
    ],
  },
]

const runtimeFlows = [
  {
    id: 'search',
    title: 'Scene retrieval flow',
    accent: 'amber',
    steps: [
      'User query enters the runtime service.',
      'Text query hits the knowledge FAISS index built from 5-layer chunks.',
      'Top scenes return timestamps, characters, narrative cues, and dialogue evidence.',
    ],
  },
  {
    id: 'multimodal',
    title: 'Media-assisted flow',
    accent: 'teal',
    steps: [
      'Image or video upload is saved by the API layer.',
      'Runtime can combine visual search, script-scene lookup, and knowledge retrieval.',
      'The UI should show how one query fans out into multiple evidence channels.',
    ],
  },
  {
    id: 'answer',
    title: 'Answer generation flow',
    accent: 'rose',
    steps: [
      'Retrieved chunks are packaged as evidence-rich context.',
      'LLM produces the answer using scene metadata instead of raw guesswork.',
      'Output remains inspectable through source chunks and supporting paths.',
    ],
  },
]

const understandingSignals = [
  { label: 'Temporal grounding', value: 'Chunk timestamps -> exact scene windows' },
  { label: 'Semantic grounding', value: 'Descriptions + situations + actions' },
  { label: 'Dialogue grounding', value: 'SRT / Whisper aligned L3 evidence' },
  { label: 'Identity grounding', value: 'Characters, cast, emotion hints' },
  { label: 'Narrative grounding', value: 'Arc, causality, screenplay context' },
]

function apiUrl(path) {
  return path
}

async function safeJsonFetch(path, fallbackValue) {
  try {
    const response = await fetch(apiUrl(path))
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    return await response.json()
  } catch (error) {
    console.error(`Fetch failed for ${path}`, error)
    return fallbackValue
  }
}

function Section({ title, subtitle, children, actions, tone = 'default' }) {
  return (
    <section className={`panel panel-${tone}`}>
      <div className="panel-header">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {actions ? <div className="panel-actions">{actions}</div> : null}
      </div>
      {children}
    </section>
  )
}

function KeyValueList({ items }) {
  return (
    <div className="kv-list">
      {items.map((item) => (
        <div className="kv-item" key={item.label}>
          <span>{item.label}</span>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  )
}

function DatasetCard({ eyebrow, title, stats, body, highlights }) {
  return (
    <article className="story-card dataset-card">
      <p className="card-eyebrow">{eyebrow}</p>
      <h3>{title}</h3>
      <div className="chip-row">
        {stats.map((stat) => (
          <span className="chip" key={stat}>
            {stat}
          </span>
        ))}
      </div>
      <p className="card-copy">{body}</p>
      <ul className="story-list">
        {highlights.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </article>
  )
}

function LayerCard({ title, tag, body, fields }) {
  return (
    <article className="story-card layer-card">
      <div className="layer-card-top">
        <h3>{title}</h3>
        <span className="mini-badge">{tag}</span>
      </div>
      <p className="card-copy">{body}</p>
      <div className="field-cloud">
        {fields.map((field) => (
          <code key={field}>{field}</code>
        ))}
      </div>
    </article>
  )
}

function StageCard({ step, title, summary, bullets }) {
  return (
    <article className="stage-card">
      <div className="stage-number">{step}</div>
      <div className="stage-body">
        <h3>{title}</h3>
        <p>{summary}</p>
        <ul className="story-list">
          {bullets.map((bullet) => (
            <li key={bullet}>{bullet}</li>
          ))}
        </ul>
      </div>
    </article>
  )
}

function FlowCard({ title, steps, accent }) {
  return (
    <article className={`flow-card flow-${accent}`}>
      <h3>{title}</h3>
      <ol className="plain-list numbered">
        {steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
    </article>
  )
}

function JsonPreview({ value, emptyLabel }) {
  if (!value) return <p className="muted">{emptyLabel}</p>
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>
}

export default function App() {
  const [tab, setTab] = useState('system')
  const [overview, setOverview] = useState(null)
  const [library, setLibrary] = useState({ rows: [], summary: {} })
  const [selectedMovie, setSelectedMovie] = useState('')
  const [storage, setStorage] = useState(null)
  const [statusMessage, setStatusMessage] = useState('Frontend ready')
  const [qaResult, setQaResult] = useState(null)
  const [qaQuery, setQaQuery] = useState('')
  const [qaImage, setQaImage] = useState(null)
  const [qaVideo, setQaVideo] = useState(null)
  const [ingestMovieId, setIngestMovieId] = useState('')
  const [ingestVideo, setIngestVideo] = useState(null)
  const [ingestSubtitle, setIngestSubtitle] = useState(null)
  const [ingestForce, setIngestForce] = useState(false)
  const [ingestResult, setIngestResult] = useState(null)

  async function fetchOverview() {
    const data = await safeJsonFetch('/api/overview', null)
    setOverview(data)
    if (!data) {
      setStatusMessage('Overview API unavailable')
    }
  }

  async function fetchLibrary(movieId = '') {
    const [libraryData, storageData] = await Promise.all([
      safeJsonFetch('/api/library', { rows: [], summary: {} }),
      safeJsonFetch(`/api/storage?movie_id=${encodeURIComponent(movieId)}`, null),
    ])
    setLibrary(libraryData && typeof libraryData === 'object' ? libraryData : { rows: [], summary: {} })
    setStorage(storageData)
  }

  useEffect(() => {
    fetchOverview()
    fetchLibrary('')
  }, [])

  async function handleIngestSubmit(event) {
    event.preventDefault()
    if (!ingestMovieId || !ingestVideo) {
      setStatusMessage('Missing movie_id or video')
      return
    }
    setStatusMessage('Running ingest pipeline...')
    const form = new FormData()
    form.append('movie_id', ingestMovieId)
    form.append('video', ingestVideo)
    form.append('force', String(ingestForce))
    if (ingestSubtitle) form.append('subtitle', ingestSubtitle)

    const response = await fetch(apiUrl('/api/ingest'), {
      method: 'POST',
      body: form,
    })
    const data = await response.json()
    setIngestResult(data)
    setStatusMessage(data.success ? 'Ingest completed' : 'Ingest failed')
    await fetchLibrary(ingestMovieId)
    await fetchOverview()
  }

  async function handleStorageInspect() {
    setStatusMessage(selectedMovie ? `Inspecting ${selectedMovie}...` : 'Inspecting storage...')
    await fetchLibrary(selectedMovie)
    setStatusMessage('Storage refreshed')
  }

  async function handleAsk(event) {
    event.preventDefault()
    setStatusMessage('Running retrieval and QA...')
    const form = new FormData()
    form.append('query', qaQuery)
    form.append('history_json', '[]')
    if (qaImage) form.append('image', qaImage)
    if (qaVideo) form.append('video', qaVideo)
    const response = await fetch(apiUrl('/api/qa'), {
      method: 'POST',
      body: form,
    })
    const data = await response.json()
    setQaResult(data)
    setStatusMessage('Query completed')
  }

  const libraryRows = Array.isArray(library?.rows) ? library.rows : []
  const librarySummary = library?.summary && typeof library.summary === 'object' ? library.summary : {}

  const movieCount = overview?.library?.movie_count ?? librarySummary?.movie_count ?? libraryRows.length ?? '...'
  const indexFileCount = overview?.library?.index_file_count ?? '...'
  const knowledgeReady = overview?.library?.knowledge_ready
  const visualReady = overview?.library?.visual_ready
  const scriptReady = overview?.library?.script_scene_ready

  return (
    <div className="app-shell">
      <header className="hero-shell">
        <div className="hero-copy">
          <p className="eyebrow">MovieRAG / VideoSceneRAG Frontend</p>
          <h1>Make the system visible from data ingestion to video understanding.</h1>
          <p className="lede">
            This frontend is now framed as a system map: where the datasets come from,
            how 5-layer chunks are built, how indexes are produced, and how a query
            becomes grounded evidence and an answer.
          </p>
          <div className="hero-actions">
            <button className="primary-button" onClick={() => setTab('pipeline')}>
              Follow the data pipeline
            </button>
            <button className="ghost-button" onClick={() => setTab('workspace')}>
              Open live workspace
            </button>
          </div>
        </div>
        <aside className="hero-panel">
          <p className="hero-panel-label">Runtime snapshot</p>
          <KeyValueList
            items={[
              { label: 'Movies visible to runtime', value: movieCount },
              { label: 'Index files', value: indexFileCount },
              { label: 'Knowledge path', value: knowledgeReady ? 'Ready' : 'Missing' },
              { label: 'Visual path', value: visualReady ? 'Ready' : 'Missing' },
              { label: 'Script scene path', value: scriptReady ? 'Ready' : 'Partial' },
              { label: 'UI status', value: statusMessage },
            ]}
          />
        </aside>
      </header>

      <nav className="tabbar">
        {tabList.map((item) => (
          <button
            key={item.id}
            className={item.id === tab ? 'tab active' : 'tab'}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      {tab === 'system' && (
        <div className="page-stack">
          <Section
            title="System intent"
            subtitle="The frontend should explain the architecture, not only expose forms."
            tone="hero"
          >
            <div className="signal-grid">
              {understandingSignals.map((signal) => (
                <div className="signal-card" key={signal.label}>
                  <span>{signal.label}</span>
                  <strong>{signal.value}</strong>
                </div>
              ))}
            </div>
          </Section>

          <Section
            title="Dataset landscape"
            subtitle="Three corpora play different roles: production scene retrieval, scaling, and cross-domain benchmarking."
          >
            <div className="story-grid three-up">
              {datasetCards.map((card) => (
                <DatasetCard key={card.id} {...card} />
              ))}
            </div>
          </Section>

          <Section
            title="What video understanding means here"
            subtitle="Understanding is layered: time, semantics, speech, identity, and narrative are stored together in one scene record."
          >
            <div className="story-grid layer-grid">
              {layerCards.map((card) => (
                <LayerCard key={card.id} {...card} />
              ))}
            </div>
          </Section>
        </div>
      )}

      {tab === 'pipeline' && (
        <div className="page-stack">
          <Section
            title="Data processing pipeline"
            subtitle="This is the backbone the UI should communicate: raw assets -> aligned scene chunks -> FAISS indexes -> runtime evidence."
          >
            <div className="stage-stack">
              {pipelineStages.map((stage) => (
                <StageCard key={stage.id} {...stage} />
              ))}
            </div>
          </Section>

          <div className="grid two-col">
            <Section
              title="What the pipeline produces"
              subtitle="The frontend can frame each artifact as a product of one stage in the system."
            >
              <ul className="story-list">
                <li>`all_chunks.json` corpora for VideoRag, ActivityNet, and YouCook2.</li>
                <li>Knowledge indexes for text retrieval and optional visual indexes for media search.</li>
                <li>Per-scene evidence fields such as dialogue, characters, and screenplay context.</li>
                <li>Benchmark outputs showing retrieval behavior across domains and tasks.</li>
              </ul>
            </Section>
            <Section
              title="Frontend design target"
              subtitle="Show operators and reviewers what happens before and after a query."
            >
              <ul className="story-list">
                <li>Explain provenance of each dataset instead of presenting anonymous numbers.</li>
                <li>Expose the 5-layer schema as an interpretation scaffold for each result.</li>
                <li>Visualize runtime fan-out: knowledge search, visual search, script evidence, then answer.</li>
                <li>Keep live API tools available for ingest and QA without hiding the architecture.</li>
              </ul>
            </Section>
          </div>
        </div>
      )}

      {tab === 'flows' && (
        <div className="page-stack">
          <Section
            title="Runtime flows"
            subtitle="Queries do not just hit a box called AI. They move through retrieval, evidence shaping, and answer generation."
          >
            <div className="story-grid three-up">
              {runtimeFlows.map((flow) => (
                <FlowCard key={flow.id} {...flow} />
              ))}
            </div>
          </Section>

          <div className="grid two-col">
            <Section
              title="Evidence the UI should surface"
              subtitle="These are the signals that make an answer trustworthy."
            >
              <div className="evidence-board">
                <div className="evidence-pill">Timestamped chunk window</div>
                <div className="evidence-pill">Scene description and situation</div>
                <div className="evidence-pill">Dialogue excerpt</div>
                <div className="evidence-pill">Characters and emotions</div>
                <div className="evidence-pill">Narrative arc and screenplay context</div>
                <div className="evidence-pill">Media path or visual hit when available</div>
              </div>
            </Section>
            <Section
              title="API surface already available"
              subtitle="The current frontend can stay grounded in real endpoints instead of mock-only storytelling."
            >
              <div className="endpoint-list">
                {['GET /api/overview', 'GET /api/library', 'GET /api/storage', 'POST /api/ingest', 'POST /api/qa'].map((endpoint) => (
                  <code key={endpoint}>{endpoint}</code>
                ))}
              </div>
            </Section>
          </div>
        </div>
      )}

      {tab === 'workspace' && (
        <div className="page-stack">
          <div className="grid workspace-grid">
            <Section
              title="Live runtime snapshot"
              subtitle="Use this panel to verify the backend state the storytelling panels refer to."
            >
              <KeyValueList
                items={[
                  { label: 'Movies', value: movieCount },
                  { label: 'Indexes', value: indexFileCount },
                  { label: 'Knowledge index', value: knowledgeReady ? 'Ready' : 'Missing' },
                  { label: 'Visual index', value: visualReady ? 'Ready' : 'Missing' },
                  { label: 'Script scene index', value: scriptReady ? 'Ready' : 'Partial' },
                  { label: 'Library rows', value: libraryRows.length || '0' },
                ]}
              />
            </Section>

            <Section title="Run ingest" subtitle="Send a new video through the backend pipeline.">
              <form className="form-stack" onSubmit={handleIngestSubmit}>
                <label>
                  <span>Movie ID</span>
                  <input value={ingestMovieId} onChange={(event) => setIngestMovieId(event.target.value)} placeholder="tt0120338" />
                </label>
                <label>
                  <span>Video file</span>
                  <input type="file" accept="video/*" onChange={(event) => setIngestVideo(event.target.files?.[0] ?? null)} />
                </label>
                <label>
                  <span>Subtitle file (.srt)</span>
                  <input type="file" accept=".srt" onChange={(event) => setIngestSubtitle(event.target.files?.[0] ?? null)} />
                </label>
                <label className="checkbox-row">
                  <input type="checkbox" checked={ingestForce} onChange={(event) => setIngestForce(event.target.checked)} />
                  <span>Force rebuild artifacts</span>
                </label>
                <button className="primary-button" type="submit">Run ingest</button>
              </form>
            </Section>
          </div>

          <div className="grid workspace-grid">
            <Section
              title="Inspect storage"
              subtitle="Refresh runtime storage and inspect what exists for a specific movie id."
              actions={
                <div className="inline-actions">
                  <input
                    value={selectedMovie}
                    onChange={(event) => setSelectedMovie(event.target.value)}
                    placeholder="movie_id"
                  />
                  <button onClick={handleStorageInspect}>Inspect</button>
                </div>
              }
            >
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>movie_id</th>
                      <th>artifacts</th>
                      <th>video</th>
                      <th>subtitle</th>
                      <th>chunks</th>
                      <th>scene graph</th>
                      <th>script</th>
                    </tr>
                  </thead>
                  <tbody>
                    {libraryRows.map((row) => (
                      <tr key={row.movie_id}>
                        <td>{row.movie_id}</td>
                        <td>{row.artifact_ratio}</td>
                        <td>{row.video ? 'yes' : 'no'}</td>
                        <td>{row.subtitle ? 'yes' : 'no'}</td>
                        <td>{row.chunks}</td>
                        <td>{row.scene_graph ? 'yes' : 'no'}</td>
                        <td>{row.script_subscenes ? 'yes' : 'no'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>

            <Section title="Storage payload" subtitle="Raw backend response for the current storage inspection.">
              <JsonPreview value={storage} emptyLabel="No storage payload loaded yet." />
            </Section>
          </div>

          <div className="grid workspace-grid">
            <Section title="Ask the system" subtitle="Run retrieval and QA against the runtime service.">
              <form className="form-stack" onSubmit={handleAsk}>
                <label>
                  <span>Query</span>
                  <textarea
                    rows="5"
                    value={qaQuery}
                    onChange={(event) => setQaQuery(event.target.value)}
                    placeholder="Find the scene where Jack and Rose are running below deck, then explain why it matters."
                  />
                </label>
                <label>
                  <span>Optional image</span>
                  <input type="file" accept="image/*" onChange={(event) => setQaImage(event.target.files?.[0] ?? null)} />
                </label>
                <label>
                  <span>Optional video</span>
                  <input type="file" accept="video/*" onChange={(event) => setQaVideo(event.target.files?.[0] ?? null)} />
                </label>
                <button className="primary-button" type="submit">Run QA</button>
              </form>
            </Section>

            <Section title="QA result" subtitle="Inspect the answer and the raw response object from the API.">
              {qaResult?.answer ? <div className="answer-box">{qaResult.answer}</div> : <p className="muted">No query has been run in this session.</p>}
              <JsonPreview value={qaResult} emptyLabel="No QA payload yet." />
            </Section>
          </div>

          <div className="grid workspace-grid">
            <Section title="Latest ingest result" subtitle="Track the last backend pipeline execution.">
              <JsonPreview value={ingestResult} emptyLabel="No ingest has been run in this session." />
            </Section>
            <Section title="Overview payload" subtitle="Raw overview response from the runtime service.">
              <JsonPreview value={overview} emptyLabel="Overview not loaded." />
            </Section>
          </div>
        </div>
      )}
    </div>
  )
}
