# 🏗️ Kiến Trúc Hệ Thống: Video Understanding Pipeline

## 1. Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         VIDEO UNDERSTANDING PIPELINE                                 │
│                                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   RAW VIDEO   │───▶│   PREPROCESS │───▶│   INDEXING   │───▶│   RETRIEVAL  │     │
│  │   (MP4/AVI)   │    │   (Frames)   │    │  (FAISS L0)  │    │  (Query)     │     │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                   │                   │               │
│         ▼                   ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   WHISPER    │    │  SEMANTIC    │    │  SCENE L1   │    │   AGENTIC    │     │
│  │   (Audio)    │───▶│   SCENE      │───▶│   INDEX     │───▶│   REASONING  │     │
│  │   STT        │    │   SEGMENT    │    │  (FAISS L1) │    │   (VLM+LLM)  │     │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                   │                   │               │
│         ▼                   ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   ACTION     │    │   SCRIPT    │    │  KNOWLEDGE   │    │   KNOWLEDGE  │     │
│  │ RECOGNITION  │───▶│   ALIGN     │───▶│   (Neo4j)   │◀───│   SYNTHESIS  │     │
│  │  (VideoMAE)  │    │             │    │   GraphRAG   │    │   (LLM)      │     │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                   │                                     │
│         ▼                   ▼                   ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                    5-LAYER SCENE METADATA STORE                            │   │
│  │  Layer 1: Temporal Anchor   │  Layer 2: Semantic Description               │   │
│  │  Layer 3: Dialogue/Speech   │  Layer 4: Cast & Characters                 │   │
│  │  Layer 5: Script & Narrative│                                              │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Chi Tiết

### 2.1 Preprocessing Pipeline

```
RAW VIDEO (.mp4)
       │
       ▼
┌──────────────────┐
│  SHOT DETECTION  │  FFmpeg + PySceneDetect
│  (L0: Frames)   │  Output: List of shots [start_frame, end_frame]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ KEYFRAME         │  Uniform sampling (1fps) + Visual quality pruning
│ EXTRACTION       │  Output: List of keyframes + timestamps
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ CLIP   │ │ VLM    │
│ EMBED  │ │ SCENE  │
│ (L0)   │ │ DESC   │
└───┬────┘ └──┬────┘
    │         │
    │         ▼
    │   ┌──────────────────┐
    │   │  FACE DETECTION │
    │   │  (L4: Cast)     │
    │   └────────┬─────────┘
    │            │
    │            ▼
    │   ┌──────────────────┐
    │   │  ACTION RECOG   │
    │   │  (L4: Activity) │
    │   └────────┬─────────┘
    │            │
    │            ▼
    │   ┌──────────────────┐
    │   │  WHISPER STT    │
    │   │  (L3: Dialogue) │
    │   └────────┬─────────┘
    │            │
    ▼            ▼
┌──────────────────────────────────┐
│     SCENE SEGMENTATION (L1)      │
│  VLM + Script + Temporal Bounded │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│   5-LAYER CHUNK GENERATION       │
│  Temporal │ Semantic │ Dialogue   │
│  Cast     │ Script   │            │
└────────┬─────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ FAISS  │ │ NEO4J  │
│ INDEX  │ │ GRAPH  │
└────────┘ └────────┘
```

---

## 3. Kiến Trúc 5-Layer Scene Metadata

### 3.1 Layer Definitions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        5-LAYER SCENE METADATA MODEL                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer 1: TEMPORAL ANCHOR (Điểm neo thời gian)                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • movie_id: string — IMDb ID                                                │
│  • shot_id: int — Shot number (from PySceneDetect)                          │
│  • chunk_id: string — Unique chunk identifier                               │
│  • start_seconds: float — Start time in seconds                            │
│  • end_seconds: float — End time in seconds                                 │
│  • timestamp_source: enum — ["annotation_frame", "shot_boundary", "scene"]  │
│  • duration_seconds: float — Chunk duration                                 │
│                                                                              │
│  Layer 2: SEMANTIC DESCRIPTION (Mô tả ngữ nghĩa)                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • situation: string — Overall situation ("arguing", "cooking", "dancing")  │
│  • description: string — Detailed scene description (from VLM)              │
│  • vision_setting: string — Environment ("beach at sunset", "dark office")  │
│  • vision_actions: string — Actions ("running", "kissing", "fighting")      │
│  • visual_focus: string — Main visual subject ("close-up on face")         │
│  • vision_objects: List[string] — Objects in scene                         │
│  • emotional_tone: string — Mood ("tense", "romantic", "comedic")          │
│                                                                              │
│  Layer 3: DIALOGUE & AUDIO (Đối thoại và âm thanh)                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • dialogue_text: string — Spoken dialogue (from Whisper/SRT)               │
│  • dialogue_full_text: string — Full transcript of segment                 │
│  • speaker: string — Who is speaking                                        │
│  • audio_events: List[string] — Non-speech audio ("door creaking", "rain")  │
│  • background_music: string — Music cues                                    │
│                                                                              │
│  Layer 4: CAST & CHARACTERS (Nhân vật và diễn viên)                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • characters: List[string] — Named characters in scene                    │
│  • cast_in_scene: List[Dict] — [{"actor": "Leonardo", "character": "Jack"}] │
│  • character_emotions: Dict — {character: emotion} mapping                  │
│  • character_actions: Dict — {character: action} mapping                    │
│  • face_tracking_ids: List[string] — Track IDs for face detection          │
│  • action_labels: List[string] — Activity recognition labels               │
│                                                                              │
│  Layer 5: SCRIPT & NARRATIVE (Kịch bản và tường thuật)                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • script_primary_heading: string — INT./EXT. location heading             │
│  • script_location: string — Detailed location ("INT. TITANIC - DECK")     │
│  • script_time_of_day: string — Time of day ("DAY", "NIGHT")               │
│  • screenplay_context_excerpt: string — Script excerpt for context         │
│  • narrative_arc: enum — ["exposition", "rising_action", "climax", ...]    │
│  • causal_relations: List[Dict] — {cause: X, effect: Y}                   │
│  • scene_graph: Dict — MovieGraphs-style scene graph                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Layer Relationships

```
                    ┌─────────────────┐
                    │   SCENE (L1)     │
                    │  Temporal Unit   │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │   SHOTS   │      │  DIALOGUE │      │  SCRIPT   │
    │  (L0)     │      │   (L3)    │      │   (L5)    │
    │ Frames    │      │ Whisper   │      │ Alignment │
    └───────────┘      └───────────┘      └───────────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ SEMANTIC  │      │   CAST    │      │ NARRATIVE │
    │  (L2)     │      │   (L4)    │      │  REASON   │
    │  VLM Desc │      │   Faces   │      │  Graph    │
    └───────────┘      └───────────┘      └───────────┘
```

---

## 4. Hierarchical Indexing Architecture

### 4.1 Multi-Level FAISS Indexes

```
┌─────────────────────────────────────────────────────────────────┐
│                   HIERARCHICAL INDEX ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 0: FRAME INDEX (frame_index.faiss)                       │
│  ──────────────────────────────────────────────────────────────── │
│  • 128,410+ vectors (512-dim CLIP ViT-L/14)                    │
│  • Metadata: shot_id, timestamp, path                          │
│  • Use case: Exact visual matching, single frame retrieval       │
│  • Search latency: <50ms                                        │
│                                                                  │
│                      ▲                                           │
│                      │ grouped by scene_id                       │
│                      │                                           │
│  Level 1: SCENE INDEX (scene_index.faiss)                      │
│  ──────────────────────────────────────────────────────────────── │
│  • ~20,000 vectors (512-dim, image_mean + text_fused)          │
│  • Metadata: scene_id, characters, description, timestamp        │
│  • Weighting: 72% visual + 28% semantic text                     │
│  • Use case: Scene-level retrieval, semantic understanding      │
│  • Search latency: <30ms                                        │
│                                                                  │
│                      ▲                                           │
│                      │ grouped by movie_id                       │
│                      │                                           │
│  Level 2: EVENT INDEX (event_index.faiss) [FUTURE]              │
│  ──────────────────────────────────────────────────────────────── │
│  • Groups of consecutive scenes forming narrative events         │
│  • Metadata: event_id, causal_links, narrative_arc              │
│                                                                  │
│                      ▲                                           │
│                      │ grouped by corpus                         │
│                      │                                           │
│  Level 3: KNOWLEDGE INDEX (knowledge_index.faiss)               │
│  ──────────────────────────────────────────────────────────────── │
│  • 189,833+ vectors (512-dim text embeddings)                   │
│  • Sources: ActivityNet captions, MovieGraphs, CMU summaries     │
│  • Metadata: video_id, text, source_dataset                     │
│  • Use case: Text-based reasoning, knowledge retrieval          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Index Search Strategy

```python
# Hierarchical Search Algorithm
def hierarchical_search(query, k=5):
    # Step 1: Query L3 Knowledge Index for semantic context
    knowledge_results = knowledge_index.search(query, k=10)

    # Step 2: Use L3 results to identify candidate movies/videos
    candidate_movies = extract_movie_ids(knowledge_results)

    # Step 3: Query L1 Scene Index with movie filter
    scene_results = scene_index.search(query, k=k*2, movie_ids=candidate_movies)

    # Step 4: Query L0 Frame Index within matched scenes
    frame_results = []
    for scene in scene_results[:k]:
        frames = frame_index.search(
            query,
            k=10,
            scene_id=scene.id,
            movie_id=scene.movie_id
        )
        frame_results.extend(frames)

    # Step 5: Cross-encoder reranking
    reranked = cross_encoder_rerank(query, frame_results + scene_results)

    return reranked[:k]
```

---

## 5. Agentic Pipeline Architecture

### 5.1 Multi-Agent System Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENTIC VIDEO UNDERSTANDING PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Query ──▶ ┌──────────────────────────────────────────────────────┐   │
│                 │  Step 1: CONTEXTUALIZER Agent                          │   │
│                 │  • Rewrite query using chat history                     │   │
│                 │  • Preserve original for multimodal queries            │   │
│                 │  Output: Standalone query                               │   │
│                 └──────────────────────────┬───────────────────────────────┘   │
│                                            │                                 │
│                                            ▼                                 │
│                 ┌──────────────────────────────────────────────────────┐   │
│                 │  Step 2: INTENT ROUTER Agent (6-way)                 │   │
│                 │  • VISUAL: Need frame/image matching                 │   │
│                 │  • KNOWLEDGE: Text-based plot/metadata query        │   │
│                 │  • MULTIMODAL: Both visual + semantic                │   │
│                 │  • DIALOG: Specific dialogue/subtitle search        │   │
│                 │  • TEMPORAL: "When does X happen?"                   │   │
│                 │  • NARRATIVE: "Why does character Y do Z?"            │   │
│                 │  Output: Intent + Explicit movie name                │   │
│                 └──────────────────────────┬───────────────────────────────┘   │
│                                            │                                 │
│        ┌───────────────────────────────────┼───────────────────────────────┐  │
│        │                                   ▼                               │  │
│        │   ┌──────────────────────────────────────────────────────┐      │  │
│        │   │  Step 3: VLM MULTI-FRAME ANALYZER                   │      │  │
│        │   │  • Extract N frames from video (temporal sampling)    │      │  │
│        │   │  • VLM describes each frame in detail                 │      │  │
│        │   │  • Conflict detection (VLM vs FAISS)                │      │  │
│        │   │  • Query distillation to keywords                    │      │  │
│        │   │  Output: VLM description + Expanded queries          │      │  │
│        │   └──────────────────────────┬───────────────────────────────┘   │  │
│        │                              │                                    │  │
│        │         ┌───────────────────┼───────────────────┐                │  │
│        │         ▼                   ▼                   ▼                │  │
│        │  ┌────────────┐     ┌────────────┐     ┌────────────┐           │  │
│        │  │ LoreAgent  │     │VisualAgent │     │ScriptAgent │           │  │
│        │  │(Knowledge) │     │ (Frames)   │     │ (Script)   │           │  │
│        │  │Neo4j Graph │     │ FAISS L0   │     │ Subtitle   │           │  │
│        │  └─────┬──────┘     └─────┬──────┘     └─────┬──────┘           │  │
│        │        │                   │                  │                  │  │
│        │        └──────────────────┼──────────────────┘                │  │
│        │                             ▼                                   │  │
│        │   ┌──────────────────────────────────────────────────────┐      │  │
│        │   │  Step 4: VERIFIER Agent (Grader)                    │      │  │
│        │   │  • Evaluate: Is retrieved context sufficient?        │      │  │
│        │   │  • If INSUFFICIENT → Generate new search queries    │      │  │
│        │   │  • Loop: max 3 iterations                            │      │  │
│        │   │  Output: SUFFICIENT/INSUFFICIENT + New queries       │      │  │
│        │   └──────────────────────────┬───────────────────────────────┘   │  │
│        │                              │                                    │  │
│        │                              ▼                                    │  │
│        │   ┌──────────────────────────────────────────────────────┐      │  │
│        │   │  Step 5: JUDGEAGENT (Re-ranker + Generator)          │      │  │
│        │   │  • Tool-calling: search_knowledge, search_visual    │      │  │
│        │   │  • Cross-reference: VLM vs FAISS vs Script          │      │  │
│        │   │  • Final answer synthesis                            │      │  │
│        │   │  • Temporal grounding JSON output                    │      │  │
│        │   │  Output: Natural language answer + Citation          │      │  │
│        │   └──────────────────────────┬───────────────────────────────┘   │  │
│        │                              │                                    │  │
│        └──────────────────────────────┼────────────────────────────────┘  │
│                                         ▼                                    │
│                              ┌──────────────────────┐                        │
│                              │   FINAL RESPONSE     │                        │
│                              │  • Answer text       │                        │
│                              │  • Keyframe paths    │                        │
│                              │  • Temporal grounding│                       │
│                              │  • Reasoning trace   │                        │
│                              └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Agent Communication Protocol

```python
# Agent Message Schema
@dataclass
class AgentMessage:
    sender: str              # "Contextualizer", "VLMAnalyzer", etc.
    intent: QueryIntent     # From 6-way router
    content: Any            # Payload (text, image, results)
    metadata: Dict          # Timestamp, confidence, source
    references: List[str]   # Referenced agents


# Inter-agent communication
class AgentCoordinator:
    def __init__(self):
        self.agents = {
            "contextualizer": ContextualizerAgent(),
            "router": IntentRouterAgent(),
            "vlm": VLMAnalyzerAgent(),
            "lore": LoreAgent(),
            "visual": VisualAgent(),
            "script": ScriptAgent(),
            "verifier": VerifierAgent(),
            "judge": JudgeAgent(),
        }
        self.message_queue = []

    def send(self, message: AgentMessage):
        """Route message to appropriate agent"""
        target = self._resolve_target(message)
        self.message_queue.append((target, message))

    def broadcast(self, message: AgentMessage, exclude: List[str] = None):
        """Broadcast to all agents except excluded"""
        for name, agent in self.agents.items():
            if name not in (exclude or []):
                self.message_queue.append((name, message))
```

---

## 6. Knowledge Graph Architecture (Neo4j)

### 6.1 Graph Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NEO4J GRAPH SCHEMA                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Node Types:                                                                 │
│  ──────────                                                                 │
│  :Movie                                                                       │
│    ├── title: string                                                         │
│    ├── year: int                                                             │
│    ├── genres: list                                                          │
│    └── imdb_id: string                                                       │
│                                                                              │
│  :Character                                                                  │
│    ├── name: string                                                          │
│    ├── actor: string                                                         │
│    └── movie: :Movie                                                         │
│                                                                              │
│  :Scene                                                                       │
│    ├── scene_id: string                                                      │
│    ├── start_time: float                                                     │
│    ├── end_time: float                                                       │
│    ├── situation: string                                                     │
│    └── movie: :Movie                                                         │
│                                                                              │
│  :Event                                                                      │
│    ├── event_id: string                                                      │
│    ├── description: string                                                  │
│    └── scenes: list[:Scene]                                                  │
│                                                                              │
│  Relationship Types:                                                         │
│  ───────────────────                                                         │
│  (Character)-[:APPEARS_IN]->(Scene)                                         │
│    └── properties: {emotion: string, action: string}                        │
│                                                                              │
│  (Scene)-[:FOLLOWS]->(Scene)                                                │
│    └── properties: {causal: bool, temporal_gap: float}                    │
│                                                                              │
│  (Scene)-[:DEPICTS]->(Event)                                                │
│                                                                              │
│  (Character)-[:INTERACTS_WITH]->(Character)                                  │
│    └── properties: {type: string, scene_id: string}                          │
│                                                                              │
│  (Movie)-[:HAS_SCENE]->(Scene)                                              │
│  (Movie)-[:HAS_CHARACTER]->(Character)                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Graph Query Patterns

```cypher
// Pattern 1: Character appearance timeline
MATCH (c:Character {name: "Jack"})-[:APPEARS_IN]->(s:Scene)-[:BELONGS_TO]->(m:Movie {title: "Titanic"})
RETURN s.scene_id, s.start_time, s.situation
ORDER BY s.start_time

// Pattern 2: Causal chain of events
MATCH (e1:Event)-[:CAUSES]->(e2:Event)
WHERE e1.movie = "Titanic"
RETURN e1.description, e2.description

// Pattern 3: Multi-hop reasoning
MATCH (c1:Character)-[:INTERACTS_WITH]->(c2:Character)
WHERE c1.movie = "Titanic" AND c2.name CONTAINS "Rose"
RETURN c1.name, c2.name
```

---

## 7. VLM Integration Architecture

### 7.1 Multi-Frame VLM Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VLM MULTI-FRAME UNDERSTANDING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Video ──▶ Temporal Sampler ──▶ N Frames                               │
│                              │                                               │
│                 ┌────────────┼────────────┐                                 │
│                 ▼            ▼            ▼                                   │
│           Frame 1      Frame N/2      Frame N                                │
│                 │            │            │                                   │
│                 ▼            ▼            ▼                                   │
│         ┌────────────────────────────────────────┐                           │
│         │         VLM (Qwen2-VL / LLaVA)        │                           │
│         │                                         │                           │
│         │  Prompt: "Describe this frame in detail│                           │
│         │  including: characters, actions, setting│                           │
│         │  emotions, and notable objects."       │                           │
│         │                                         │                           │
│         │  Output: Structured Frame Description   │                           │
│         └──────────────────┬─────────────────────┘                           │
│                            │                                                 │
│                            ▼                                                 │
│         ┌────────────────────────────────────────┐                           │
│         │        Frame Description Fusion         │                           │
│         │                                         │                           │
│         │  1. Merge character mentions           │                           │
│         │  2. Align temporal actions             │                           │
│         │  3. Detect scene continuity            │                           │
│         │  4. Generate unified scene narrative   │                           │
│         │                                         │                           │
│         │  Output: Fused Scene Description       │                           │
│         └──────────────────┬─────────────────────┘                           │
│                            │                                                 │
│                 ┌──────────┴──────────┐                                      │
│                 ▼                     ▼                                       │
│        ┌──────────────┐      ┌──────────────┐                               │
│        │ Conflict     │      │ Query        │                               │
│        │ Detector     │      │ Distiller    │                               │
│        │ (VLM vs      │      │ (Keywords    │                               │
│        │  FAISS)       │      │  for search) │                               │
│        └──────────────┘      └──────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 VLM Prompt Templates

```python
# Frame Description Prompt
FRAME_DESCRIPTION_PROMPT = """
You are an expert film analyst. Analyze this movie frame carefully.

Provide the following information:
1. CHARACTERS: Who is visible? Describe their appearance, clothing, expressions.
2. SETTING: Where is this scene taking place? Time of day, location type.
3. ACTIONS: What actions are occurring? Be specific about who does what.
4. EMOTIONS: What is the emotional tone? How do characters appear to feel?
5. NOTABLE OBJECTS: What objects are prominently featured?
6. CAMERA: How is this shot framed? (close-up, wide, POV, etc.)

Format your response as JSON:
{
    "characters": [{"name": "...", "description": "...", "emotion": "..."}],
    "setting": "...",
    "actions": ["..."],
    "emotional_tone": "...",
    "objects": ["..."],
    "camera_style": "...",
    "confidence": 0.0-1.0
}
"""

# Scene Fusion Prompt
SCENE_FUSION_PROMPT = """
You are analyzing {N} consecutive frames from the same scene.
Frames are sampled at {fps} fps from a movie.

Frame 1: {desc1}
Frame 2: {desc2}
...
Frame N: {descN}

Task:
1. Identify the MAIN ACTION that spans these frames
2. List all CHARACTERS involved
3. Determine the EXACT TIMING of key moments
4. Note any CHANGES within the sequence
5. Generate a single COHERENT scene description

Output format:
{
    "scene_description": "...",
    "main_action": "...",
    "characters": ["..."],
    "key_moments": [{"time": 0.0, "description": "..."}],
    "continuity_notes": "..."
}
"""
```

---

## 8. Temporal Grounding Module

### 8.1 Temporal Reasoning Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TEMPORAL GROUNDING MODULE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Query: "When does Rose first appear in Titanic?"                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Component 1: Temporal Anchor Extraction                              │   │
│  │  ─────────────────────────────────────────────────────────────────  │   │
│  │  • Parse time expressions: "first", "last", "after X", "before Y"    │   │
│  │  • Resolve relative time: "30 minutes in", "halfway through"        │   │
│  │  • Identify temporal keywords: "opening", "ending", "climax"       │   │
│  │                                                                     │   │
│  │  Output: {entity: "Rose", relation: "first_appearance", ...}       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Component 2: Knowledge Graph Temporal Query                        │   │
│  │  ─────────────────────────────────────────────────────────────────  │   │
│  │  Neo4j Query:                                                        │   │
│  │  MATCH (c:Character {name: "Rose"})-[:APPEARS_IN]->(s:Scene)       │   │
│  │  WHERE s.movie = "Titanic"                                          │   │
│  │  RETURN s ORDER BY s.start_time LIMIT 1                            │   │
│  │                                                                     │   │
│  │  Output: {start_time: 576.0, end_time: 612.0, confidence: 0.95}    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Component 3: Cross-Reference Verification                         │   │
│  │  ─────────────────────────────────────────────────────────────────  │   │
│  │  • Verify with dialogue transcript (Whisper)                        │   │
│  │  • Cross-check with script alignment                                │   │
│  │  • Validate with subtitle timestamps                                │   │
│  │                                                                     │   │
│  │  Output: {verified: true, source: "script_alignment"}             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│                         FINAL ANSWER:                                        │
│                   "00:09:36 - 00:10:12 (576s - 612s)"                        │
│                   First scene: Southampton Dock, Rose boards Titanic        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. System Integration Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FULL SYSTEM INTEGRATION                              │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        USER INTERFACE (Gradio)                      │   │
│   │   • Text query input                                                │   │
│   │   • Image upload                                                    │   │
│   │   • Video upload                                                     │   │
│   │   • Keyframe gallery                                                 │   │
│   │   • Reasoning trace display                                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                         │
│                                     ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     ORCHESTRATION LAYER                             │   │
│   │   AgentCoordinator ──▶ Session Manager ──▶ Cache Manager            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    │              │              │                           │
│        ┌───────────┼──────────────┼──────────────┼───────────┐              │
│        │           │              │              │           │              │
│        ▼           ▼              ▼              ▼           ▼              │
│   ┌─────────┐ ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │ Context │ │ Intent  │  │   VLM    │  │ Retrieval│  │  Judge   │         │
│   │ Agent   │ │ Router  │  │ Analyzer │  │  Agents  │  │  Agent   │         │
│   └────┬────┘ └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│        │           │             │             │             │              │
│        └───────────┴─────────────┴─────────────┴─────────────┘              │
│                                     │                                         │
│                                     ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       RETRIEVAL LAYER                                │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│   │   │ FAISS    │  │ FAISS    │  │ FAISS    │  │  Neo4j   │            │   │
│   │   │ L0 Frame │  │ L1 Scene │  │ L3 Know. │  │  Graph   │            │   │
│   │   └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                         │
│                                     ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     PREPROCESSING PIPELINE                            │   │
│   │   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐          │   │
│   │   │ Shot   │  │ Key-   │  │ Whis-  │  │ VLM    │  │ Face   │          │   │
│   │   │ Detect │  │ frame  │  │ per    │  │ Scene  │  │ Detect │          │   │
│   │   └────────┘  └────────┘  └────────┘  └────────┘  └────────┘          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. API Design

### 10.1 Core Endpoints

```python
# FastAPI Endpoints
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Video Understanding API")

@app.post("/api/understand")
async def understand_video(
    query: str,
    video: UploadFile = File(None),
    image: UploadFile = File(None)
):
    """
    Main Video Understanding endpoint.

    Returns:
    - answer: Natural language answer
    - temporal_grounding: {start_time, end_time}
    - keyframes: List of relevant keyframe paths
    - reasoning_trace: Agent reasoning steps
    - confidence: Confidence score
    """
    pass

@app.post("/api/temporal_grounding")
async def temporal_grounding(
    query: str,
    video_id: str,
    temporal_constraint: str = None  # "first", "last", "during X"
):
    """Find specific moment in video based on temporal query."""
    pass

@app.post("/api/scene_understanding")
async def scene_understanding(
    video_id: str,
    start_time: float,
    end_time: float
):
    """Get deep understanding of a specific scene."""
    pass

@app.post("/api/character_tracking")
async def character_tracking(
    character_name: str,
    video_id: str
):
    """Track character appearances throughout video."""
    pass

@app.post("/api/narrative_reasoning")
async def narrative_reasoning(
    query: str,
    video_id: str
):
    """Answer causal/narrative questions about video."""
    pass
```

---

## 11. Memory và State Management

### 11.1 Multi-Level Caching

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CACHING STRATEGY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  L1: Embedding Cache (Redis)                                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Key: hash(video_id + timestamp + model_name)                            │
│  • Value: CLIP/VLM embeddings                                              │
│  • TTL: 7 days                                                             │
│  • Size: ~500MB for 100K embeddings                                        │
│                                                                              │
│  L2: Metadata Cache (SQLite)                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Key: video_id                                                           │
│  • Value: 5-layer metadata JSON                                            │
│  • TTL: Permanent (until video updated)                                   │
│                                                                              │
│  L3: Query Result Cache (in-memory LRU)                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Key: hash(query + video_id + params)                                    │
│  • Value: (answer, grounding, confidence)                                  │
│  • TTL: 1 hour                                                             │
│  • Size: 1000 most recent queries                                          │
│                                                                              │
│  L4: Session State (Redis)                                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Key: session_id                                                         │
│  • Value: chat history, user preferences                                  │
│  • TTL: 24 hours                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Error Handling và Fallback

### 12.1 Graceful Degradation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ERROR HANDLING FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Primary System Failure                                                      │
│  ─────────────────────                                                      │
│  FAISS L0 unavailable ──▶ Fallback to L1 Scene Index                       │
│  FAISS L1 unavailable ──▶ Fallback to L3 Knowledge Index                   │
│  Neo4j unavailable  ──▶ Fallback to local JSON graph                      │
│  VLM unavailable    ──▶ Fallback to CLIP embeddings only                  │
│  Whisper unavailable ──▶ Use existing SRT/subtitle files                  │
│                                                                              │
│  Partial Retrieval Results                                                   │
│  ───────────────────────                                                    │
│  No scene match ──▶ Return frame-level results with warning               │
│  No temporal grounding ──▶ Return scene-level estimate + confidence       │
│  Low confidence ──▶ Return results + suggest query refinement            │
│                                                                              │
│  External API Failures                                                      │
│  ────────────────────                                                       │
│  LLM API timeout ──▶ Retry 3x with exponential backoff                    │
│  LLM API error  ──▶ Fallback to cached responses or basic summary         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
