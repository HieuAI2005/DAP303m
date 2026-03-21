"""
Agentic VideoRAG Pipeline — Video Understanding Edition
======================================================
Multi-agent pipeline: Contextualize → Route Intent → Retrieve → Grade → Generate.
Supports: Temporal Grounding, Causal Reasoning, VLM Scene Analysis, Whisper STT.
Inspired by RagLaw LangGraph + SceneRAG + DrVideo + VideoSceneRAG patterns.
"""

import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

# ── Video Understanding Modules ─────────────────────────────────────────────────
# Lazy imports to avoid hard dependencies when modules aren't installed
_LAZY_IMPORTS = {
    "temporal_grounding": None,
    "vlm_scene_analyzer": None,
    "causal_reasoner": None,
    "whisper_transcriber": None,
    "action_recognizer": None,
    "face_tracker": None,
    "video_captioner": None,
    "script_aligner": None,
}

def _lazy_import(module_name: str):
    """Lazy import a Video Understanding module."""
    if _LAZY_IMPORTS.get(module_name) is not None:
        return _LAZY_IMPORTS[module_name]
    try:
        if module_name == "temporal_grounding":
            from movierag.indexing.temporal_grounding import TemporalGroundingEngine
        elif module_name == "vlm_scene_analyzer":
            from movierag.indexing.vlm_scene_analyzer import VLMSceneAnalyzer
        elif module_name == "causal_reasoner":
            from movierag.indexing.causal_reasoner import CausalReasoner
        elif module_name == "whisper_transcriber":
            from movierag.indexing.whisper_transcriber import WhisperTranscriber
        elif module_name == "action_recognizer":
            from movierag.indexing.action_recognizer import ActionRecognizer
        elif module_name == "face_tracker":
            from movierag.indexing.face_tracker import FaceTracker
        elif module_name == "video_captioner":
            from movierag.indexing.video_captioner import VideoCaptioner
        elif module_name == "script_aligner":
            from movierag.indexing.script_aligner import ScriptSceneAligner
        else:
            return None
        _LAZY_IMPORTS[module_name] = locals()[module_name.replace("_", "").title() + "Engine"]
        return _LAZY_IMPORTS[module_name]
    except ImportError as e:
        logger.debug(f"Lazy import of {module_name} failed: {e}")
        return None

logger = logging.getLogger(__name__)

MAX_REWRITE = 3


class QueryIntent(str, Enum):
    """
    6-way intent classification for VideoSceneRAG queries.

    Extends VideoRag's 4 intents with:
      - TEMPORAL: "When does X happen?", temporal grounding queries
      - NARRATIVE: "Why did X happen?", causal/narrative reasoning queries
    """

    VISUAL = "VISUAL"  # Needs frame/image retrieval
    KNOWLEDGE = "KNOWLEDGE"  # Text-only (cast, plot, metadata)
    MULTIMODAL = "MULTIMODAL"  # Both visual + knowledge
    CHAT = "CHAT"  # General conversation, no retrieval
    MACRO_KNOWLEDGE = "MACRO_KNOWLEDGE"  # Plot summary, big picture synopsis
    DIALOG = "DIALOG"  # Dialog path: Subtitle/quote search
    TEMPORAL = "TEMPORAL"  # Temporal grounding: "When does X happen?"
    NARRATIVE = "NARRATIVE"  # Causal reasoning: "Why did X happen?"


class AgenticVideoRAGPipeline:
    """
    Multi-agent pipeline for MovieRAG.

    Flow:
        contextualize → route_intent → [retrieve_visual | retrieve_knowledge | both]
        → grade → [rewrite × 3] → generate → format_response
    """

    def __init__(
        self,
        visual_indexer=None,
        knowledge_indexer=None,
        script_scene_indexer=None,
        dialogue_indexer=None,
        llm_generator=None,
        model_id: str = os.getenv(
            "MOVIERAG_RUNTIME_LLM_MODEL",
            os.getenv("MOVIERAG_LLM_MODEL", "moonshotai/kimi-k2-instruct"),
        ),
        api_key: Optional[str] = None,
    ):
        self.visual_indexer = visual_indexer
        self.knowledge_indexer = knowledge_indexer
        self.script_scene_indexer = script_scene_indexer
        self.dialogue_indexer = dialogue_indexer
        self.llm_generator = llm_generator
        self.model_id = model_id
        self.visual_search_strategy = os.getenv(
            "MOVIERAG_VISUAL_SEARCH_STRATEGY", "hierarchical"
        ).strip().lower()
        self.visual_score_threshold = float(
            os.getenv("MOVIERAG_VISUAL_SCORE_THRESHOLD", "0.18")
        )
        cluster_visual_weight = float(
            os.getenv("MOVIERAG_SCENE_CLUSTER_VISUAL_WEIGHT", "0.6")
        )
        cluster_script_weight = float(
            os.getenv("MOVIERAG_SCENE_CLUSTER_SCRIPT_WEIGHT", "0.4")
        )
        cluster_weight_total = cluster_visual_weight + cluster_script_weight
        if cluster_weight_total <= 0:
            cluster_visual_weight, cluster_script_weight = 0.6, 0.4
            cluster_weight_total = 1.0
        self.scene_cluster_visual_weight = (
            cluster_visual_weight / cluster_weight_total
        )
        self.scene_cluster_script_weight = (
            cluster_script_weight / cluster_weight_total
        )

        # Initialize LLM client for routing/grading
        self._llm_client = None
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._init_llm_client()

        # Grader
        from movierag.pipeline.grader import DocumentGrader

        self.grader = DocumentGrader(llm_client=self._llm_client, model_id=model_id)

        # GraphRAG store (Neo4j primary, local fallback secondary)
        self._neo4j_driver = None
        self._graph_store = None
        try:
            from movierag.indexing.neo4j_graph_store import Neo4jGraphStore

            self._graph_store = Neo4jGraphStore()
            self._neo4j_driver = getattr(self._graph_store, "driver", None)
            if self._neo4j_driver:
                logger.info("Connected to Neo4j Graph Database for GraphRAG.")
            else:
                logger.warning(
                    "Neo4j unavailable. Graph queries will use local fallback."
                )
        except Exception as e:
            logger.warning(f"Could not initialize graph store: {e}")

        # ── Video Understanding Modules (lazy-loaded) ──────────────────────────
        self._vlm_analyzer = None
        self._temporal_grounder = None
        self._causal_reasoner = None
        self._whisper_transcriber = None
        self._action_recognizer = None
        self._face_tracker = None
        self._video_captioner = None

    @property
    def vlm_analyzer(self):
        """Lazy-load VLM Scene Analyzer."""
        if self._vlm_analyzer is None:
            try:
                from movierag.indexing.vlm_scene_analyzer import VLMSceneAnalyzer
                self._vlm_analyzer = VLMSceneAnalyzer(llm_client=self._llm_client)
                logger.info("VLM Scene Analyzer initialized.")
            except ImportError as e:
                logger.warning(f"VLM Scene Analyzer not available: {e}")
        return self._vlm_analyzer

    @property
    def temporal_grounder(self):
        """Lazy-load Temporal Grounding Engine."""
        if self._temporal_grounder is None:
            try:
                from movierag.indexing.temporal_grounding import TemporalGroundingEngine
                self._temporal_grounder = TemporalGroundingEngine(
                    scene_metadata_retriever=self._scene_metadata_retriever,
                    script_retriever=self._script_retriever,
                    dialogue_retriever=self._dialogue_retriever,
                    neo4j_store=self._graph_store,
                )
                logger.info("Temporal Grounding Engine initialized.")
            except ImportError as e:
                logger.warning(f"Temporal Grounding Engine not available: {e}")
        return self._temporal_grounder

    @property
    def causal_reasoner(self):
        """Lazy-load Causal Reasoner."""
        if self._causal_reasoner is None:
            try:
                from movierag.indexing.causal_reasoner import CausalReasoner
                self._causal_reasoner = CausalReasoner(
                    neo4j_store=self._graph_store,
                    scene_retriever=self._scene_metadata_retriever,
                )
                logger.info("Causal Reasoner initialized.")
            except ImportError as e:
                logger.warning(f"Causal Reasoner not available: {e}")
        return self._causal_reasoner

    def _init_llm_client(self):
        try:
            from movierag.generation.universal_client import UniversalLLMClient

            self._llm_client = UniversalLLMClient(model_id=self.model_id)
        except Exception as e:
            logger.warning(f"Could not initialize pipeline UniversalLLMClient: {e}")

    # ─── Node 1: Contextualize ───────────────────────────────────────

    def contextualize(
        self, query: str, history: List[Dict], has_media: bool = False
    ) -> str:
        """Rewrite query using chat history for standalone context.

        Only rewrites for pure text queries. When image/video is present,
        the original query is preserved to avoid LLM misinterpreting
        the user's intent without seeing the media.
        """
        # Skip rewrite for multimodal queries — media provides its own context
        if has_media:
            return query

        if not history or not self._llm_client:
            return query

        # Only use last 3 turns for context
        recent = history[-6:]  # 3 user + 3 assistant messages
        history_text = ""
        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]
                history_text += f"{role}: {content}\n"

        if not history_text.strip():
            return query

        try:
            prompt = (
                f"Dựa trên lịch sử chat:\n{history_text}\n\n"
                f"Câu hỏi mới: {query}\n\n"
                f"Viết lại câu hỏi thành câu độc lập, đầy đủ ngữ cảnh. "
                f"Nếu đã đầy đủ, giữ nguyên. Chỉ trả về câu hỏi mới."
            )
            response = self._llm_client.models.generate_content(
                model=self.model_id, contents=prompt
            )
            rewritten = response.text.strip()
            if rewritten and len(rewritten) < 500:
                logger.info(f"[Contextualize] '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Contextualize failed: {e}")

        return query

    # ─── Node 2: Route Intent & Extract Explicit Movie ─────────────────────────────

    def route_intent_and_movie(
        self, query: str, has_image: bool = False
    ) -> tuple[QueryIntent, str]:
        """Classify user intent AND extract explicit movie name using language model."""
        if has_image:
            logger.info("[Route] Image detected -> MULTIMODAL")
            # We still might want to extract movie name if they gave one with the image!
            # Let's run the LLM even if has_image is true, just force the intent to MULTIMODAL.

        if not self._llm_client:
            return (QueryIntent.MULTIMODAL if has_image else QueryIntent.KNOWLEDGE), ""

        prompt = (
            "Bạn là một hệ thống phân tích câu hỏi của người dùng về ĐIỆN ẢNH.\n"
            "Nhiệm vụ 1: Phân loại ý định người dùng thành 1 trong 6 loại:\n"
            "- VISUAL: Yêu cầu xem hình ảnh, xem mặt mũi, xem cách bố trí, hình dáng, màu sắc.\n"
            "- KNOWLEDGE: Tìm thông tin chữ (Ai đóng vai?, Tóm tắt phim, Diễn viên là ai?, Quote này của ai?).\n"
            "- MULTIMODAL: Yêu cầu phức tạp cần đọc cả nội dung LẪN xem hình ảnh để kết luận.\n"
            "- CHAT: Trò chuyện bình thường, chào hỏi và trả lời thân thiện.\n"
            "- TEMPORAL: Hỏi về THỜI GIAN — 'Khi nào X xảy ra?', 'Cảnh nào X xuất hiện lần đầu?', 'Lúc mấy giờ...?', 'Tìm cảnh where...'.\n"
            "- NARRATIVE: Hỏi về NGUYÊN NHÂN — 'Tại sao X xảy ra?', 'Điều gì khiến X...?', 'Điều gì dẫn đến...?'.\n\n"
            "Nhiệm vụ 2: RÚT TRÍCH TÊN PHIM (EXPLICIT MOVIE NAME)\n"
            "- Nếu người dùng CỐ TÌNH NHẮC ĐẾN tên một bộ phim cụ thể (Ví dụ: 'Trong phim Titanic...', 'Watchmen có cảnh nào...'), hãy rút trích tên phim đó ra.\n"
            "- Nếu câu hỏi chung chung không nhắc rõ tên phim (VD: 'Cảnh ông già lùn trần truồng là phim nào?'), để trống.\n\n"
            "CHỈ OUTPUT JSON THEO ĐỊNH DẠNG SAU, KHÔNG GIẢI THÍCH:\n"
            '{"intent": "TEMPORAL", "explicit_movie": "Titanic"}\n\n'
            f"Câu hỏi: {query}\n"
            "JSON:"
        )
        try:
            res = self._llm_client.models.generate_content(
                model=self.model_id, contents=prompt
            )
            # parse json
            import json as _json
            import re as _re

            text = res.text.strip()
            # Clean markdown code blocks
            text = _re.sub(r"```json\n|\n```|```", "", text).strip()
            data = _json.loads(text)

            intent_str = data.get("intent", "").upper()
            explicit_movie = data.get("explicit_movie", "").strip()

            # If user uploaded an image, override intent to MULTIMODAL regardless of text
            final_intent = (
                QueryIntent.MULTIMODAL if has_image else QueryIntent.KNOWLEDGE
            )

            if not has_image:
                if "TEMPORAL" in intent_str:
                    final_intent = QueryIntent.TEMPORAL
                elif "NARRATIVE" in intent_str:
                    final_intent = QueryIntent.NARRATIVE
                elif "VISUAL" in intent_str:
                    final_intent = QueryIntent.VISUAL
                elif "CHAT" in intent_str:
                    final_intent = QueryIntent.CHAT
                elif "MULTIMODAL" in intent_str:
                    final_intent = QueryIntent.MULTIMODAL

            logger.info(
                f"[Route] '{query}' -> Intent: {final_intent.name}, Movie: '{explicit_movie}'"
            )
            return final_intent, explicit_movie

        except Exception as e:
            logger.warning(f"Intent & Movie routing LLM failed: {e}")
            return (QueryIntent.MULTIMODAL if has_image else QueryIntent.KNOWLEDGE), ""

    # ─── Node 3: Retrieve ────────────────────────────────────────────

    def retrieve_visual(
        self, query: str, k: int = 6, movie_id: Optional[str] = None
    ) -> list:
        """Retrieve visual frames using CLIP FAISS."""
        if not self.visual_indexer:
            return []
        if not (
            hasattr(self.visual_indexer, "_is_loaded")
            and self.visual_indexer._is_loaded
        ):
            return []

        try:
            search_k = max(k * 8, 24) if movie_id else k
            if (
                self.visual_search_strategy == "basic"
                and hasattr(self.visual_indexer, "search_by_text")
            ):
                res = self.visual_indexer.search_by_text(
                    query, k=search_k, movie_id=movie_id
                )
            elif (
                self.visual_search_strategy == "hybrid"
                and hasattr(self.visual_indexer, "hybrid_search")
            ):
                res = self.visual_indexer.hybrid_search(
                    query, k=search_k, movie_id=movie_id
                )
            elif hasattr(self.visual_indexer, "hierarchical_search"):
                res = self.visual_indexer.hierarchical_search(
                    query, k=search_k, scene_k=5, movie_id=movie_id
                )
            else:
                res = self.visual_indexer.search_by_text(
                    query, k=search_k, movie_id=movie_id
                )

            # Filter low confidence
            filtered_res = [
                r
                for r in res
                if getattr(r, "score", 0.0) >= self.visual_score_threshold
            ]
            for item in filtered_res:
                metadata = getattr(item, "metadata", None)
                if not isinstance(metadata, dict):
                    metadata = {}
                    setattr(item, "metadata", metadata)
                metadata["_rerank_score"] = float(
                    getattr(item, "score", 0.0) or 0.0
                ) + self._visual_lexical_bonus(query, item)
            if movie_id:
                movie_filtered = [
                    r for r in filtered_res if getattr(r, "movie_id", "") == movie_id
                ]
                if movie_filtered:
                    return self._sort_results_by_score(movie_filtered)[:k]
            return self._sort_results_by_score(filtered_res)[:k]
        except Exception as e:
            logger.error(f"Visual retrieval failed: {e}")
            return []

    def retrieve_visual_by_image(self, image_path: str, k: int = 6) -> list:
        """Retrieve visual frames by image similarity (FAISS direct)."""
        if not self.visual_indexer:
            return []
        try:
            results = self.visual_indexer.search_by_image(
                image_path, k=k, exclude_same=False
            )
            return results[:k]
        except Exception as e:
            logger.error(f"Visual image retrieval failed: {e}")
            return []

    def retrieve_knowledge(
        self, query: str, k: int = 5, movie_id: Optional[str] = None
    ) -> list:
        """Retrieve text documents from knowledge FAISS index."""
        if not self.knowledge_indexer:
            return []
        # Let exceptions bubble up to the grading loop so they can be captured in thoughts
        return self.knowledge_indexer.search(query, k=k, movie_id=movie_id)

    def retrieve_script_scenes(
        self, query: str, k: int = 4, movie_id: Optional[str] = None
    ) -> list:
        """Retrieve screenplay-derived sub-scenes from the dedicated index."""
        if not self.script_scene_indexer:
            return []
        if not (
            hasattr(self.script_scene_indexer, "_is_loaded")
            and self.script_scene_indexer._is_loaded
        ):
            return []
        return self.script_scene_indexer.search(query, k=k, movie_id=movie_id)

    def retrieve_graph_context(
        self, query: str, k: int = 4, movie_id: Optional[str] = None
    ) -> list:
        """Retrieve graph-backed evidence from Neo4j or local graph artifacts."""
        graph_store = getattr(self, "_graph_store", None)
        if not graph_store:
            return []

        try:
            docs = graph_store.search_as_documents(query, movie_id=movie_id, limit=k)
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []

        try:
            from movierag.indexing.knowledge_indexer import TextSearchResult
        except Exception:
            return []

        results = []
        for doc in docs:
            results.append(
                TextSearchResult(
                    movie_id=doc.get("movie_id", movie_id or "unknown"),
                    clip_id=doc.get("clip_id", "graph_node"),
                    text=doc.get("text", ""),
                    score=float(doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                )
            )
        return results

    def query_graph(
        self, query: str, movie_id: Optional[str] = None, limit: int = 5
    ) -> list:
        """Query the graph store directly for tool-calling or diagnostics."""
        graph_store = getattr(self, "_graph_store", None)
        if not graph_store:
            return [{"error": "graph_store_unavailable", "query": query}]
        return graph_store.search(query, movie_id=movie_id, limit=limit)

    @staticmethod
    def _result_key(result: Any) -> str:
        return str(
            getattr(
                result,
                "clip_id",
                getattr(result, "id", getattr(result, "chunk_id", str(result))),
            )
        )

    @staticmethod
    def _result_sort_score(result: Any) -> float:
        metadata = getattr(result, "metadata", {}) or {}
        if isinstance(metadata, dict) and "_rerank_score" in metadata:
            try:
                return float(metadata["_rerank_score"] or 0.0)
            except Exception:
                pass
        return float(getattr(result, "score", 0.0) or 0.0)

    @staticmethod
    def _normalize_text_for_match(value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"[^\w\s]", " ", str(value).lower()).strip()

    @staticmethod
    def _stem_token(token: str) -> str:
        token = str(token or "").strip().lower()
        if not token:
            return ""
        for suffix in ("ing", "ers", "ies", "ied", "ed", "es", "s"):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    @classmethod
    def _visual_lexical_bonus(cls, query: str, result: Any) -> float:
        metadata = getattr(result, "metadata", {}) or {}
        query_norm = cls._normalize_text_for_match(query)
        if not query_norm:
            return 0.0

        text_parts = [
            metadata.get("title", ""),
            metadata.get("description", ""),
            metadata.get("vision_actions", ""),
            metadata.get("vision_setting", ""),
            metadata.get("visual_focus", ""),
            metadata.get("scene_label", ""),
            metadata.get("script_primary_heading", ""),
            metadata.get("script_location", ""),
            metadata.get("dialogue_text", ""),
            " ".join(metadata.get("vision_objects", []) or []),
            " ".join(metadata.get("characters", []) or []),
        ]
        combined = cls._normalize_text_for_match(" ".join(part for part in text_parts if part))
        if not combined:
            return 0.0
        heading_norm = cls._normalize_text_for_match(
            metadata.get("script_primary_heading", "") or metadata.get("scene_label", "")
        )

        query_tokens = [cls._stem_token(tok) for tok in query_norm.split() if tok]
        combined_tokens = {
            cls._stem_token(tok) for tok in combined.split() if tok
        }
        bonus = 0.0

        if query_norm in combined:
            bonus += 0.35

        overlap = sum(1 for token in query_tokens if token and token in combined_tokens)
        if query_tokens:
            bonus += 0.45 * (overlap / len(query_tokens))

        disaster_aliases = {
            "chìm": {"chìm", "chim", "sink", "sinking", "plunge", "plung", "submerg", "underwater", "drown"},
            "tàu": {"tau", "ship", "boat", "liner", "vessel", "titanic"},
            "gãy": {"gay", "break", "broke", "broken", "splitting", "split", "hull", "stern", "bow", "final"},
            "nước": {"nuoc", "water", "flood", "flooding", "ocean", "sea"},
        }
        disaster_terms = set()
        for variants in disaster_aliases.values():
            disaster_terms.update(variants)

        query_has_disaster = any(token in disaster_terms for token in query_tokens)
        ship_sinking_query = (
            any(token in query_tokens for token in {"tau", "tàu", "ship", "boat", "titanic"})
            and any(
                token in query_tokens
                for token in {"chim", "chìm", "sink", "sinking", "plunge", "break", "split", "gay", "gãy"}
            )
        )
        if query_has_disaster:
            matched_disaster = sum(
                1 for token in disaster_terms if token in combined_tokens
            )
            if matched_disaster:
                bonus += min(0.9, 0.08 * matched_disaster)

            if (
                any(token in query_tokens for token in {"sink", "sinking", "chim", "chìm", "plunge"})
                and any(token in combined_tokens for token in {"sink", "sinking", "plunge", "underwater", "flood", "flooding"})
            ):
                bonus += 0.75

            if (
                any(token in query_tokens for token in {"split", "splitting", "break", "gãy", "gay"})
                and any(token in combined_tokens for token in {"split", "splitting", "break", "broken", "hull", "stern", "bow"})
            ):
                bonus += 0.55

            if "titanic" in query_tokens and "titanic" in combined_tokens:
                bonus += 0.2

        start_seconds = metadata.get("start_seconds")
        if query_has_disaster and isinstance(start_seconds, (int, float)):
            if float(start_seconds) >= 7200:
                bonus += 0.2
            elif float(start_seconds) < 5400:
                bonus -= 0.1

        if ship_sinking_query:
            catastrophic_terms = {
                "sink",
                "sinking",
                "plunge",
                "underwater",
                "submerg",
                "split",
                "splitting",
                "break",
                "broken",
                "stern",
                "bow",
                "funnel",
                "hull",
            }
            ship_scope_terms = {
                "ship",
                "deck",
                "bridge",
                "funnel",
                "stern",
                "bow",
                "ocean",
                "surface",
                "port",
                "starboard",
                "water",
            }
            catastrophic_phrases = (
                "under water",
                "sinking ship",
                "ship is sinking",
                "final plunge",
                "split in half",
                "break in half",
                "chaos on deck",
            )
            local_interior_terms = {
                "office",
                "room",
                "cabin",
                "corridor",
                "hall",
                "stateroom",
                "master at arms",
            }

            scope_hits = sum(1 for token in ship_scope_terms if token in combined_tokens)
            catastrophe_hits = sum(
                1 for token in catastrophic_terms if token in combined_tokens
            )
            phrase_hits = sum(1 for phrase in catastrophic_phrases if phrase in combined)
            interior_hits = sum(1 for token in local_interior_terms if token in heading_norm)

            if heading_norm.startswith("ext"):
                bonus += 0.35
            elif heading_norm.startswith("int"):
                bonus -= 0.2

            if scope_hits:
                bonus += min(0.45, 0.07 * scope_hits)
            if catastrophe_hits:
                bonus += min(0.85, 0.16 * catastrophe_hits)
            if phrase_hits:
                bonus += min(1.1, 0.35 * phrase_hits)

            if catastrophe_hits == 0 and phrase_hits == 0:
                bonus -= 0.4
            if interior_hits:
                bonus -= min(0.45, 0.15 * interior_hits)

            if isinstance(start_seconds, (int, float)):
                if float(start_seconds) >= 9000:
                    bonus += 0.25
                elif float(start_seconds) >= 8400:
                    bonus += 0.1

            if "under water" in combined or "sinking ship" in combined:
                bonus += 0.45
            if "flood" in combined and "under water" not in combined and "sinking ship" not in combined:
                bonus -= 0.15

        return bonus

    @classmethod
    def _sort_results_by_score(cls, results: list) -> list:
        return sorted(
            results,
            key=cls._result_sort_score,
            reverse=True,
        )

    @staticmethod
    def _result_metadata(result: Any) -> Dict[str, Any]:
        metadata = getattr(result, "metadata", {}) or {}
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _first_non_empty(*values: Any) -> Any:
        for value in values:
            if value not in ("", None, [], {}, ()):
                return value
        return ""

    @classmethod
    def _scene_identity_from_metadata(
        cls, metadata: Dict[str, Any], fallback: str
    ) -> tuple[str, str]:
        scene_context = metadata.get("scene_context", {}) or {}
        if metadata.get("scene_group_id") or scene_context.get("scene_group_id"):
            return (
                str(
                    metadata.get("scene_group_id")
                    or scene_context.get("scene_group_id")
                ),
                str(
                    metadata.get("scene_group_type")
                    or scene_context.get("scene_group_type")
                    or "scene"
                ),
            )
        if metadata.get("script_scene_uid"):
            return f"script_scene::{metadata['script_scene_uid']}", "script_scene"
        if metadata.get("parent_scene_id"):
            return f"semantic_scene::{metadata['parent_scene_id']}", "semantic_scene"
        if metadata.get("subscene_id"):
            return f"script_subscene::{metadata['subscene_id']}", "script_subscene"
        if metadata.get("chunk_id"):
            return f"chunk::{metadata['chunk_id']}", "chunk"
        if metadata.get("clip_id"):
            return f"clip::{metadata['clip_id']}", "clip"
        return fallback, "candidate"

    @classmethod
    def _scene_heading_from_metadata(cls, metadata: Dict[str, Any]) -> str:
        scene_context = metadata.get("scene_context", {}) or {}
        return str(
            cls._first_non_empty(
                metadata.get("script_primary_heading"),
                metadata.get("script_heading"),
                scene_context.get("script_primary_heading"),
                metadata.get("scene_label"),
                scene_context.get("scene_label"),
                metadata.get("script_location"),
                "Scene cluster",
            )
        )

    @staticmethod
    def _scene_time_fields(metadata: Dict[str, Any]) -> Dict[str, Any]:
        scene_context = metadata.get("scene_context", {}) or {}
        start_seconds = AgenticVideoRAGPipeline._first_non_empty(
            metadata.get("start_seconds"),
            scene_context.get("scene_timestamp"),
        )
        end_seconds = AgenticVideoRAGPipeline._first_non_empty(
            metadata.get("end_seconds"),
            scene_context.get("scene_timestamp_end"),
        )
        return {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "start_time": AgenticVideoRAGPipeline._first_non_empty(
                metadata.get("start_time"),
                scene_context.get("start_time"),
            ),
            "end_time": AgenticVideoRAGPipeline._first_non_empty(
                metadata.get("end_time"),
                scene_context.get("end_time"),
            ),
        }

    @classmethod
    def _scene_ranking_key(cls, cluster: Dict[str, Any]) -> tuple:
        return (
            float(cluster.get("score", 0.0) or 0.0),
            float(cluster.get("best_visual_score", 0.0) or 0.0),
            float(cluster.get("best_script_score", 0.0) or 0.0),
            len(cluster.get("visual_hits", []) or []),
            len(cluster.get("script_hits", []) or []),
        )

    def _build_scene_results(
        self, visual_results: List[Any], script_results: List[Any], limit: int = 6
    ) -> List[Dict[str, Any]]:
        clusters: Dict[str, Dict[str, Any]] = {}

        def ensure_cluster(metadata: Dict[str, Any], fallback: str) -> Dict[str, Any]:
            cluster_id, cluster_type = self._scene_identity_from_metadata(
                metadata, fallback
            )
            cluster = clusters.get(cluster_id)
            if cluster is None:
                time_fields = self._scene_time_fields(metadata)
                cluster = {
                    "id": cluster_id,
                    "cluster_type": cluster_type,
                    "scene_group_id": self._first_non_empty(
                        metadata.get("scene_group_id"),
                        (metadata.get("scene_context", {}) or {}).get(
                            "scene_group_id"
                        ),
                    ),
                    "movie_id": self._first_non_empty(
                        metadata.get("movie_id"),
                        getattr(metadata, "movie_id", ""),
                    ),
                    "heading": self._scene_heading_from_metadata(metadata),
                    "location": self._first_non_empty(
                        metadata.get("script_location"),
                        metadata.get("location"),
                    ),
                    "scene_label": self._first_non_empty(
                        metadata.get("scene_label"),
                        (metadata.get("scene_context", {}) or {}).get("scene_label"),
                    ),
                    "start_seconds": time_fields["start_seconds"],
                    "end_seconds": time_fields["end_seconds"],
                    "start_time": time_fields["start_time"],
                    "end_time": time_fields["end_time"],
                    "best_visual_score": 0.0,
                    "best_script_score": 0.0,
                    "best_scene_score": 0.0,
                    "score": 0.0,
                    "representative_frame": "",
                    "visual_hits": [],
                    "script_hits": [],
                    "visual_cues": [],
                    "script_subscenes": [],
                    "keyframe_paths": [],
                    "characters": set(),
                    "script_characters": set(),
                    "clip_ids": set(),
                    "chunk_ids": set(),
                    "semantic_scene_ids": set(),
                    "script_scene_uids": set(),
                    "script_subscene_ids": set(),
                }
                clusters[cluster_id] = cluster
            return cluster

        def update_time_window(cluster: Dict[str, Any], metadata: Dict[str, Any]) -> None:
            time_fields = self._scene_time_fields(metadata)
            start_seconds = time_fields["start_seconds"]
            end_seconds = time_fields["end_seconds"]
            if isinstance(start_seconds, (int, float)):
                existing = cluster.get("start_seconds")
                if not isinstance(existing, (int, float)) or start_seconds < existing:
                    cluster["start_seconds"] = float(start_seconds)
                    if time_fields["start_time"]:
                        cluster["start_time"] = time_fields["start_time"]
            elif not cluster.get("start_time") and time_fields["start_time"]:
                cluster["start_time"] = time_fields["start_time"]
            if isinstance(end_seconds, (int, float)):
                existing = cluster.get("end_seconds")
                if not isinstance(existing, (int, float)) or end_seconds > existing:
                    cluster["end_seconds"] = float(end_seconds)
                    if time_fields["end_time"]:
                        cluster["end_time"] = time_fields["end_time"]
            elif not cluster.get("end_time") and time_fields["end_time"]:
                cluster["end_time"] = time_fields["end_time"]

        for index, result in enumerate(visual_results[:12]):
            metadata = self._result_metadata(result)
            cluster = ensure_cluster(metadata, f"visual::{index}")
            update_time_window(cluster, metadata)

            score = self._result_sort_score(result)
            cluster["best_visual_score"] = max(cluster["best_visual_score"], score)
            scene_score = float(
                (metadata.get("scene_context", {}) or {}).get("scene_score", 0.0) or 0.0
            )
            cluster["best_scene_score"] = max(cluster["best_scene_score"], scene_score)

            if not cluster["movie_id"]:
                cluster["movie_id"] = self._first_non_empty(
                    getattr(result, "movie_id", ""),
                    metadata.get("movie_id"),
                )
            if not cluster["heading"] or cluster["heading"] == "Scene cluster":
                cluster["heading"] = self._scene_heading_from_metadata(metadata)
            if not cluster["location"]:
                cluster["location"] = self._first_non_empty(
                    metadata.get("script_location"),
                    metadata.get("location"),
                )
            if not cluster["scene_label"]:
                cluster["scene_label"] = metadata.get("scene_label", "")

            clip_id = self._first_non_empty(
                metadata.get("parent_clip_id"),
                metadata.get("clip_id"),
            )
            if clip_id:
                cluster["clip_ids"].add(str(clip_id))
            if metadata.get("chunk_id"):
                cluster["chunk_ids"].add(str(metadata["chunk_id"]))
            if metadata.get("parent_scene_id"):
                cluster["semantic_scene_ids"].add(str(metadata["parent_scene_id"]))
            if metadata.get("script_scene_uid"):
                cluster["script_scene_uids"].add(str(metadata["script_scene_uid"]))
            for script_scene_uid in metadata.get("script_scene_uids", []) or []:
                if script_scene_uid:
                    cluster["script_scene_uids"].add(str(script_scene_uid))

            for character in metadata.get("characters", []) or []:
                if character:
                    cluster["characters"].add(str(character))
            for character in metadata.get("script_characters", []) or []:
                if character:
                    cluster["script_characters"].add(str(character))

            frame_path = self._first_non_empty(
                getattr(result, "path", ""),
                metadata.get("path"),
            )
            if frame_path and frame_path not in cluster["keyframe_paths"]:
                cluster["keyframe_paths"].append(frame_path)
            if frame_path and (
                not cluster["representative_frame"]
                or score >= cluster.get("representative_frame_score", 0.0)
            ):
                cluster["representative_frame"] = frame_path
                cluster["representative_frame_score"] = score

            cue = " | ".join(
                str(part).strip()
                for part in (
                    metadata.get("vision_setting", ""),
                    metadata.get("vision_actions", ""),
                    metadata.get("visual_focus", ""),
                )
                if str(part).strip()
            )
            if cue and cue not in cluster["visual_cues"]:
                cluster["visual_cues"].append(cue)

            cluster["visual_hits"].append(
                {
                    "id": getattr(result, "id", metadata.get("id", "")),
                    "score": score,
                    "start_time": metadata.get("start_time", ""),
                    "end_time": metadata.get("end_time", ""),
                    "shot_id": metadata.get("shot_id", ""),
                    "chunk_id": metadata.get("chunk_id", ""),
                }
            )

        for index, result in enumerate(script_results[:12]):
            metadata = self._result_metadata(result)
            cluster = ensure_cluster(metadata, f"script::{index}")
            update_time_window(cluster, metadata)

            score = self._result_sort_score(result)
            cluster["best_script_score"] = max(cluster["best_script_score"], score)
            if not cluster["movie_id"]:
                cluster["movie_id"] = self._first_non_empty(
                    getattr(result, "movie_id", ""),
                    metadata.get("movie_id"),
                )
            if not cluster["heading"] or cluster["heading"] == "Scene cluster":
                cluster["heading"] = self._scene_heading_from_metadata(metadata)
            if not cluster["location"]:
                cluster["location"] = self._first_non_empty(
                    metadata.get("script_location"),
                    metadata.get("location"),
                )
            if metadata.get("parent_scene_id"):
                cluster["semantic_scene_ids"].add(str(metadata["parent_scene_id"]))
            if metadata.get("script_scene_uid"):
                cluster["script_scene_uids"].add(str(metadata["script_scene_uid"]))
            if metadata.get("subscene_id"):
                cluster["script_subscene_ids"].add(str(metadata["subscene_id"]))
            if metadata.get("parent_chunk_id"):
                cluster["chunk_ids"].add(str(metadata["parent_chunk_id"]))
            if metadata.get("clip_id"):
                cluster["clip_ids"].add(str(metadata["clip_id"]))
            for character in metadata.get("script_characters", []) or []:
                if character:
                    cluster["script_characters"].add(str(character))

            script_hit = {
                "id": getattr(result, "clip_id", getattr(result, "id", "")),
                "score": score,
                "heading": metadata.get("script_heading", ""),
                "location": metadata.get("script_location", ""),
                "start_time": metadata.get("start_time", ""),
                "end_time": metadata.get("end_time", ""),
                "subscene_id": metadata.get("subscene_id", ""),
            }
            cluster["script_hits"].append(script_hit)
            if metadata.get("script_heading") or metadata.get("start_time"):
                cluster["script_subscenes"].append(script_hit)

        scene_results: List[Dict[str, Any]] = []
        for cluster in clusters.values():
            cluster["visual_hits"].sort(key=lambda item: float(item["score"]), reverse=True)
            cluster["script_hits"].sort(key=lambda item: float(item["score"]), reverse=True)
            cluster["script_subscenes"] = cluster["script_subscenes"][:4]
            cluster["score"] = max(
                (
                    self.scene_cluster_visual_weight * cluster["best_visual_score"]
                    + self.scene_cluster_script_weight * cluster["best_script_score"]
                ),
                cluster["best_scene_score"],
            )
            cluster["clip_ids"] = sorted(cluster["clip_ids"])
            cluster["chunk_ids"] = sorted(cluster["chunk_ids"])
            cluster["semantic_scene_ids"] = sorted(cluster["semantic_scene_ids"])
            cluster["script_scene_uids"] = sorted(cluster["script_scene_uids"])
            cluster["script_subscene_ids"] = sorted(cluster["script_subscene_ids"])
            cluster["characters"] = sorted(cluster["characters"])
            cluster["script_characters"] = sorted(cluster["script_characters"])
            cluster["evidence_count"] = len(cluster["visual_hits"]) + len(
                cluster["script_hits"]
            )
            cluster.pop("representative_frame_score", None)
            scene_results.append(cluster)

        scene_results.sort(key=self._scene_ranking_key, reverse=True)
        return scene_results[:limit]

    def _format_scene_results_for_prompt(
        self, scene_results: List[Dict[str, Any]], limit: int = 4
    ) -> str:
        """Format fused scene clusters for prompt consumption."""
        if not scene_results:
            return ""

        lines = []
        for i, scene in enumerate(scene_results[:limit]):
            heading = self._first_non_empty(
                scene.get("heading"), scene.get("scene_label"), "Scene cluster"
            )
            location = str(scene.get("location", "") or "").strip()
            start_time = str(scene.get("start_time", "") or "")
            end_time = str(scene.get("end_time", "") or "")
            cluster_type = str(scene.get("cluster_type", "") or "")
            score = float(scene.get("score", 0.0) or 0.0)
            visual_score = float(scene.get("best_visual_score", 0.0) or 0.0)
            script_score = float(scene.get("best_script_score", 0.0) or 0.0)
            evidence_count = int(scene.get("evidence_count", 0) or 0)

            lines.append(
                f"[{i + 1}] {heading} | {start_time}->{end_time} | type={cluster_type} | "
                f"score={score:.3f} | visual={visual_score:.3f} | script={script_score:.3f} | evidence={evidence_count}"
            )
            if location:
                lines.append(f"location: {location}")

            visual_cues = [
                str(v).strip() for v in (scene.get("visual_cues", []) or []) if str(v).strip()
            ]
            if visual_cues:
                lines.append("visual_cues: " + ", ".join(visual_cues[:3]))

            characters = [
                str(v).strip() for v in (scene.get("characters", []) or []) if str(v).strip()
            ]
            if characters:
                lines.append("characters: " + ", ".join(characters[:6]))

            script_characters = [
                str(v).strip()
                for v in (scene.get("script_characters", []) or [])
                if str(v).strip()
            ]
            if script_characters:
                lines.append("script_characters: " + ", ".join(script_characters[:6]))

            script_heads = [
                str(item.get("heading", "")).strip()
                for item in (scene.get("script_subscenes", []) or [])[:3]
                if str(item.get("heading", "")).strip()
            ]
            if script_heads:
                lines.append("script_windows: " + " | ".join(script_heads))

            semantic_ids = [
                str(v).strip()
                for v in (scene.get("semantic_scene_ids", []) or [])
                if str(v).strip()
            ]
            if semantic_ids:
                lines.append("semantic_scene_ids: " + ", ".join(semantic_ids[:3]))

            script_scene_uids = [
                str(v).strip()
                for v in (scene.get("script_scene_uids", []) or [])
                if str(v).strip()
            ]
            if script_scene_uids:
                lines.append("script_scene_uids: " + ", ".join(script_scene_uids[:3]))

            representative_frame = str(scene.get("representative_frame", "") or "")
            if representative_frame:
                lines.append(f"representative_frame: {os.path.basename(representative_frame)}")

            lines.append("")

        return "\n".join(lines).strip()

    # ─── Node 5: Generate ────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        intent: QueryIntent,
        visual_results: list,
        knowledge_results: list,
        script_results: list,
        scene_results: list,
        history: List[Dict],
    ) -> str:
        """Generate final answer using LLM."""
        if intent == QueryIntent.CHAT:
            return self._generate_chat(query, history)

        if not self.llm_generator:
            return "⚠️ LLM chưa được khởi tạo."

        try:
            # Map intent to route for LLM prompt
            route = self._intent_to_route(intent)
            return self.llm_generator.generate_answer(
                query=query,
                context_results=knowledge_results,
                visual_results=visual_results,
                script_results=script_results,
                scene_results=scene_results,
                history=history,
                route=route,
            )
        except Exception as e:
            return f"⚠️ LLM Error: {e}"

    def _generate_chat(self, query: str, history: List[Dict]) -> str:
        """Direct chat response without retrieval."""
        if not self._llm_client:
            return "Xin chào! Tôi là MovieRAG, hệ thống tìm kiếm phim thông minh. Bạn có thể hỏi về phim, tìm cảnh, hoặc tra cứu thông tin diễn viên."
        try:
            prompt = (
                "Bạn là MovieRAG, trợ lý tìm kiếm phim thông minh. "
                "Trả lời ngắn gọn, thân thiện.\n\n"
                f"Câu hỏi: {query}"
            )
            response = self._llm_client.models.generate_content(
                model=self.model_id, contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Xin chào! Tôi là MovieRAG. Hãy hỏi tôi về phim! (Error: {e})"

    def _intent_to_route(self, intent: QueryIntent):
        """Map QueryIntent to old QueryRoute for backward compatibility."""
        try:
            from movierag.routing.query_router import QueryRoute

            mapping = {
                QueryIntent.VISUAL: QueryRoute.FACTUAL,
                QueryIntent.KNOWLEDGE: QueryRoute.FACTUAL,
                QueryIntent.MULTIMODAL: QueryRoute.REASONING,
                QueryIntent.CHAT: QueryRoute.FACTUAL,
                QueryIntent.MACRO_KNOWLEDGE: QueryRoute.FACTUAL,
                QueryIntent.DIALOG: QueryRoute.DIALOG,
                QueryIntent.TEMPORAL: QueryRoute.TEMPORAL,
                QueryIntent.NARRATIVE: QueryRoute.REASONING,
            }
            return mapping.get(intent, QueryRoute.FACTUAL)
        except ImportError:
            return None

    # ── Video Understanding: TEMPORAL & NARRATIVE handlers ───────────────

    def _handle_temporal_query(
        self,
        query: str,
        movie_id: str,
        identified_movie: Optional[str],
        identified_from: str,
        thoughts: List[str],
    ) -> Dict[str, Any]:
        """
        Handle TEMPORAL queries: 'When does X happen?', 'Find the scene where...'

        Uses TemporalGroundingEngine to resolve temporal expressions
        and localize events in the video timeline.
        """
        grounding_result = None
        temporal_grounder = self.temporal_grounder  # triggers lazy-load

        if temporal_grounder and movie_id:
            try:
                grounding_result = temporal_grounder.ground(
                    query=query,
                    movie_id=movie_id,
                    candidate_scenes=None,
                    k=10,
                )
                thoughts.append(
                    f"⏱️ Temporal Grounding: [{grounding_result.segment[0]:.1f}s - "
                    f"{grounding_result.segment[1]:.1f}s] (confidence: "
                    f"{grounding_result.confidence:.2f})"
                )
            except Exception as e:
                thoughts.append(f"⚠️ Temporal grounding failed: {e}")

        # Also run standard retrieval for supporting context
        scene_results = []
        if identified_movie:
            scene_results = self._scene_retriever(identified_movie, query, k=6)

        return {
            "temporal_grounding": grounding_result,
            "scene_results": scene_results,
            "segment": grounding_result.segment if grounding_result else (0.0, 0.0),
        }

    def _handle_narrative_query(
        self,
        query: str,
        movie_id: str,
        identified_movie: Optional[str],
        identified_from: str,
        thoughts: List[str],
    ) -> Dict[str, Any]:
        """
        Handle NARRATIVE queries: 'Why did X happen?', 'What caused Y?'

        Uses CausalReasoner to build causal chains and generate
        narrative explanations from the knowledge graph.
        """
        causal_answer = None
        causal_reasoner = self.causal_reasoner  # triggers lazy-load

        if causal_reasoner and movie_id:
            try:
                causal_answer = causal_reasoner.answer_why(
                    query=query,
                    movie_id=movie_id,
                    context_scenes=None,
                )
                thoughts.append(
                    f"🔗 Causal Reasoning: {len(causal_answer.supporting_evidence)} "
                    f"causal triples found (confidence: {causal_answer.confidence:.2f})"
                )
                if causal_answer.causal_chain:
                    thoughts.append(
                        f"   Chain depth: {causal_answer.causal_chain.depth}"
                    )
            except Exception as e:
                thoughts.append(f"⚠️ Causal reasoning failed: {e}")

        # Also retrieve supporting scene context
        scene_results = []
        if identified_movie:
            scene_results = self._scene_retriever(identified_movie, query, k=6)

        return {
            "causal_answer": causal_answer,
            "scene_results": scene_results,
            "causal_explanation": (
                causal_answer.causal_explanation if causal_answer else ""
            ),
            "supporting_evidence": (
                causal_answer.supporting_evidence if causal_answer else []
            ),
        }

    # ── Shortcut: TEMPORAL/NARRATIVE go straight to generation ────────

    def _generate_with_context(
        self,
        query: str,
        intent: QueryIntent,
        visual_results: list,
        knowledge_results: list,
        script_results: list,
        scene_results: list,
        thoughts: list,
        extra_context: str = "",
        temporal_grounding_result: Optional[Any] = None,
        causal_answer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer for TEMPORAL or NARRATIVE queries using pre-computed context.

        Skips the full swarm loop for speed — reasoning is already done by
        the temporal/causal handlers.
        """
        identified_movie = ""
        for r in (visual_results + knowledge_results):
            mid = getattr(r, "movie_id", "")
            if mid:
                identified_movie = mid
                break

        # Build scene results string
        scene_str = self._format_scene_results_for_prompt(scene_results, identified_movie)

        # Build context for generation
        context_parts = []
        if scene_str:
            context_parts.append(f"SCENE CLUSTERS:\n{scene_str}")
        if extra_context:
            context_parts.append(extra_context)
        full_context = "\n\n".join(context_parts)

        # Generate answer using LLM
        answer = self._generate_answer_text(query, full_context, intent, thoughts)

        # Collect keyframe paths
        keyframe_paths = []
        for r in (visual_results + knowledge_results)[:6]:
            meta = getattr(r, "metadata", {})
            if isinstance(meta, dict):
                img_path = meta.get("path", "")
                if img_path and os.path.exists(img_path):
                    shot = meta.get("shot_id", "frame")
                    keyframe_paths.append((
                        img_path,
                        f"{getattr(r, 'movie_id', 'Unknown')} | {shot}"
                    ))

        # Temporal grounding JSON
        temporal_json = None
        if temporal_grounding_result:
            tg = self.temporal_grounder
            if tg:
                temporal_json = tg.to_json(temporal_grounding_result)

        return {
            "answer": answer,
            "intent": intent.value if hasattr(intent, "value") else str(intent),
            "visual_results": visual_results,
            "knowledge_results": knowledge_results,
            "script_results": script_results,
            "scene_results": scene_results,
            "keyframe_paths": keyframe_paths,
            "temporal_grounding": temporal_json,
            "causal_explanation": (
                causal_answer.causal_explanation if causal_answer else ""
            ),
            "causal_chain": (
                {
                    "depth": causal_answer.causal_chain.depth,
                    "events": causal_answer.causal_chain.events,
                    "confidence": causal_answer.causal_chain.confidence,
                }
                if causal_answer and causal_answer.causal_chain
                else None
            ),
            "supporting_evidence": (
                causal_answer.supporting_evidence if causal_answer else []
            ),
            "thoughts": thoughts,
        }

    def _generate_answer_text(
        self,
        query: str,
        context: str,
        intent: QueryIntent,
        thoughts: list,
    ) -> str:
        """Generate answer text using LLM with given context."""
        if not context:
            return "I couldn't find relevant context to answer this query."

        if intent == QueryIntent.TEMPORAL:
            system_hint = (
                "You are answering a temporal grounding question. "
                "State the exact time segment where the event occurs."
            )
        elif intent == QueryIntent.NARRATIVE:
            system_hint = (
                "You are answering a causal/narrative question. "
                "Explain WHY the event happened, referencing the causal chain."
            )
        else:
            system_hint = "You are a film expert. Answer based on the provided evidence."

        prompt = f"""{system_hint}

Query: {query}

Evidence:
{context[:3000]}

Provide a clear, accurate answer based on the evidence above."""
        try:
            answer = self._llm_client.generate_content(
                model=None,
                contents=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            return answer.strip() if answer else "Limited information available."
        except Exception as e:
            thoughts.append(f"⚠️ Answer generation failed: {e}")
            return "I encountered an error generating the answer."

    # ─── Main Entry Point ────────────────────────────────────────────

    def respond(
        self,
        query: str,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Full agentic pipeline: Contextualize → Route → Retrieve → Grade → Generate.

        Returns:
            {
                "answer": str,
                "intent": str,
                "visual_results": list,
                "knowledge_results": list,
                "script_results": list,
                "scene_results": list,
                "keyframe_paths": list,
                "thoughts": list,  # Agent reasoning trace
            }
        """
        history = history or []
        thoughts = []

        # ─── Step 1: Contextualize (text-only, skip for multimodal) ───
        has_media = image_path is not None or video_path is not None
        contextualized = self.contextualize(query, history, has_media=has_media)
        if contextualized != query:
            thoughts.append(f"🔄 Query rewritten: '{contextualized}'")

        # ─── Step 2: Route Intent & Extract Explicit Movie ───
        intent, explicit_movie_name = self.route_intent_and_movie(
            contextualized, has_image=image_path is not None or video_path is not None
        )
        thoughts.append(f"🔀 Intent: **{intent.value}**")

        known_movie_mappings = {
            "titanic": "tt0120338",
            "godfather": "tt0068646",
            "bố già": "tt0068646",
            "watchmen": "tt1127180",
            "pulp fiction": "tt0110912",
            "forrest gump": "tt0109830",
            "inception": "tt1375666",
            "shutter island": "tt1130884",
            "interstellar": "tt0816692",
            "joker": "tt7286456",
            "batman": "tt0468569",
            "the dark knight": "tt0468569",
        }

        identified_movie = None
        if explicit_movie_name:
            movie_lower = explicit_movie_name.lower()
            for name, mid in known_movie_mappings.items():
                if name in movie_lower or movie_lower in name:
                    identified_movie = mid
                    thoughts.append(
                        f"🎯 Explicit Movie Identified: '{explicit_movie_name}' -> ID: {identified_movie}"
                    )
                    break

        # ─── Step 3: Video Understanding — TEMPORAL & NARRATIVE routing ───
        scene_results: list = []
        temporal_grounding_result: Optional[Any] = None
        causal_answer: Optional[Any] = None

        if intent in (QueryIntent.TEMPORAL, QueryIntent.NARRATIVE):
            # TEMPORAL: "When does X happen?" → Temporal Grounding
            if intent == QueryIntent.TEMPORAL:
                temporal_out = self._handle_temporal_query(
                    query=contextualized,
                    movie_id=identified_movie or "",
                    identified_movie=identified_movie,
                    identified_from="temporal_intent",
                    thoughts=thoughts,
                )
                temporal_grounding_result = temporal_out.get("temporal_grounding")
                scene_results = temporal_out.get("scene_results", [])
                thoughts.append(
                    f"⏱️ TEMPORAL query → [{temporal_out.get('segment', (0, 0))[0]:.1f}s – "
                    f"{temporal_out.get('segment', (0, 0))[1]:.1f}s]"
                )

            # NARRATIVE: "Why did X happen?" → Causal Reasoning
            elif intent == QueryIntent.NARRATIVE:
                narrative_out = self._handle_narrative_query(
                    query=contextualized,
                    movie_id=identified_movie or "",
                    identified_movie=identified_movie,
                    identified_from="narrative_intent",
                    thoughts=thoughts,
                )
                causal_answer = narrative_out.get("causal_answer")
                scene_results = narrative_out.get("scene_results", [])
                thoughts.append(
                    f"🔗 NARRATIVE query → {len(narrative_out.get('supporting_evidence', []))} "
                    f"causal triples"
                )

            # Skip swarm loop for pure temporal/narrative — go straight to generation
            # Build a minimal context and go to generation
            if intent == QueryIntent.TEMPORAL and temporal_grounding_result:
                extra_context = (
                    f"Temporal Grounding Result: The event occurs at "
                    f"[{temporal_grounding_result.segment[0]:.1f}s – "
                    f"{temporal_grounding_result.segment[1]:.1f}s] with "
                    f"confidence {temporal_grounding_result.confidence:.2f}.\n"
                    f"Reasoning: {temporal_grounding_result.reasoning_trace}\n"
                )
            elif intent == QueryIntent.NARRATIVE and causal_answer:
                extra_context = (
                    f"Causal Explanation: {causal_answer.causal_explanation}\n"
                    f"Supporting Evidence:\n" +
                    "\n".join(
                        f"  - {e['cause']} → {e['effect']}"
                        for e in causal_answer.supporting_evidence[:5]
                    ) +
                    f"\nReasoning Trace: {causal_answer.reasoning_trace}\n"
                )
            else:
                extra_context = ""
            # Jump to generation step with pre-computed context
            return self._generate_with_context(
                query=contextualized,
                intent=intent,
                visual_results=visual_results,
                knowledge_results=knowledge_results,
                script_results=script_results,
                scene_results=scene_results,
                thoughts=thoughts,
                extra_context=extra_context,
                temporal_grounding_result=temporal_grounding_result,
                causal_answer=causal_answer,
            )

        # ─── Step 4: Swarm Verification Loop (No Tools for Gemma) ───
        max_iterations = 3
        current_queries = [contextualized]
        rewrite_count = 0

        visual_results = []
        knowledge_results = []
        script_results = []

        # ─── Step 2.5: Pre-Loop Visual Processing & VLM Analysis ───
        vlm_analysis_text = ""
        movie_meta_cache = {}
        target_media_for_vlm = image_path

        if video_path:
            thoughts.append(f"🎥 Đang trích xuất khung hình từ video: {video_path}")
            try:
                import cv2
                import tempfile
                import os as _os

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                n_samples = min(8, max(1, total_frames))
                sample_positions = [
                    int(i * total_frames / n_samples) for i in range(n_samples)
                ]

                frame_results_all = []
                extracted_frame_paths = []  # Keep actual video frames for VLM
                movie_vote = {}  # majority voting: movie_id -> count

                for pos in sample_positions:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    tmp_f = tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False, dir=_os.environ.get("TEMP", "/tmp")
                    )
                    cv2.imwrite(tmp_f.name, frame)
                    tmp_f.close()
                    frame_matches = self.retrieve_visual_by_image(tmp_f.name, k=3)
                    for m in frame_matches:
                        mid = getattr(m, "movie_id", "")
                        movie_vote[mid] = movie_vote.get(mid, 0) + 1
                        frame_results_all.append(m)
                    extracted_frame_paths.append(tmp_f.name)  # Keep for VLM
                cap.release()

                # Keep the middle frame for VLM, clean up the rest
                vlm_frame_path = None
                for idx, fp in enumerate(extracted_frame_paths):
                    if idx == len(extracted_frame_paths) // 2:
                        vlm_frame_path = fp  # Keep middle frame
                    else:
                        try:
                            _os.unlink(fp)
                        except Exception:
                            pass

                if movie_vote:
                    identified_movie = max(movie_vote, key=movie_vote.get)
                    thoughts.append(
                        f"🎩 Video Majority Voting: phim được nhận diện là **{identified_movie}**"
                    )
                    for r in frame_results_all:
                        if getattr(r, "movie_id", "") == identified_movie:
                            _key = getattr(r, "id", str(r))
                            if _key not in [
                                getattr(x, "id", str(x)) for x in visual_results
                            ]:
                                visual_results.append(r)
                    if not contextualized or contextualized.strip() == "":
                        contextualized = f"Phim {identified_movie}"
                    current_queries = [f"{contextualized} {identified_movie}"]
                    # Use ACTUAL extracted video frame for VLM (not DB keyframe!)
                    if vlm_frame_path:
                        target_media_for_vlm = vlm_frame_path
                    elif frame_results_all:
                        # Fallback to DB keyframe only if no extracted frame
                        target_media_for_vlm = getattr(
                            frame_results_all[0],
                            "path",
                            getattr(frame_results_all[0].metadata, "path", None),
                        )
                else:
                    thoughts.append("⚠️ Không thể nhận diện phim từ video.")
            except Exception as e:
                thoughts.append(f"❌ Lỗi xử lý video: {e}")

        elif image_path:
            thoughts.append(f"👁️ Tìm kiếm Visual gốc cho ảnh tải lên...")
            vr = self.retrieve_visual_by_image(image_path, k=5)
            # Seed visual results
            for r in vr:
                _key = getattr(r, "id", str(r))
                if _key not in [getattr(x, "id", str(x)) for x in visual_results]:
                    visual_results.append(r)
            movie_vote = {}
            for r in vr:
                mid = getattr(r, "movie_id", "")
                if mid:
                    movie_vote[mid] = movie_vote.get(mid, 0) + 1
            if movie_vote:
                identified_movie = max(movie_vote, key=movie_vote.get)
                thoughts.append(
                    f"🎩 Image Majority Voting: phim được nhận diện là **{identified_movie}**"
                )

        # ── Step 2.6: Extract enriched metadata from FAISS matched shots ──
        matched_shots_context = ""
        if visual_results:
            lines = []
            for i, v in enumerate(visual_results[:8]):  # Top 8 matches
                meta = getattr(v, "metadata", {})
                mid = getattr(v, "movie_id", meta.get("movie_id", ""))
                shot_id = getattr(v, "id", meta.get("id", ""))
                score = getattr(v, "score", 0.0)
                start_t = meta.get("start_time", "")
                end_t = meta.get("end_time", "")
                desc = meta.get("description", "")
                dial = meta.get("dialogue_text", "")
                chars = meta.get("characters", [])
                situation = meta.get("situation", "")
                title = meta.get("title", mid)

                line = f"[{i + 1}] {mid}›{shot_id} · {score:.2f}"
                if title and title != mid:
                    line += f" | Title: {title}"
                if start_t:
                    line += f" | Time: {start_t}→{end_t}"
                if situation:
                    line += f" | Situation: {situation}"
                if desc:
                    line += f" | Desc: {desc[:120]}"
                if chars:
                    line += f" | Characters: {', '.join(chars[:5])}"
                if dial:
                    line += f" | Dialogue: {dial[:80]}"
                lines.append(line)

            matched_shots_context = "\n".join(lines)
            thoughts.append(
                f"📋 Trích xuất metadata trực tiếp từ {len(lines)} shots khớp nhất"
            )

        # ── Step 2.7: VLM Analysis — ONLY for image queries ──
        # Video: VLM only sees 1 frame → incomplete analysis. FAISS majority voting + metadata is enough.
        # Image: VLM sees the full uploaded image → useful for scene context + verification.
        if target_media_for_vlm and image_path and not video_path:
            try:
                import json as _json
                from pathlib import Path as _Path

                def _load_movie_meta_vlm(movie_id: str) -> dict:
                    _META_DIRS = [
                        _Path("data/movienet_subset/meta"),
                        _Path("data/unified_dataset/meta"),
                        _Path("../data/movienet_subset/meta"),
                        _Path("../data/unified_dataset/meta"),
                    ]
                    for d in _META_DIRS:
                        p = d / f"{movie_id}.json"
                        if p.exists():
                            try:
                                return _json.loads(p.read_text(encoding="utf-8"))
                            except Exception:
                                pass
                    return {}

                # Load meta for identified movie
                meta = {}
                title = identified_movie or ""
                if identified_movie:
                    meta = _load_movie_meta_vlm(identified_movie)
                    if meta:
                        movie_meta_cache[identified_movie] = meta
                        title = meta.get("title", identified_movie)

                thoughts.append(
                    "👁️ Gọi VLM (Vision Model) phân tích bối cảnh và xác minh shots..."
                )

                # VLM prompt: describe scene + verify against FAISS matches
                vlm_prompt = (
                    "Hãy quan sát thật kỹ bức ảnh/frame này và thực hiện 2 nhiệm vụ:\n\n"
                    "1. MÔ TẢ CHI TIẾT: Nhân vật (ngoại hình, trang phục, biểu cảm), "
                    "bối cảnh (nội/ngoại, ánh sáng, vật thể), hành động đang diễn ra.\n\n"
                )
                if matched_shots_context:
                    vlm_prompt += (
                        "2. XÁC MINH: Hệ thống FAISS đã tìm được các shots sau. "
                        "Dựa vào nội dung ảnh, hãy cho biết shot nào khớp nhất (hoặc KHÔNG khớp):\n"
                        f"{matched_shots_context}\n\n"
                        "Trả lời: [MÔ TẢ chi tiết], sau đó [SHOT PHÙ HỢP NHẤT: số thứ tự]"
                    )
                else:
                    vlm_prompt += (
                        "Mô tả cực kì chi tiết cấu trúc hạt nhân của bức ảnh "
                        "để làm Query tìm kiếm Visual Search."
                    )

                vlm_res = self._llm_client.generate_vision_content(
                    prompt=vlm_prompt, image_path=target_media_for_vlm
                )
                vlm_analysis_text = (
                    f"👁️ KẾT QUẢ VLM (Scene Context + Verification):\n{vlm_res}\n\n"
                )
                thoughts.append(f"✅ VLM Response: {vlm_res[:200]}...")

                # ── Check VLM-FAISS conflict: does VLM description match FAISS movie? ──
                vlm_conflict_warning = ""
                if vlm_res and identified_movie and matched_shots_context:
                    try:
                        conflict_prompt = (
                            f"VLM mô tả ảnh: {vlm_res[:300]}\n"
                            f"FAISS cho rằng đây là phim: {title} (ID: {identified_movie})\n\n"
                            f"Hỏi: Nội dung VLM mô tả có KHỚP với phim {title} không?\n"
                            f"Trả lời CHỈ MỘT từ: MATCH hoặc MISMATCH"
                        )
                        conflict_res = self._llm_client.models.generate_content(
                            model=self.model_id, contents=conflict_prompt
                        )
                        conflict_text = conflict_res.text.strip().upper()
                        if "MISMATCH" in conflict_text:
                            vlm_conflict_warning = (
                                f"⚠️ CẢNH BÁO: VLM phát hiện nội dung ảnh KHÔNG KHỚP với phim {title}. "
                                f"Ảnh có thể không nằm trong cơ sở dữ liệu. "
                                f"Hãy trả lời dựa trên mô tả VLM thay vì FAISS results.\n"
                            )
                            thoughts.append(
                                f"⚠️ VLM-FAISS CONFLICT: VLM mô tả không khớp với {title}! Ảnh có thể không trong DB."
                            )
                    except Exception:
                        pass  # Conflict check failed, proceed normally

                # ── Distill VLM output into clean search keywords ──
                if vlm_res and len(vlm_res.split()) > 1:
                    try:
                        distill_prompt = (
                            f"Từ mô tả VLM sau, trích xuất 5-10 từ khóa tìm kiếm hình ảnh (tiếng Anh, ngắn gọn, phân cách bằng dấu cách).\n"
                            f"KHÔNG giải thích, chỉ trả về từ khóa.\n\n"
                            f"VLM: {vlm_res[:500]}\n\nKeywords:"
                        )
                        distill_res = self._llm_client.models.generate_content(
                            model=self.model_id, contents=distill_prompt
                        )
                        keywords = distill_res.text.strip()[:150]
                    except Exception:
                        keywords = " ".join(vlm_res.split()[:10])

                    injected_query = f"{title} {keywords}"
                    thoughts.append(f"💉 Bơm VLM keywords: '{injected_query}'")
                    current_queries.append(injected_query)
                    if len(query.split()) <= 4 and intent == QueryIntent.MULTIMODAL:
                        query = injected_query

            except Exception as e:
                thoughts.append(f"⚠️ VLM Analysis failed: {e}")
                vlm_res = ""
        else:
            vlm_conflict_warning = ""
            vlm_res = ""

        # ── Step 2.8: LLM Context Booster (for Video / non-VLM cases) ──
        # Focus on generating keywords based on the video context or explicitly mentioned movie
        vlm_res = locals().get("vlm_res", "")
        if not vlm_res and (matched_shots_context or explicit_movie_name):
            try:
                thoughts.append(
                    "🧠 Gọi LLM Context Booster: Tổng hợp metadata và ngữ cảnh thành truy vấn nâng cao..."
                )
                boost_prompt = (
                    "Bạn là Query Booster. Dựa trên thông tin đầu vào (Tên phim hoặc Dữ liệu FAISS Video), "
                    "hãy trích xuất 2-4 TỪ KHÓA TÌM KIẾM (tiếng Anh) ngắn gọn, sắc bén để tìm kiếm trong CSDL RAG.\n"
                    "Trả về định dạng các cụm từ cách nhau bởi dấu `|`. KHÔNG giải thích thêm.\n"
                    "Ví dụ: rose titanic | jack drawing | door sinking\n\n"
                    f"Câu hỏi gốc: {query}\n"
                )
                if explicit_movie_name:
                    boost_prompt += (
                        f"Bối cảnh (Tên phim trực tiếp): {explicit_movie_name}\n"
                    )
                if matched_shots_context:
                    boost_prompt += f"Dữ liệu FAISS Video:\n{matched_shots_context}\n\n"

                boost_prompt += "Output:"
                boost_res = self._llm_client.models.generate_content(
                    model=self.model_id, contents=boost_prompt
                )
                boost_text = boost_res.text.strip()

                # Strip <think>...</think> reasoning tokens (Qwen3, etc.)
                import re as _re

                boost_text = _re.sub(r"<think>[\s\S]*?</think>", "", boost_text).strip()

                new_queries = [
                    q.strip() for q in boost_text.split("|") if len(q.strip()) > 5
                ]
                if new_queries:
                    current_queries.extend(new_queries)
                    thoughts.append(f"💉 Bơm Metadata Keywords: {new_queries}")
            except Exception as e:
                thoughts.append(f"⚠️ LLM Context Booster failed: {e}")

        if not vlm_res:
            vlm_conflict_warning = ""

        # For MULTIMODAL with image/video: FAISS visual = primary evidence, limit retries
        effective_max_iter = max_iterations
        if intent == QueryIntent.MULTIMODAL and (image_path or video_path):
            effective_max_iter = (
                1  # Visual matching IS the answer, no need to retry text
            )
            thoughts.append(
                "🎯 Multimodal: FAISS visual là evidence chính, giới hạn 1 vòng Verifier."
            )

        if intent == QueryIntent.CHAT:
            thoughts.append("💬 Chat mode — no retrieval needed")
        else:
            for iteration in range(effective_max_iter):
                lore_report = ""
                visual_report = ""
                script_report = ""

                # Deduplication dictionaries for this iteration
                k_results_dict = {
                    self._result_key(k): k for k in knowledge_results
                }
                v_results_dict = {getattr(v, "id", str(v)): v for v in visual_results}
                s_results_dict = {
                    getattr(s, "clip_id", getattr(s, "id", str(s))): s
                    for s in script_results
                }

                thoughts.append(
                    f"🔄 Vòng {iteration + 1}: Chạy tìm kiếm cho {len(current_queries)} truy vấn đồng thời..."
                )

                for q in current_queries:
                    # ─── Lore Execution (Static Python Calls) ───
                    if intent in (
                        QueryIntent.KNOWLEDGE,
                        QueryIntent.MULTIMODAL,
                        QueryIntent.MACRO_KNOWLEDGE,
                        QueryIntent.DIALOG,
                    ):
                        # Bỏ qua tìm text FAISS nếu có ảnh VÀ câu hỏi quá ngắn/chung chung (VD: "Ai đây?", "Đây là phim gì?")
                        skip_text_search = False
                        if image_path and len(q.split()) <= 4:
                            skip_text_search = True
                            thoughts.append(
                                f"⏩ Bỏ qua tìm text cho '{q}' vì có ảnh và câu hỏi chung chung."
                            )

                        if not skip_text_search:
                            thoughts.append(
                                f"📜 Gọi LoreAgent truy xuất Đồ thị và Kịch bản cho: '{q}'"
                            )
                            # If VLM injected a new query (i.e. not the main query), drop the strict movie filter so it can find correct movies
                            movie_filter = (
                                identified_movie if q == current_queries[0] else None
                            )
                            try:
                                kr = self.retrieve_knowledge(
                                    q, k=4, movie_id=movie_filter
                                )
                                for k in kr:
                                    k_results_dict[self._result_key(k)] = k
                                lore_report += (
                                    f"Knowledge Docs cho '{q}': {len(kr)} found.\n"
                                )
                            except Exception as e:
                                thoughts.append(f"⚠️ Lỗi Search Text FAISS: {e}")
                                logger.error(
                                    f"Knowledge search error: {e}", exc_info=True
                                )

                            try:
                                gr = self.retrieve_graph_context(
                                    q, k=3, movie_id=movie_filter
                                )
                                for g in gr:
                                    k_results_dict[self._result_key(g)] = g
                                lore_report += (
                                    f"Graph Hits cho '{q}': {len(gr)} found.\n"
                                )
                            except Exception as e:
                                thoughts.append(f"⚠️ Lỗi Search Graph: {e}")
                                logger.error(
                                    f"Graph search error: {e}", exc_info=True
                                )

                            try:
                                sr = self.retrieve_script_scenes(
                                    q, k=4, movie_id=movie_filter
                                )
                                for s in sr:
                                    s_results_dict[
                                        getattr(s, "clip_id", getattr(s, "id", str(s)))
                                    ] = s
                                script_report += (
                                    f"Script Sub-scenes cho '{q}': {len(sr)} found.\n"
                                )
                            except Exception as e:
                                thoughts.append(f"⚠️ Lỗi Search Script Scenes: {e}")
                                logger.error(
                                    f"Script scene search error: {e}", exc_info=True
                                )

                        dia_idx = getattr(self, "dialogue_indexer", None)
                        if dia_idx:
                            movie_filter = (
                                identified_movie if q == current_queries[0] else None
                            )
                            try:
                                dr = dia_idx.search(q, k=2, movie_id=movie_filter)
                            except TypeError:
                                # Fallback if dialogue_indexer hasn't fully reloaded the new signature in worker thread
                                dr = dia_idx.search(q, k=2)
                            except Exception as e:
                                logger.error(f"Dialogue search error: {e}")
                                dr = []

                            if dr:
                                lore_report += f"Dialogue Results cho '{q}':\n"
                                for i, r in enumerate(dr):
                                    lore_report += f"[{i + 1}] {r.get('movie_id', '')} - {r.get('start_time', 0)}: '{r.get('text', '')}'\n"

                    # ─── Visual Execution (Static Python Calls) ───
                    if intent in (QueryIntent.VISUAL, QueryIntent.MULTIMODAL):
                        thoughts.append(
                            f"👁️ Gọi VisualAgent tìm kiếm Database Hình ảnh cho: '{q}'"
                        )
                        # If this is the original query and we have an image, search by image.
                        # If the VLM expanded/corrected the query, use CLIP text-to-image search over the new semantic context!
                        movie_filter = identified_movie if identified_movie else None
                        if image_path and q == current_queries[0]:
                            vr = self.retrieve_visual_by_image(image_path)
                        else:
                            vr = self.retrieve_visual(q, k=4, movie_id=movie_filter)
                        for v in vr:
                            v_results_dict[getattr(v, "id", str(v))] = v
                        visual_report += (
                            f"Visual Docs cho '{q}': {len(vr)} frames found.\n"
                        )

                # Update the main results list with the deduplicated values
                knowledge_results = self._sort_results_by_score(
                    list(k_results_dict.values())
                )
                visual_results = self._sort_results_by_score(
                    list(v_results_dict.values())
                )
                script_results = self._sort_results_by_score(
                    list(s_results_dict.values())
                )

                thoughts.append(
                    f"📦 Tổng hợp vòng {iteration + 1}: {len(knowledge_results)} văn bản, {len(visual_results)} khung hình, {len(script_results)} script sub-scenes."
                )

                # ─── Verification Agent (Gemma Mode - NO TOOLS/no system_instruction) ───
                thoughts.append(
                    f"🔎 Verifier đang kiểm chứng dữ liệu (Vòng {iteration + 1}/{max_iterations})..."
                )
                verify_prompt = (
                    "System: Bạn là Verifier Agent. Nhiệm vụ của bạn là đánh giá xem Báo cáo Lore và Báo cáo Visual có ĐỦ thông tin để trả lời Câu hỏi gốc hay không.\n"
                    "Nếu ĐỦ, trả lời chính xác từ: 'SUFFICIENT'.\n"
                    "Nếu THIẾU thông tin cốt lõi, hãy MỞ RỘNG VÀ TÁCH câu hỏi thành 2-3 TỪ KHÓA/CÂU TRUY VẤN MỚI, HOÀN TOÀN KHÁC NHAU để hệ thống tìm kiếm đa chiều.\n"
                    "Trả lời theo cú pháp: 'INSUFFICIENT: [Query 1] | [Query 2]'. Ví dụ: 'INSUFFICIENT: rose titanic | door sinking scene'. KHÔNG CẦN GIẢI THÍCH GÌ THÊM.\n\n"
                    f"Câu hỏi gốc: {query}\n\nBáo cáo Lore:\n{lore_report}\n\nBáo cáo Visual:\n{visual_report}\n\nBáo cáo Script:\n{script_report}\n\n"
                    "User: Đánh giá của bạn là gì?"
                )

                try:
                    ver_res = self._llm_client.models.generate_content(
                        model=self.model_id,
                        contents=verify_prompt,
                    )
                    ver_text = ver_res.text.strip()
                    if (
                        "SUFFICIENT" in ver_text.upper()
                        and "INSUFFICIENT" not in ver_text.upper()
                    ):
                        thoughts.append("✅ Verifier: ĐÃ ĐỦ DỮ LIỆU ĐỂ TRẢ LỜI.")
                        break
                    else:
                        parts = ver_text.split("INSUFFICIENT:")
                        if len(parts) > 1 and iteration < effective_max_iter - 1:
                            new_queries_raw = parts[1].strip()
                            current_queries = [
                                q.strip()
                                for q in new_queries_raw.split("|")
                                if q.strip()
                            ]
                            if current_queries:
                                thoughts.append(
                                    f"🔄 Verifier yêu cầu tìm lại đa luồng với các truy vấn: {current_queries}"
                                )
                                rewrite_count += 1
                            else:
                                thoughts.append(
                                    "⚠️ Verifier: Parsing mảng query thất bại, dừng vòng lặp."
                                )
                                break
                        else:
                            thoughts.append(
                                "⚠️ Verifier: Không đủ dữ liệu nhưng đã hết lượt tìm."
                            )
                            break
                except Exception as e:
                    thoughts.append(f"⚠️ Lỗi Verifier: {e}. Bỏ qua kiểm chứng.")
                    break

        # ─── Step 4: Tool-Calling JudgeAgent (Kimi K2 via Groq) ───
        thoughts.append(
            "⚖️ JudgeAgent (Kimi K2 + tools) đang tổng hợp và tự gọi công cụ..."
        )

        scene_results = self._build_scene_results(visual_results, script_results, limit=6)

        # ── Pre-compute temporal_grounding from best visual/scene match ──
        best_start_time = ""
        best_end_time = ""
        if visual_results:
            # Pick the highest-scoring match belonging to identified_movie
            for v in sorted(
                visual_results, key=self._result_sort_score, reverse=True
            ):
                v_meta = getattr(v, "metadata", {})
                v_mid = getattr(v, "movie_id", v_meta.get("movie_id", ""))
                st = v_meta.get("start_time", "")
                et = v_meta.get("end_time", "")
                if st and et:
                    if not identified_movie or v_mid == identified_movie:
                        best_start_time = st
                        best_end_time = et
                        break
        if (not best_start_time or not best_end_time) and scene_results:
            top_scene = scene_results[0]
            best_start_time = str(top_scene.get("start_time", "") or "")
            best_end_time = str(top_scene.get("end_time", "") or "")

        # Build temporal grounding line for JudgeAgent
        temporal_hint = ""
        if best_start_time and best_end_time:
            temporal_hint = (
                f"\n⏱️ TEMPORAL GROUNDING (từ FAISS metadata — CHÍNH XÁC, dùng nguyên giá trị):\n"
                f'```json\n{{\n  "temporal_grounding": {{\n    "start_time": "{best_start_time}",\n    "end_time": "{best_end_time}"\n  }}\n}}\n```\n'
            )

        # ── Build evidence context for JudgeAgent ──
        lore_context = "\n".join(
            [
                f"[{i + 1}] {getattr(k, 'text', str(k))[:300]}"
                for i, k in enumerate(knowledge_results)
            ]
        )
        scene_context = self._format_scene_results_for_prompt(scene_results, limit=4)
        script_context = "\n".join(
            [
                (
                    f"[{i + 1}] {getattr(s, 'metadata', {}).get('script_heading', '')} | "
                    f"{getattr(s, 'metadata', {}).get('script_location', '')} | "
                    f"{getattr(s, 'metadata', {}).get('start_time', '')}->{getattr(s, 'metadata', {}).get('end_time', '')}\n"
                    f"{getattr(s, 'metadata', {}).get('dialogue_excerpt', '')[:220]}"
                )
                for i, s in enumerate(script_results)
            ]
        )

        # Load movie metadata for high-confidence visual matches
        import json as _json
        from pathlib import Path as _Path
        from preprocess_data.config import PreprocessConfig as _PreCfg

        _META_DIRS = [_PreCfg.get_meta_dir(), *getattr(_PreCfg, "META_SEARCH_DIRS", [])]
        _META_DIRS = list(dict.fromkeys(_META_DIRS))

        _CHUNK_DIR = _PreCfg.get_temporal_chunks_dir()

        def _load_movie_meta(movie_id: str) -> dict:
            for d in _META_DIRS:
                p = d / f"{movie_id}.json"
                if p.exists():
                    try:
                        return _json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        pass
            return {}

        def _load_temporal_chunk(
            movie_id: str, shot_id: int = None, chunk_id: str = ""
        ) -> dict:
            if not movie_id:
                return {}
            chunk_file = _CHUNK_DIR / f"{movie_id}_chunks.json"
            if not chunk_file.exists():
                return {}
            try:
                chunks = _json.loads(chunk_file.read_text(encoding="utf-8"))
                if chunk_id:
                    for chunk in chunks:
                        if chunk.get("chunk_id") == chunk_id:
                            return chunk
                if shot_id is None:
                    return {}
                for chunk in chunks:
                    if (
                        chunk.get("shot_start", 0)
                        <= shot_id
                        <= chunk.get("shot_end", float("inf"))
                    ):
                        return chunk
            except Exception as e:
                thoughts.append(f"⚠️ Failed to parse {chunk_file.name}: {e}")
            return {}

        # Build enriched visual context with movie metadata and full 5-layer chunks
        visual_context_lines = []
        for i, v in enumerate(visual_results):
            mid = getattr(v, "movie_id", "")
            meta = getattr(v, "metadata", {})

            # Extract basic FAISS info
            shot_str = meta.get("shot_id", "")
            chunk_id = meta.get("chunk_id", "")

            # Attempt to parse shot ID to fetch full chunk data
            shot_id_num = None
            import re

            if shot_str:
                shot_match = re.search(r"shot[_-]?(\d+)", str(shot_str), re.IGNORECASE)
                if shot_match:
                    shot_id_num = int(shot_match.group(1))
                elif isinstance(shot_str, int):
                    shot_id_num = shot_str
                elif str(shot_str).isdigit():
                    shot_id_num = int(shot_str)

            # Check if we have the rich chunk info from temporal_chunks
            chunk_data = {}
            if mid and (shot_id_num is not None or chunk_id):
                chunk_data = _load_temporal_chunk(
                    mid, shot_id=shot_id_num, chunk_id=chunk_id
                )

            # If we found chunk data, OVERRIDE the FAISS metadata with the rich 5-layer chunk
            if chunk_data:
                time_str = ""
                start_time = chunk_data.get("start_time", "")
                end_time = chunk_data.get("end_time", "")

                chunk_desc = chunk_data.get("description", "")
                chunk_characters = chunk_data.get("characters", [])
                chunk_dialogue = chunk_data.get("dialogue_text", "")
                chunk_situation = chunk_data.get("situation", "")
                chunk_cast = chunk_data.get("cast_in_scene", [])
                chunk_script_heading = chunk_data.get("script_primary_heading", "")
                chunk_script_location = chunk_data.get("script_location", "")
                chunk_script_chars = chunk_data.get("script_characters", [])
                timestamp_source = chunk_data.get(
                    "timestamp_source", "annotation_frame"
                )
            else:
                # Fallback to FAISS pure metadata if chunks are missing (unlikely if built correctly)
                time_str = meta.get("time", "")
                start_time = meta.get("start_time", "")
                end_time = meta.get("end_time", "")

                chunk_desc = meta.get("description", "")
                chunk_characters = meta.get("characters", [])
                chunk_dialogue = meta.get("dialogue_text", "")
                chunk_situation = meta.get("situation", "")
                chunk_cast = meta.get("cast_in_scene", [])
                chunk_script_heading = meta.get("script_primary_heading", "")
                chunk_script_location = meta.get("script_location", "")
                chunk_script_chars = meta.get("script_characters", [])
                timestamp_source = meta.get("timestamp_source", "")

            # Fallback to scene_context if direct time is missing from FAISS and Chunk
            if not start_time and not end_time and not time_str:

                def _fmt(sec):
                    h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
                    return f"{h:02d}:{m:02d}:{s:02d}"

                scene_ctx = meta.get("scene_context", {})
                t_start = scene_ctx.get("scene_timestamp", 0)
                t_end = scene_ctx.get("scene_timestamp_end", 0)

                if t_start or t_end:
                    start_time = _fmt(t_start)
                    t_end = t_end if t_end else t_start + 15
                    end_time = _fmt(t_end)
                elif shot_str:
                    import re

                    shot_match = re.search(
                        r"shot[_-]?(\d+)", str(shot_str), re.IGNORECASE
                    )
                    if shot_match:
                        shot_num = int(shot_match.group(1))
                        t_start = shot_num * 3
                        t_end = t_start + 15
                        start_time = _fmt(t_start)
                        end_time = _fmt(t_end)

            # Format time display
            if start_time and end_time:
                ts_label = (
                    "Exact" if timestamp_source == "annotation_frame" else "Approx"
                )
                time_disp = f"{start_time} → {end_time} [{ts_label}]"
            elif time_str:
                time_disp = time_str
            else:
                time_disp = "N/A"

            score = getattr(v, "score", 0.0)
            title_hint = meta.get("title", mid)
            line = f"┌─────────────────────────────────────────────────────────────┐\n"
            line += f"│  Temporal Chunk (shot_id: {shot_str})                       │\n"
            line += f"│                                                             │\n"
            line += (
                f"│  📍 Layer 1 — Temporal Anchor:                               │\n"
            )
            line += (
                f"│     movie: {title_hint} [{mid}]                              │\n"
            )
            line += f"│     time: {time_disp}                                      │\n"
            line += f"│                                                             │\n"

            if chunk_situation or chunk_desc:
                line += f"│  📝 Layer 2 — Semantic Description:                          │\n"
                if chunk_situation:
                    line += f"│     situation: {chunk_situation}                               │\n"
                if chunk_desc:
                    line += f"│     description: {chunk_desc[:150]}...                         │\n"
                line += (
                    f"│                                                             │\n"
                )

            if chunk_dialogue:
                line += (
                    f"│  🗣️ Layer 3 — Dialogue (from SRT):                           │\n"
                )
                line += f'│     dialogue_text: "{chunk_dialogue[:100]}..."               │\n'
                line += (
                    f"│                                                             │\n"
                )

            if chunk_characters or chunk_cast:
                line += f"│  🎬 Layer 4 — Movie Metadata:                                │\n"
                if chunk_characters:
                    line += f"│     characters: {', '.join(chunk_characters[:5])}              │\n"
                if chunk_cast:
                    cast_str = ", ".join(
                        f"{c['actor']} → {c['character']}" for c in chunk_cast[:3]
                    )
                    line += f"│     cast_in_scene: {cast_str}                               │\n"
                line += (
                    f"│                                                             │\n"
                )

            if chunk_script_heading or chunk_script_location or chunk_script_chars:
                line += f"│  📜 Layer 5 — Script Focus:                                  │\n"
                if chunk_script_heading:
                    line += f"│     heading: {chunk_script_heading[:45]}                         │\n"
                if chunk_script_location:
                    line += f"│     location: {chunk_script_location[:45]}                       │\n"
                if chunk_script_chars:
                    line += f"│     script_chars: {', '.join(chunk_script_chars[:4])}            │\n"
                line += f"│                                                             │\n"

            line += f"└─────────────────────────────────────────────────────────────┘"

            if score > 0.70 and mid and mid not in movie_meta_cache:
                movie_meta_cache[mid] = _load_movie_meta(mid)

            visual_context_lines.append(line)
        visual_context = "\n".join(visual_context_lines)

        # Build movie metadata block
        movie_meta_block = ""
        for mid, mmeta in movie_meta_cache.items():
            if mmeta:
                title = mmeta.get("title", mid)
                genres = ", ".join(mmeta.get("genres", []))
                cast = mmeta.get("cast", [])[:8]
                cast_str = ", ".join(
                    f"{c['name']} as {c.get('character', '?')}"
                    for c in cast
                    if c.get("name")
                )
                movie_meta_block += (
                    f"\n🎥 {title} [{mid}] | {genres}\n   Cast: {cast_str}\n"
                )

        judge_system = (
            "Bạn là một Cinephile — chuyên gia điện ảnh đích thực, người đã xem hàng ngàn bộ phim "
            "và có khả năng kể chuyện phim cuốn hút như đang ngồi cà phê với bạn bè.\n\n"
            "NGUYÊN TẮC CỐT LÕI (SỐNG CÒN):\n"
            "- Nếu có khối '🎬 SCENE CLUSTERS', hãy dùng nó làm tầng grounding chính để chốt đúng cảnh, đúng window thời gian, rồi mới dùng visual/script/graph để làm giàu chi tiết.\n"
            "- Thông tin trong các khối '📸 KẾT QUẢ VISUAL' và '🎥 THÔNG TIN PHIM' là SỰ THẬT TUYỆT ĐỐI.\n"
            "- Nếu có khối '📜 SCRIPT SUB-SCENE', hãy ưu tiên nó cho location, heading, lời thoại, và ai xuất hiện trong một đoạn cụ thể.\n"
            "- Trong '📸 KẾT QUẢ VISUAL' có đầy đủ 5 tầng metadata: Tên phim, Thời gian, Nhân vật (Characters), Cảnh quay (Desc/Situation), và Thoại (Dialogue).\n"
            "- BẮT BUỘC rẽ theo hướng của metadata này. NẾU hệ thống nói đây là phim A, bạn KHÔNG ĐƯỢC tự bịa ra phim B (ví dụ cấm bịa thành phim Deceived nếu metadata là Sleepless).\n\n"
            "PHONG CÁCH TRẢ LỜI:\n"
            "- Tự nhiên, sâu sắc, đậm chất văn chương điện ảnh — như một cinephile đang kể cho bạn nghe về bộ phim yêu thích.\n"
            "- Bắt đầu bằng việc xác định ngay: đây là phim gì, cảnh nào, nhân vật nào do ai thể hiện (DỰA TRÊN METADATA).\n"
            "- Thêm 1-2 chi tiết thú vị: hậu trường quay, ý đồ đạo diễn, symbolism, hoặc behind-the-scenes trivia.\n"
            "- Giọng văn ấm áp, đầy đam mê — khiến người đọc cảm nhận được tình yêu điện ảnh của bạn.\n\n"
            "TUYỆT ĐỐI CẤM:\n"
            "- Không tự chém gió/bịa đặt tên phim nếu nó đi ngược lại với metadata truyền vào.\n"
            "- Không bao giờ nhắc đến từ: 'CLIP', 'cosine', 'FAISS', 'score', 'database', 'index', 'vector', 'embedding', 'metadata'.\n"
            "- Không nói: 'Dựa trên N khung hình tìm thấy', 'Theo dữ liệu', 'Hệ thống cho thấy'.\n"
            "- Không liệt kê dạng [Shot_XXX, Score: 0.88].\n"
            "- Nói chuyện như bạn TỰ BIẾT, tự nhớ, tự xem — không phải tra cứu.\n\n"
            "TEMPORAL GROUNDING (BẮT BUỘC):\n"
            "- Ở cuối câu trả lời, bạn PHẢI copy nguyên khối JSON temporal_grounding mà hệ thống đã cung cấp sẵn trong mục ⏱️.\n"
            "- COPY NGUYÊN, KHÔNG ĐƯỢC SỬA GIÁ TRỊ, KHÔNG ĐƯỢC TỰ BỊA TIMESTAMP MỚI.\n"
            "- Nếu không có mục ⏱️ thì bỏ qua, không cần tạo JSON.\n\n"
            "CÔNG CỤ (TOOLS):\n"
            "- Nếu dữ liệu ban đầu chưa đủ để kể chuyện tự tin, hãy gọi tools: search_knowledge, search_visual, search_script_scenes, search_dialogue, query_graph.\n"
            "- Gọi tool khi cần thêm thông tin về cast, cốt truyện, hoặc cảnh liên quan.\n"
        )

        # ── Build user-facing evidence block ──
        judge_user_content = ""

        if movie_meta_block:
            judge_user_content += f"🎥 THÔNG TIN PHIM:\n{movie_meta_block}\n\n"

        if vlm_analysis_text:
            judge_user_content += f"{vlm_analysis_text}\n"

        if vlm_conflict_warning:
            judge_user_content += f"{vlm_conflict_warning}\n"

        if scene_context:
            judge_user_content += (
                "🎬 SCENE CLUSTERS (semantic scene + script scene + clip fused grounding):\n"
                f"{scene_context}\n\n"
            )

        if visual_context:
            judge_user_content += (
                f"📸 KẾT QUẢ VISUAL (các cảnh khớp nhất):\n{visual_context}\n\n"
            )

        if lore_context:
            judge_user_content += f"📚 KỊCH BẢN & ĐỒ THỊ TRI THỨC:\n{lore_context}\n\n"

        if script_context:
            judge_user_content += f"📜 SCRIPT SUB-SCENE:\n{script_context}\n\n"

        if temporal_hint:
            judge_user_content += f"{temporal_hint}\n"

        judge_user_content += (
            f"❓ CÂU HỎI CỦA NGƯỜI DÙNG: {query}\n\n"
            "Hãy tổng hợp toàn bộ thông tin trên thành câu trả lời tự nhiên, cuốn hút. "
            "Nhớ copy nguyên khối JSON ⏱️ temporal_grounding vào cuối nếu có."
        )

        judge_prompt = f"System: {judge_system}\n\nUser: {judge_user_content}"

        # Build a simple tool executor that calls our Python retrieval methods
        def _tool_executor(tool_name: str, tool_args: dict) -> str:
            k_arg = tool_args.get("k", 5)
            q_arg = tool_args.get("query", query)
            if tool_name == "search_knowledge":
                results = self.retrieve_knowledge(q_arg, k=k_arg)
                thoughts.append(
                    f"🔧 Kimi called search_knowledge('{q_arg}') → {len(results)} docs"
                )
                return "\n".join(
                    [
                        f"[{i + 1}] {getattr(r, 'text', str(r))[:300]}"
                        for i, r in enumerate(results)
                    ]
                )
            elif tool_name == "search_visual":
                results = self.retrieve_visual(
                    q_arg, k=k_arg, movie_id=identified_movie
                )
                thoughts.append(
                    f"🔧 Kimi called search_visual('{q_arg}') → {len(results)} frames"
                )
                # also extend visual_results for keyframe display
                v_dict = {getattr(v, "id", str(v)): v for v in visual_results}
                for r in results:
                    v_dict[getattr(r, "id", str(r))] = r
                visual_results[:] = self._sort_results_by_score(list(v_dict.values()))
                return "\n".join(
                    [
                        f"[{i + 1}] Movie:{getattr(r, 'movie_id', '')} Shot:{getattr(r, 'metadata', {}).get('shot_id', '')} Score:{getattr(r, 'score', 0):.2f}"
                        for i, r in enumerate(results)
                    ]
                )
            elif tool_name == "search_script_scenes":
                results = self.retrieve_script_scenes(q_arg, k=k_arg)
                thoughts.append(
                    f"🔧 Kimi called search_script_scenes('{q_arg}') → {len(results)} hits"
                )
                s_dict = {
                    getattr(s, "clip_id", getattr(s, "id", str(s))): s
                    for s in script_results
                }
                for result in results:
                    s_dict[getattr(result, "clip_id", getattr(result, "id", str(result)))] = result
                script_results[:] = self._sort_results_by_score(list(s_dict.values()))
                return "\n".join(
                    [
                        f"[{i + 1}] {getattr(r, 'metadata', {}).get('script_heading', '')} | {getattr(r, 'metadata', {}).get('script_location', '')} | {getattr(r, 'metadata', {}).get('start_time', '')}->{getattr(r, 'metadata', {}).get('end_time', '')}"
                        for i, r in enumerate(results)
                    ]
                )
            elif tool_name == "search_dialogue":
                di = getattr(self, "dialogue_indexer", None)
                if di:
                    results = di.search(q_arg, k=k_arg)
                    thoughts.append(
                        f"🔧 Kimi called search_dialogue('{q_arg}') → {len(results)} lines"
                    )
                    return "\n".join(
                        [
                            f"[{i + 1}] {r.get('text', '')[:200]}"
                            for i, r in enumerate(results)
                        ]
                    )
                return "[dialogue index not available]"
            elif tool_name == "query_graph":
                if hasattr(self, "query_graph"):
                    graph_query = tool_args.get("query", q_arg)
                    thoughts.append(f"🔧 Kimi called query_graph('{graph_query}')")
                    graph_hits = self.query_graph(
                        graph_query, movie_id=identified_movie, limit=k_arg
                    )
                    graph_store = getattr(self, "_graph_store", None)
                    if graph_store:
                        return graph_store.format_hits(graph_hits)
                    return str(graph_hits)
                return "[Graph Database not connected]"
            return f"[Unknown tool: {tool_name}]"

        try:
            judge_res = self._llm_client.generate_with_tools(
                prompt=judge_prompt,
                tool_executor=_tool_executor,
                max_tool_rounds=5,
            )
            answer = judge_res.text.strip()
        except Exception as e:
            thoughts.append(
                f"⚠️ Tool-calling JudgeAgent failed: {e}. Trying plain generate."
            )
            try:
                plain_res = self._llm_client.models.generate_content(
                    model="kimi", contents=judge_prompt
                )
                answer = plain_res.text.strip()
            except Exception as e2:
                answer = f"Lỗi JudgeAgent: {e2}. Vui lòng thử lại sau."
                thoughts.append(f"❌ Lỗi JudgeAgent: {e2}")

        # ─── Step 5: Collect keyframe paths ───
        keyframe_paths = []
        for r in visual_results[:6]:
            img_path = getattr(r, "path", "")
            meta_dict = getattr(r, "metadata", {})
            if not isinstance(meta_dict, dict):
                meta_dict = {}

            if not img_path:
                img_path = meta_dict.get("path", "")

            if img_path and os.path.exists(img_path):
                shot = meta_dict.get("shot_id", "frame")
                caption = f"{getattr(r, 'movie_id', 'Unknown')} | {shot} | {getattr(r, 'score', 0.0):.2f}"
                keyframe_paths.append((img_path, caption))

        return {
            "answer": answer,
            "intent": intent.value if hasattr(intent, "value") else str(intent),
            "visual_results": visual_results,
            "knowledge_results": knowledge_results,
            "script_results": script_results,
            "scene_results": scene_results,
            "keyframe_paths": keyframe_paths,
            "temporal_grounding": {
                "start_time": best_start_time,
                "end_time": best_end_time,
            }
            if best_start_time and best_end_time
            else None,
            "causal_explanation": "",  # populated for NARRATIVE intent only
            "supporting_evidence": [],   # populated for NARRATIVE intent only
            "thoughts": thoughts,
        }
