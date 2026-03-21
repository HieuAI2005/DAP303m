"""
LLM Generator Module for MovieRAG.

Handles formatting retrieved context and calling the Google GenAI API
to generate natural language answers seamlessly.
"""

import os
import logging
from typing import List, Dict, Optional, Any

try:
    from google import genai

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from movierag.indexing.knowledge_indexer import TextSearchResult
from movierag.indexing.visual_indexer import SearchResult as VisualSearchResult

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Generates answers using Google GenAI API based on retrieved context.
    """

    def __init__(
        self,
        model_id: str = os.getenv(
            "MOVIERAG_RUNTIME_LLM_MODEL",
            os.getenv("MOVIERAG_LLM_MODEL", "moonshotai/kimi-k2-instruct"),
        ),
        api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM Generator using Google GenAI.

        Args:
            model_id: The model ID to use (e.g., 'gemini-2.5-flash', 'gemma-2-27b-it').
            api_key: Google Gemini API token. If None, looks for GEMINI_API_KEY.
        """
        self.model_id = model_id
        self.client = None

        try:
            from movierag.generation.universal_client import UniversalLLMClient

            self.client = UniversalLLMClient(model_id=self.model_id)
            logger.info(f"Initialized LLM Generator with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize UniversalLLMClient: {e}")

    @staticmethod
    def _compact_items(values: Any, limit: int = 4) -> str:
        """Join a list-like field into a short prompt-friendly string."""
        if not values:
            return ""
        if isinstance(values, str):
            return values.strip()
        compact = [str(v).strip() for v in values if str(v).strip()]
        return ", ".join(compact[:limit])

    def _format_scene_cluster_evidence(
        self, scene_results: List[Dict[str, Any]], limit: int = 4
    ) -> str:
        """Format fused scene clusters so the LLM sees scene-level grounding first."""
        if not scene_results:
            return ""

        lines = ["--- SCENE CLUSTER EVIDENCE (semantic scene + screenplay + clip fused grounding) ---"]
        for i, scene in enumerate(scene_results[:limit]):
            heading = (
                str(scene.get("heading", "") or "").strip()
                or str(scene.get("scene_label", "") or "").strip()
                or "Scene cluster"
            )
            movie = str(scene.get("movie_id", "") or "unknown")
            location = str(scene.get("location", "") or "").strip()
            start_time = str(scene.get("start_time", "") or "")
            end_time = str(scene.get("end_time", "") or "")
            cluster_type = str(scene.get("cluster_type", "") or "")
            score = float(scene.get("score", 0.0) or 0.0)
            visual_score = float(scene.get("best_visual_score", 0.0) or 0.0)
            script_score = float(scene.get("best_script_score", 0.0) or 0.0)
            evidence_count = int(scene.get("evidence_count", 0) or 0)
            lines.append(
                f"[Scene Cluster {i + 1} | Phim: {movie} | Heading: {heading} | "
                f"Window: {start_time}->{end_time} | Type: {cluster_type} | "
                f"Score: {score:.3f} | Visual: {visual_score:.3f} | Script: {script_score:.3f} | "
                f"Evidence: {evidence_count}]"
            )
            if location:
                lines.append(f"Location: {location}")

            script_subscenes = scene.get("script_subscenes", []) or []
            script_heads = self._compact_items(
                [item.get("heading", "") for item in script_subscenes], limit=3
            )
            if script_heads:
                lines.append(f"Script windows: {script_heads}")

            visual_cues = self._compact_items(scene.get("visual_cues", []) or [], limit=3)
            if visual_cues:
                lines.append(f"Visual cues: {visual_cues}")

            characters = self._compact_items(scene.get("characters", []) or [], limit=6)
            if characters:
                lines.append(f"Characters: {characters}")

            script_characters = self._compact_items(
                scene.get("script_characters", []) or [], limit=6
            )
            if script_characters:
                lines.append(f"Script characters: {script_characters}")

            semantic_ids = self._compact_items(
                scene.get("semantic_scene_ids", []) or [], limit=3
            )
            if semantic_ids:
                lines.append(f"Semantic scene ids: {semantic_ids}")

            script_scene_uids = self._compact_items(
                scene.get("script_scene_uids", []) or [], limit=3
            )
            if script_scene_uids:
                lines.append(f"Script scene ids: {script_scene_uids}")

            representative_frame = str(scene.get("representative_frame", "") or "")
            if representative_frame:
                lines.append(
                    f"Representative frame: {os.path.basename(representative_frame)}"
                )
            lines.append("")

        return "\n".join(lines) + "\n"

    def format_prompt(
        self,
        query: str,
        context_results: List[Any],
        visual_results: List[Any],
        script_results: List[Any],
        scene_results: List[Dict[str, Any]],
        history: List[Dict],
        route: Optional[Any] = None,
    ) -> str:
        """
        Format prompt dynamically based on retrieved context and intent route.
        """
        # Intent handling (supports both old QueryRoute and new QueryIntent)
        intent_value = (
            route.value
            if route and hasattr(route, "value")
            else str(route)
            if route
            else "MULTIMODAL"
        )

        base_prompt = (
            "Bạn là MovieRAG, trợ lý chuyên sâu về phim ảnh có năng lực trích xuất video.\n"
            "Hệ thống truy xuất của chúng tôi ĐÃ TÌM THẤY các thông tin sau từ Database.\n"
            "Nhiệm vụ của bạn là TỔNG HỢP các thông tin này để trả lời câu hỏi.\n"
            "BẮT BUỘC trích dẫn nguồn bằng cách thêm số `[1]`, `[2]` vào cuối câu nếu dùng thông tin đó.\n"
            "Nếu thông tin được cung cấp bên dưới không đủ để trả lời, hãy nói rõ, TUYỆT ĐỐI KHÔNG TỰ BỊA ĐẶT.\n\n"
            "⏰ **Yêu cầu BẮT BUỘC về Temporal Grounding (Khoanh vùng thời gian)**:\n"
            "Nếu câu trả lời của bạn có mô tả một cảnh phim cụ thể được lấy từ 'Visual Evidence', bạn PHẢI in ra một block JSON duy nhất ở cuối câu trả lời chứa `start_time` và `end_time` của cảnh đó, ví dụ:\n"
            "```json\n"
            "{\n"
            '  "temporal_grounding": {\n'
            '    "start_time": "00:01:23",\n'
            '    "end_time": "00:01:28"\n'
            "  }\n"
            "}\n"
            "```\n"
            "Chỉ in JSON này nếu bạn chắc chắn về mốc thời gian từ dữ liệu cung cấp. Nếu không, bỏ qua.\n\n"
        )

        # ── Route-Specific Instructions ──
        if intent_value == "VISUAL":
            base_prompt += (
                "📌 CHÚ Ý [TÌM CẢNH PHIM]: Câu hỏi này yêu cầu tìm kiếm bằng hình ảnh.\n"
                "- Ưu tiên xác định scene cluster phù hợp nhất từ phần 'SCENE CLUSTER EVIDENCE'.\n"
                "- Ưu tiên mô tả các kết quả từ phần 'Visual Evidence'.\n"
                "- Phân tích mô tả cảnh, nhân vật, hành động trong các frame đó.\n\n"
            )
        elif intent_value == "KNOWLEDGE":
            base_prompt += (
                "📌 CHÚ Ý [TRA CỨU THÔNG TIN]: Câu hỏi này hỏi về thông tin văn bản (sự kiện, diễn viên, đạo diễn).\n"
                "- Tập trung vào phần 'Knowledge Evidence'.\n"
                "- Bỏ qua hình ảnh nếu không cần thiết.\n\n"
            )
        elif intent_value == "DIALOG":
            base_prompt += (
                "📌 CHÚ Ý [TÌM LỜI THOẠI]: Câu hỏi này xoay quanh lời thoại phim.\n"
                "- Kiểm tra kỹ subtitle trong 'Knowledge Evidence' và 'Script Sub-Scene Evidence'.\n"
                "- Ưu tiên lời thoại/ranh giới hẹp hơn từ Script Sub-Scene nếu có.\n\n"
            )
        elif intent_value == "MULTIMODAL" or intent_value in ["REASONING", "TEMPORAL"]:
            base_prompt += (
                "📌 CHÚ Ý [SUY LUẬN ĐA PHƯƠNG THỨC]: Câu hỏi này cần kết hợp cả thông tin text và hình ảnh.\n"
                "- Xác định scene cluster đúng trước, rồi mới dùng visual/script/graph để làm giàu chi tiết.\n"
                "- Tìm sự kết nối giữa 'Knowledge Evidence' và 'Visual Evidence'.\n"
                "- Đặc biệt chú ý đến timestamp/thời gian để khớp cảnh với sự kiện.\n\n"
            )
        else:  # Default/Factual
            base_prompt += "📌 CHÚ Ý [TRẢ LỜI THỰC TẾ]: Cung cấp câu trả lời ngắn gọn, trực tiếp dựa trên các kết quả truy xuất.\n\n"

        base_prompt += (
            "ƯU TIÊN EVIDENCE:\n"
            "- Nếu câu hỏi hỏi về một cảnh cụ thể, thời điểm, bối cảnh, chuyển cảnh, hoặc 'đoạn nào/ở đâu/khi nào': ưu tiên SCENE CLUSTER EVIDENCE trước để chốt window và scene cluster đúng.\n"
            "- Khi đã xác định được scene cluster, dùng Visual / Script / Graph Evidence cùng cluster đó để bổ sung chi tiết.\n"
            "- Nếu câu hỏi hỏi về location, scene heading, ai xuất hiện trong đoạn nào, hoặc lời thoại cụ thể: ưu tiên Script Sub-Scene Evidence trước.\n"
            "- Khi Script Sub-Scene Evidence có screenplay action/dialogue, coi đó là textual source mạnh hơn phần semantic summary ngắn.\n"
            "- Nếu Script Sub-Scene và Visual cùng trỏ tới một parent chunk/scene, dùng time window hẹp hơn của Script Sub-Scene.\n\n"
            "- Nếu câu hỏi hỏi về ai xuất hiện trong cảnh nào, quan hệ giữa các thực thể, hoặc cảnh nào xảy ra trước/sau: ưu tiên GRAPH EVIDENCE trước.\n\n"
        )

        # ── Context Formatting ──
        context_str = ""
        script_focus_by_chunk = {}
        graph_results = []
        plain_context_results = []

        if script_results:
            for result in script_results:
                metadata = getattr(result, "metadata", {}) or {}
                parent_chunk_id = metadata.get("parent_chunk_id", "")
                if not parent_chunk_id or parent_chunk_id in script_focus_by_chunk:
                    continue
                script_focus_by_chunk[parent_chunk_id] = metadata

        if context_results:
            for result in context_results:
                metadata = getattr(result, "metadata", {}) or {}
                if metadata.get("category") == "moviegraph":
                    graph_results.append(result)
                else:
                    plain_context_results.append(result)

        if plain_context_results:
            context_str += "--- KNOWLEDGE EVIDENCE (từ cơ sở dữ liệu phim) ---\n"
            for i, result in enumerate(plain_context_results):
                content = result.text.strip()[:500]
                movie = result.movie_id
                category = result.metadata.get("category", "info")
                title = result.metadata.get("title", movie)
                context_str += f"[Doc {i + 1} | Phim: {title} ({movie}) | Loại: {category}]\n{content}\n\n"

        if graph_results:
            context_str += "--- GRAPH EVIDENCE (Neo4j / liên kết quan hệ-cảnh) ---\n"
            for i, result in enumerate(graph_results):
                metadata = getattr(result, "metadata", {}) or {}
                movie = result.movie_id
                title = metadata.get("title", movie)
                node_type = metadata.get("node_type", "GraphNode")
                graph_heading = metadata.get("graph_heading", "")
                graph_location = metadata.get("graph_location", "")
                character_names = metadata.get("character_names", []) or []
                content = result.text.strip()[:500]
                context_str += (
                    f"[Graph {i + 1} | Phim: {title} | Node: {node_type} | "
                    f"Heading: {graph_heading} | Location: {graph_location}]\n"
                )
                if character_names:
                    context_str += f"Characters: {', '.join(character_names[:8])}\n"
                context_str += f"{content}\n\n"

        scene_context = self._format_scene_cluster_evidence(scene_results or [])
        if scene_context:
            context_str += scene_context

        # Add Visual Context
        if visual_results:
            context_str += "--- VISUAL EVIDENCE (các khung hình hệ thống của chúng tôi tìm thấy) ---\n"
            for i, r in enumerate(visual_results):
                movie = r.movie_id
                shot_id = r.metadata.get("shot_id", "unknown")
                score = r.score
                context_str += f"[Visual Match {i + 1} | Phim: {movie} | Cảnh: {shot_id} | Độ tin cậy: {score:.3f}]\n"
                chunk_id = r.metadata.get("chunk_id", "")
                if chunk_id and chunk_id in script_focus_by_chunk:
                    script_meta = script_focus_by_chunk[chunk_id]
                    context_str += (
                        f"  Script focus: {script_meta.get('script_heading', '')} | "
                        f"{script_meta.get('start_time', '')} -> {script_meta.get('end_time', '')} | "
                        f"{script_meta.get('script_location', '')}\n"
                    )
                description = r.metadata.get("description", "").strip()[:280]
                if description:
                    context_str += f"  {description}\n"
                vision_setting = r.metadata.get("vision_setting", "").strip()[:220]
                if vision_setting:
                    context_str += f"  Visual setting: {vision_setting}\n"
                vision_actions = r.metadata.get("vision_actions", "").strip()[:220]
                if vision_actions:
                    context_str += f"  Visual actions: {vision_actions}\n"
                visual_focus = r.metadata.get("visual_focus", "").strip()[:140]
                if visual_focus:
                    context_str += f"  Visual focus: {visual_focus}\n"
                screenplay_hint = r.metadata.get("screenplay_action_excerpt", "").strip()[:220]
                if screenplay_hint:
                    context_str += f"  Screenplay: {screenplay_hint}\n"
                dialogue = r.metadata.get("dialogue_text", "").strip()[:220]
                if dialogue:
                    context_str += f"  Dialogue: {dialogue}\n"
                context_str += "\n"

        if script_results:
            context_str += "--- SCRIPT SUB-SCENE EVIDENCE (screenplay-aware retrieval) ---\n"
            for i, result in enumerate(script_results):
                metadata = getattr(result, "metadata", {}) or {}
                movie = getattr(result, "movie_id", metadata.get("movie_id", "unknown"))
                heading = metadata.get("script_heading", "")
                location = metadata.get("script_location", "")
                start_time = metadata.get("start_time", "")
                end_time = metadata.get("end_time", "")
                anchor_quality = metadata.get("anchor_quality", "")
                alignment_confidence = metadata.get(
                    "alignment_confidence", metadata.get("confidence_score", 0.0)
                )
                context_str += (
                    f"[Script Match {i + 1} | Phim: {movie} | Heading: {heading} | "
                    f"Location: {location} | Window: {start_time}->{end_time} | "
                    f"Anchor: {anchor_quality} | Confidence: {alignment_confidence}]\n"
                )
                screenplay_action = metadata.get("screenplay_action_excerpt", "").strip()[:320]
                if screenplay_action:
                    context_str += f"Screenplay action: {screenplay_action}\n"
                screenplay_turns = metadata.get("screenplay_dialogue_turns", []) or []
                if screenplay_turns:
                    turns_preview = " | ".join(str(turn).strip() for turn in screenplay_turns[:4])
                    if turns_preview:
                        context_str += f"Screenplay turns: {turns_preview[:360]}\n"
                screenplay_dialogue = metadata.get("screenplay_dialogue_excerpt", "").strip()[:320]
                if screenplay_dialogue:
                    context_str += f"Screenplay dialogue: {screenplay_dialogue}\n"
                dialogue_excerpt = metadata.get("dialogue_excerpt", "").strip()[:320]
                if dialogue_excerpt:
                    context_str += f"Subtitle-aligned dialogue: {dialogue_excerpt}\n"
                semantic_description = metadata.get("semantic_description", "").strip()[:280]
                if semantic_description:
                    context_str += f"Semantic: {semantic_description}\n"
                context_str += "\n"

        if not context_str:
            context_str = (
                "No relevant results were found in our database for this query.\n"
            )

        # Add History
        history_str = ""
        if history:
            history_str = "--- CONVERSATION HISTORY ---\n"
            for msg in history[-5:]:
                if isinstance(msg, dict):
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    history_str += f"{role}: {msg.get('content', '')}\n"
                elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                    history_str += f"User: {msg[0]}\nAssistant: {msg[1]}\n"
            history_str += "\n"

        base_instructions = (
            "You are MovieRAG, a movie expert AI assistant. "
            "Our retrieval system has ALREADY searched the database and found the results below. "
            "Your job is to SYNTHESIZE these results into a helpful, natural language answer. "
            "NEVER say you cannot search or find information — the search is already done for you.\n"
            "When SCENE CLUSTER EVIDENCE is present, treat it as the primary fused grounding for concrete scene, location, and timestamp answers.\n"
            "When GRAPH EVIDENCE is present, treat it as the strongest source for scene-to-character links, scene transitions, and relationship questions.\n"
        )

        if intent_value == "MULTIMODAL" or intent_value in ["REASONING"]:
            instruction = "Analyze the evidence deeply: explain character motives, causality, and connections."
        elif intent_value == "TEMPORAL":
            instruction = "Focus on WHEN events happen, the chronological sequence, and timestamps."
        elif intent_value == "DIALOG":
            instruction = (
                "Focus on dialog and quotes. Identify who said what and the context."
            )
        else:  # FACTUAL / KNOWLEDGE / VISUAL
            instruction = (
                "Provide a concise, direct answer based on the retrieved results."
            )

        prompt = f"""{base_instructions}
Task: {instruction}

Rules:
- Use the RETRIEVED results below to form your answer. They are real search results.
- Reference specific movies, shots, and details from the context.
- If a user uploads an image and visual matches exist, tell them which movie and scene was identified.
- Answer in the same language as the user's question.

{history_str}
{context_str}

User's Question: {query}
Answer:"""

        return prompt

    def generate_answer(
        self,
        query: str,
        context_results: List[TextSearchResult],
        visual_results: Optional[List[VisualSearchResult]] = None,
        script_results: Optional[List[TextSearchResult]] = None,
        scene_results: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        route: Any = None,
    ) -> str:
        """
        Generate an answer using the LLM.
        """
        if not self.client:
            return (
                "LLM generation is unavailable. Please ensure `google-genai` is installed "
                "and a valid API key is provided (GEMINI_API_KEY environment variable)."
            )

        prompt = self.format_prompt(
            query,
            context_results,
            visual_results,
            script_results or [],
            scene_results or [],
            history,
            route=route,
        )

        contents = [prompt]
        if visual_results:
            try:
                from PIL import Image

                for r in visual_results:
                    if hasattr(r, "path") and r.path and os.path.exists(r.path):
                        img = Image.open(r.path)
                        contents.append(img)
            except Exception as e:
                logger.warning(f"Could not attach images: {e}")

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            return f"Error generating answer: {str(e)}"
