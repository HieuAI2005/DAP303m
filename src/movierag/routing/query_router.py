import logging
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryRoute(str, Enum):
    FACTUAL = "FACTUAL"  # BOLT path: Direct visual/text match
    REASONING = "REASONING"  # DrVideo path: Deep causality, motives
    TEMPORAL = "TEMPORAL"  # MomentGPT path: Timestamps, sequences
    DIALOG = "DIALOG"  # Dialog path: Subtitle/quote search
    MACRO_KNOWLEDGE = "MACRO_KNOWLEDGE"  # Plot summary, big picture synopsis


class QueryClassification(BaseModel):
    route: QueryRoute = Field(description="The classification of the user query.")
    reason: str = Field(description="Brief reason why this route was chosen.")


class QueryRouter:
    """
    Intelligent Query Router inspired by DrVideo, BOLT, and MomentGPT.
    Selects the optimal execution path for a given query to balance speed and accuracy.
    Uses Rule-based heuristics for fast routing, and LLM fallback for ambiguous queries.
    """

    def __init__(
        self,
        model_id: str = "gemma-3-27b-it",
        api_key: Optional[str] = None,
    ):
        self.model_id = model_id
        self.client = None

        try:
            from movierag.generation.universal_client import UniversalLLMClient

            self.client = UniversalLLMClient(model_id=self.model_id)
        except Exception as e:
            logger.warning(
                f"Could not initialize UniversalLLMClient for QueryRouter: {e}"
            )

        # Heuristic keywords for ultra-fast routing (< 10ms)
        self.temporal_keywords = [
            "when",
            "what time",
            "how long",
            "before",
            "after",
            "during",
            "timestamp",
            "khi nào",
            "bao giờ",
            "lúc nào",
            "bao lâu",
            "trước khi",
            "sau khi",
            "khi",
        ]

        self.reasoning_keywords = [
            "why",
            "how",
            "reason",
            "purpose",
            "explain",
            "meaning",
            "motive",
            "tại sao",
            "vị sao",
            "như thế nào",
            "lý do",
            "mục đích",
            "giải thích",
            "ý nghĩa",
            "nguyên nhân",
        ]

        self.dialog_keywords = [
            "who said",
            "who says",
            "dialog",
            "dialogue",
            "quote",
            "line",
            "say",
            "said",
            "spoken",
            "ai nói",
            "câu thoại",
            "lời thoại",
            "hội thoại",
            "câu nói",
        ]

    def _rule_based_route(self, query: str) -> Optional[QueryRoute]:
        """Fast heuristic routing based on keyword detection."""
        lower_query = query.lower()

        # Check dialog first (quote/dialog-specific queries)
        if any(
            re.search(r"\b" + kw + r"\b", lower_query) for kw in self.dialog_keywords
        ):
            return QueryRoute.DIALOG

        # Check temporal (high specificity)
        if any(
            re.search(r"\b" + kw + r"\b", lower_query) for kw in self.temporal_keywords
        ):
            return QueryRoute.TEMPORAL

        # Check reasoning (requires deeper context)
        if any(
            re.search(r"\b" + kw + r"\b", lower_query) for kw in self.reasoning_keywords
        ):
            return QueryRoute.REASONING

        return None

    def route_query(self, query: str) -> QueryRoute:
        """
        Determines the optimal architecture route for the question.
        """
        if not query or not query.strip():
            return QueryRoute.FACTUAL

        # 1. Fast Rule-Based Pass
        rule_route = self._rule_based_route(query)
        if rule_route:
            logger.debug(f"Rule-based routing selected: {rule_route.value}")
            return rule_route

        # 2. LLM-Based Pass (If heuristics fail and client is available)
        if self.client:
            try:
                prompt_text = (
                    "You are the 'Query Router' for a multi-modal Video RAG system.\n"
                    "Classify the following query into exactly one of three categories:\n"
                    "1. TEMPORAL: Questions asking about timing, sequence of events, or duration (e.g., 'When did he arrive?', 'What happens after the explosion?').\n"
                    "2. REASONING: Questions requiring deep understanding of character motives, causality, or complex Why/How connections (e.g., 'Why is she crying?', 'How did they escape?').\n"
                    "3. FACTUAL: Direct questions asking to find a specific scene, person, object, or action without deep causality (e.g., 'Find the car chase scene', 'Who is the man in the red hat', 'What is he holding?').\n\n"
                    f"Query: '{query}'\n\n"
                    "Output a JSON containing 'route' and 'reason'."
                )

                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt_text,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": QueryClassification,
                        "temperature": 0.0,  # Zero temperature for deterministic routing
                    },
                )

                result = json.loads(response.text)
                route_str = result.get("route", "FACTUAL").upper()

                if route_str in ["FACTUAL", "REASONING", "TEMPORAL", "DIALOG"]:
                    logger.debug(
                        f"LLM routing selected: {route_str} (Reason: {result.get('reason')})"
                    )
                    return QueryRoute(route_str)

            except Exception as e:
                logger.error(f"LLM routing failed: {e}. Defaulting to FACTUAL.")

        # 3. Default Fallback
        logger.debug("Default routing selected: FACTUAL")
        return QueryRoute.FACTUAL
