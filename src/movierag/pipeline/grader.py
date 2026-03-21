"""
Document Relevance Grader
=========================
LLM-based grading for retrieved documents.
Inspired by RagLaw _grade_documents pattern.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DocumentGrader:
    """
    Grade retrieved documents for relevance to a query.
    Uses LLM to score each document 0-1.
    Falls back to score-based heuristic if LLM unavailable.
    """

    def __init__(self, llm_client=None, model_id: str = "gemini-2.5-flash"):
        self.llm_client = llm_client
        self.model_id = model_id
        self._init_client()

    def _init_client(self):
        if self.llm_client is None:
            try:
                from google import genai

                self.llm_client = genai.Client()
            except Exception as e:
                logger.warning(f"Could not initialize grader LLM: {e}")

    def grade_knowledge_results(
        self,
        query: str,
        results: List[Any],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Grade knowledge search results for relevance.

        Returns:
            {
                "relevant": [...],
                "avg_score": float,
                "needs_rewrite": bool,
            }
        """
        if not results:
            return {"relevant": [], "avg_score": 0.0, "needs_rewrite": True}

        # Fast path: use retrieval scores as relevance proxy
        relevant = []
        scores = []

        for r in results:
            score = r.score if hasattr(r, "score") else 0.0
            scores.append(score)
            if score >= threshold:
                relevant.append(r)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # If no results pass threshold, keep top 3 anyway
        if not relevant and results:
            relevant = results[:3]

        return {
            "relevant": relevant,
            "avg_score": avg_score,
            "needs_rewrite": avg_score < threshold and len(relevant) < 2,
        }

    def grade_visual_results(
        self,
        query: str,
        results: List[Any],
        threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """Grade visual search results."""
        if not results:
            return {"relevant": [], "avg_score": 0.0, "needs_rewrite": True}

        relevant = []
        scores = []

        for r in results:
            score = r.score if hasattr(r, "score") else 0.0
            scores.append(score)
            if score >= threshold:
                relevant.append(r)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        if not relevant and results:
            relevant = results[:3]

        return {
            "relevant": relevant,
            "avg_score": avg_score,
            "needs_rewrite": avg_score < threshold,
        }

    def rewrite_query(self, original_query: str, attempt: int = 1) -> str:
        """
        Rewrite query for better retrieval.
        Uses LLM to rephrase, expand, or clarify.
        """
        if not self.llm_client:
            return original_query

        try:
            prompt = (
                f"Câu hỏi sau không tìm được kết quả tốt trong cơ sở dữ liệu phim:\n"
                f"'{original_query}'\n\n"
                f"Hãy viết lại câu hỏi (lần thử {attempt}/3) để tìm kiếm hiệu quả hơn.\n"
                f"Mở rộng từ khóa, thêm ngữ cảnh, hoặc dịch sang tiếng Anh nếu cần.\n"
                f"Chỉ trả về câu hỏi mới, không giải thích."
            )
            response = self.llm_client.models.generate_content(
                model=self.model_id,
                contents=prompt,
            )
            rewritten = response.text.strip()
            logger.info(f"Query rewrite {attempt}: '{original_query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return original_query
