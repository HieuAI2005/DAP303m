# ─────────────────────────────────────────────────────────────────────────────
# temporal_grounding.py
# Temporal Grounding Engine — Video Understanding Pipeline
# Resolves temporal expressions and localizes events in video timeline
# ─────────────────────────────────────────────────────────────────────────────
"""
 Temporal Grounding Engine for Video Understanding.

 Resolves queries like:
   "When does Rose first appear?"
   "Find the scene where Jack draws Rose"
   "What happens at 1:30:00?"
   "When is the last time we see Jack?"

 Key capabilities:
   1. Temporal Expression Parsing — resolve "first", "last", "before X", etc.
   2. Temporal Reasoning — combine temporal constraints with semantic retrieval
   3. Moment Localization — return precise [start, end] timestamps
   4. Cross-reference Verification — verify grounding against scene metadata

 Output schema:
   {
     "query": str,
     "temporal_expression": str | None,
     "expression_type": str,        # "first_occurrence", "last_occurrence", "at_time", etc.
     "temporal_segment": [float, float],  # [start_seconds, end_seconds]
     "groundedness": str,           # "strong", "moderate", "weak"
     "confidence": float,          # 0.0 - 1.0
     "evidence": {
       "scene_clusters": [...],
       "temporal_chunks": [...],
       "script_excerpts": [...],
     },
     "reasoning_trace": str,
   }
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Temporal Expression Types ───────────────────────────────────────────────────

@dataclass
class TemporalExpression:
    """Parsed temporal expression from a natural language query."""
    raw: str                          # Original phrase
    expr_type: str                     # "first", "last", "at_time", "before", "after", "between"
    resolved_seconds: Optional[float]  # Absolute time if applicable
    anchor_entity: Optional[str]      # Entity being anchored (e.g., "Rose")
    constraint: Optional[str]         # Additional constraint ("before dinner")
    confidence: float = 1.0


@dataclass
class TemporalGrounding:
    """Final temporal grounding result."""
    query: str
    expression: Optional[TemporalExpression]
    segment: Tuple[float, float]      # [start, end] in seconds
    movie_id: str
    grounded: bool = True
    confidence: float = 1.0
    evidence_sources: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: str = ""


# ── Temporal Expression Parser ─────────────────────────────────────────────────

class TemporalExpressionParser:
    """
    Parses temporal expressions in natural language queries.

    Supported patterns:
      - "first time X appears"     → first_occurrence
      - "last time we see X"       → last_occurrence
      - "at HH:MM:SS"              → at_time
      - "at N minutes/seconds"     → at_time (relative)
      - "before X happens"         → before_event
      - "after X"                  → after_event
      - "between X and Y"          → between_events
      - "during X"                 → during_event
    """

    # Patterns ordered by specificity
    PATTERNS: List[Tuple[str, str, Callable[[re.Match], TemporalExpression]]] = [
        # "at 1:30:00" or "at 01:30:00" or "at 1:30"
        (
            r'at\s+(\d{1,2}:\d{2}(?::\d{2})?)',
            "at_time",
            lambda m: TemporalExpression(
                raw=m.group(0),
                expr_type="at_time",
                resolved_seconds=_parse_timestamp(m.group(1)),
                anchor_entity=None,
                constraint=None,
                confidence=1.0,
            )
        ),
        # "at N seconds/minutes"
        (
            r'at\s+(\d+(?:\.\d+)?)\s*(seconds?|minutes?|mins?|secs?|hours?|hrs?)',
            "at_time",
            lambda m: TemporalExpression(
                raw=m.group(0),
                expr_type="at_time",
                resolved_seconds=_parse_duration(m.group(1), m.group(2)),
                anchor_entity=None,
                constraint=None,
                confidence=0.9,
            )
        ),
        # "the first time X appears/see"
        (
            r'(?:the\s+)?first\s+(?:time\s+)?(?:we\s+|I\s+)?(?:see|see\s+|see\s+)?([A-Z][a-zA-Z\s]+?)(?:\s+(?:appears?|see|appear|shown))',
            "first_occurrence",
            lambda m: TemporalExpression(
                raw=m.group(0),
                expr_type="first_occurrence",
                resolved_seconds=None,
                anchor_entity=m.group(1).strip(),
                constraint=None,
                confidence=0.9,
            )
        ),
        # "first X" (simplified)
        (
            r'first\s+([a-z][a-zA-Z]+)',
            "first_occurrence",
            lambda m: TemporalExpression(
                raw=m.group(0),
                expr_type="first_occurrence",
                resolved_seconds=None,
                anchor_entity=m.group(1).strip(),
                constraint=None,
                confidence=0.8,
            )
        ),
        # "last time we see X"
        (
            r'last\s+(?:time\s+)?(?:we\s+|I\s+)?(?:see|see\s+|saw)\s+([A-Z][a-zA-Z]+)',
            "last_occurrence",
            lambda m: TemporalExpression(
                raw=m.group(0),
                expr_type="last_occurrence",
                resolved_seconds=None,
                anchor_entity=m.group(1).strip(),
                constraint=None,
                confidence=0.9,
            )
        ),
        # "before X happens"
        (
            r'before\s+([A-Z][a-zA-Z]+)\s+(?:happens?|appears?)',
            "before_event",
            lambda m: TemporalExpression(
                raw=m.group(0),
                expr_type="before_event",
                resolved_seconds=None,
                anchor_entity=None,
                constraint=m.group(1).strip(),
                confidence=0.85,
            )
        ),
        # "after X"
        (
            r'after\s+([A-Z][a-zA-Z]+)',
            "after_event",
            lambda m: TemporalExpression(
                raw=m.group(0),
                expr_type="after_event",
                resolved_seconds=None,
                anchor_entity=None,
                constraint=m.group(1).strip(),
                confidence=0.85,
            )
        ),
        # "when does X happen/appear"
        (
            r'when\s+does\s+([A-Z][a-zA-Z]+)\s+(happen|appear|show)',
            "any_occurrence",
            lambda m: TemporalExpression(
                raw=m.group(0),
                expr_type="any_occurrence",
                resolved_seconds=None,
                anchor_entity=m.group(1).strip(),
                constraint=None,
                confidence=0.85,
            )
        ),
        # "during X"
        (
            r'during\s+(?:the\s+)?([A-Z][a-zA-Z]+)',
            "during_event",
            lambda m: TemporalExpression(
                raw=m.group(0),
                expr_type="during_event",
                resolved_seconds=None,
                anchor_entity=None,
                constraint=m.group(1).strip(),
                confidence=0.8,
            )
        ),
    ]

    def parse(self, query: str) -> Optional[TemporalExpression]:
        """
        Parse temporal expression from query string.

        Returns None if no temporal expression is found.
        """
        for pattern, expr_type, factory in self.PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                expr = factory(match)
                logger.debug(f"Parsed temporal expression: {expr}")
                return expr

        return None


# ── Timestamp Utilities ────────────────────────────────────────────────────────

def _parse_timestamp(ts: str) -> float:
    """Parse 'HH:MM:SS' or 'MM:SS' into seconds."""
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def _parse_duration(value: str, unit: str) -> float:
    """Parse 'N unit' into seconds."""
    v = float(value)
    unit = unit.lower()
    if unit.startswith("min"):
        return v * 60
    elif unit.startswith("hr"):
        return v * 3600
    elif unit.startswith("sec"):
        return v
    else:
        return v


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to 'HH:MM:SS' format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ── Temporal Grounding Engine ─────────────────────────────────────────────────

class TemporalGroundingEngine:
    """
    Resolves temporal queries against video scene metadata.

    The engine combines:
      1. Temporal Expression Parsing (regex-based)
      2. Semantic retrieval (CLIP FAISS + scene metadata)
      3. Temporal constraint scoring
      4. Cross-reference with script/dialogue data
    """

    def __init__(
        self,
        scene_metadata_retriever: Optional[Any] = None,
        script_retriever: Optional[Any] = None,
        dialogue_retriever: Optional[Any] = None,
        neo4j_store: Optional[Any] = None,
        default_segment_duration: float = 30.0,
    ):
        """
        Args:
            scene_metadata_retriever: Callable returning scene metadata list.
                Signature: (movie_id, entity, limit) → List[Dict]
                Each dict should have: start_seconds, end_seconds, description, characters
            script_retriever: Callable returning script scenes.
                Signature: (movie_id, keywords, limit) → List[Dict]
            dialogue_retriever: Callable returning dialogue chunks.
                Signature: (movie_id, keywords, limit) → List[Dict]
            neo4j_store: Neo4jGraphStore instance for graph-based temporal reasoning.
            default_segment_duration: Default segment length in seconds.
        """
        self.parser = TemporalExpressionParser()
        self.scene_retriever = scene_metadata_retriever
        self.script_retriever = script_retriever
        self.dialogue_retriever = dialogue_retriever
        self.neo4j = neo4j_store
        self.default_duration = default_segment_duration

    def ground(
        self,
        query: str,
        movie_id: str,
        candidate_scenes: Optional[List[Dict[str, Any]]] = None,
        k: int = 10,
    ) -> TemporalGrounding:
        """
        Main entry point: resolve temporal query to a video segment.

        Args:
            query: Natural language query (may contain temporal expressions).
            movie_id: Movie identifier.
            candidate_scenes: Pre-retrieved scene metadata (optional).
            k: Number of candidates to retrieve if not provided.

        Returns:
            TemporalGrounding with segment, confidence, and evidence.
        """
        # Step 1: Parse temporal expression
        expr = self.parser.parse(query)

        # Step 2: Retrieve candidate scenes
        if candidate_scenes is None:
            anchor_entity = expr.anchor_entity if expr else None
            entity_query = anchor_entity or query
            candidate_scenes = self._retrieve_candidates(
                movie_id, entity_query, k
            )

        # Step 3: Score and rank candidates
        scored = self._score_candidates(query, expr, candidate_scenes, movie_id)

        # Step 4: Build reasoning trace
        reasoning = self._build_reasoning_trace(query, expr, scored)

        # Step 5: Extract best segment
        if scored:
            best = scored[0]
            segment = (best["start_seconds"], best["end_seconds"])
            confidence = best["temporal_score"] * best.get("semantic_score", 0.8)
            grounded = confidence >= 0.3
        else:
            segment = (0.0, self.default_duration)
            confidence = 0.0
            grounded = False

        return TemporalGrounding(
            query=query,
            expression=expr,
            segment=segment,
            movie_id=movie_id,
            grounded=grounded,
            confidence=round(confidence, 3),
            evidence_sources=scored[:3],
            reasoning_trace=reasoning,
        )

    # ── Candidate Retrieval ──────────────────────────────────────────────────

    def _retrieve_candidates(
        self,
        movie_id: str,
        query: str,
        k: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve scene candidates from multiple sources."""
        candidates: List[Dict[str, Any]] = []

        # From scene metadata retriever
        if self.scene_retriever:
            try:
                scenes = self.scene_retriever(movie_id, query, k)
                for s in scenes:
                    s["_source"] = "scene_metadata"
                candidates.extend(scenes)
            except Exception as e:
                logger.warning(f"Scene retriever failed: {e}")

        # From script retriever
        if self.script_retriever:
            try:
                scripts = self.script_retriever(movie_id, query, k)
                for s in scripts:
                    s["_source"] = "script"
                candidates.extend(scripts)
            except Exception as e:
                logger.warning(f"Script retriever failed: {e}")

        # From dialogue retriever
        if self.dialogue_retriever:
            try:
                dialogues = self.dialogue_retriever(movie_id, query, k)
                for d in dialogues:
                    d["_source"] = "dialogue"
                candidates.extend(dialogues)
            except Exception as e:
                logger.warning(f"Dialogue retriever failed: {e}")

        # From Neo4j temporal search
        if self.neo4j:
            try:
                temporal_hits = self.neo4j.search(
                    f"temporal:{query}", top_k=k, filters={"movie_id": movie_id}
                )
                for h in temporal_hits:
                    h["_source"] = "graph"
                candidates.extend(temporal_hits)
            except Exception as e:
                logger.warning(f"Neo4j temporal search failed: {e}")

        return candidates

    # ── Candidate Scoring ─────────────────────────────────────────────────────

    def _score_candidates(
        self,
        query: str,
        expr: Optional[TemporalExpression],
        candidates: List[Dict[str, Any]],
        movie_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Score and sort candidates based on temporal expression matching.
        """
        for c in candidates:
            # Temporal score: how well this candidate satisfies the expression
            temporal_score = self._temporal_score(c, expr, query)
            # Semantic score: baseline relevance
            semantic_score = c.get("relevance_score", 0.5)
            # Combined
            c["temporal_score"] = temporal_score
            c["semantic_score"] = semantic_score
            c["combined_score"] = temporal_score * 0.6 + semantic_score * 0.4

        scored = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
        return scored

    def _temporal_score(
        self,
        candidate: Dict[str, Any],
        expr: Optional[TemporalExpression],
        query: str,
    ) -> float:
        """
        Score a candidate based on temporal expression.

        Returns score in [0, 1].
        """
        if expr is None:
            # No temporal expression — just use duration penalty
            duration = candidate.get("end_seconds", 0) - candidate.get("start_seconds", 0)
            if 0 < duration <= 120:
                return 1.0
            return max(0.5, 1.0 - (duration - 120) / 120)

        expr_type = expr.expr_type
        start = candidate.get("start_seconds", 0.0)
        end = candidate.get("end_seconds", 0.0)
        duration = end - start

        if expr_type == "at_time":
            if expr.resolved_seconds is not None:
                # Check if candidate overlaps with specified time
                if start <= expr.resolved_seconds <= end:
                    return 1.0
                dist = min(abs(start - expr.resolved_seconds), abs(end - expr.resolved_seconds))
                return max(0, 1.0 - dist / 300)  # decay over 5 minutes

        elif expr_type == "first_occurrence":
            # Prefer earliest occurrence
            if start < 60:  # within first minute
                return 1.0
            # Look for "first" keyword in description
            desc = candidate.get("description", "").lower()
            if "first" in desc:
                return 0.9
            # Penalize later occurrences
            return max(0.2, 1.0 - start / 7200)  # decay over 2 hours

        elif expr_type == "last_occurrence":
            # Prefer latest occurrence
            # Higher score for later timestamps
            return min(1.0, start / 7200 + 0.3)

        elif expr_type in ("before_event", "after_event", "during_event"):
            # Use graph-based temporal ordering
            anchor = expr.constraint or expr.anchor_entity
            if anchor:
                # Score based on proximity to anchor
                anchor_start = candidate.get(f"{anchor.lower()}_start", None)
                if anchor_start is not None:
                    if expr_type == "before_event":
                        return 1.0 if end <= anchor_start else max(0, 1.0 - (start - anchor_start) / 300)
                    elif expr_type == "after_event":
                        return 1.0 if start >= anchor_start else max(0, 1.0 - (anchor_start - start) / 300)
                    elif expr_type == "during_event":
                        return 0.9
            return 0.6

        elif expr_type == "any_occurrence":
            # Prefer scenes with characters mentioned
            anchor = expr.anchor_entity
            if anchor:
                chars = candidate.get("characters", [])
                if isinstance(chars, list) and any(anchor.lower() in str(c).lower() for c in chars):
                    return 0.95
            return 0.5

        return 0.5  # default

    # ── Reasoning Trace ───────────────────────────────────────────────────────

    def _build_reasoning_trace(
        self,
        query: str,
        expr: Optional[TemporalExpression],
        scored: List[Dict[str, Any]],
    ) -> str:
        """Build human-readable reasoning trace."""
        lines = [f"Query: {query}"]

        if expr:
            lines.append(f"Temporal Expression: '{expr.raw}' (type: {expr.expr_type})")
            if expr.anchor_entity:
                lines.append(f"  Anchor entity: {expr.anchor_entity}")
            if expr.resolved_seconds is not None:
                lines.append(f"  Resolved to: {_seconds_to_timestamp(expr.resolved_seconds)}")
        else:
            lines.append("No explicit temporal expression detected.")

        if scored:
            best = scored[0]
            lines.append(f"Best candidate: scene at [{_seconds_to_timestamp(best['start_seconds'])}, {_seconds_to_timestamp(best['end_seconds'])}]")
            lines.append(f"  Combined score: {best['combined_score']:.3f} (temporal: {best['temporal_score']:.3f})")
            if "description" in best:
                lines.append(f"  Scene description: {best['description'][:120]}...")
        else:
            lines.append("No matching candidates found.")

        return "\n".join(lines)

    # ── Output Formatting ─────────────────────────────────────────────────────

    def to_json(self, grounding: TemporalGrounding) -> Dict[str, Any]:
        """Serialize TemporalGrounding to JSON-compatible dict."""
        expr_dict = None
        if grounding.expression:
            e = grounding.expression
            expr_dict = {
                "raw": e.raw,
                "expr_type": e.expr_type,
                "resolved_seconds": e.resolved_seconds,
                "anchor_entity": e.anchor_entity,
                "constraint": e.constraint,
                "confidence": e.confidence,
            }

        return {
            "query": grounding.query,
            "temporal_expression": expr_dict,
            "temporal_segment": list(grounding.segment),
            "movie_id": grounding.movie_id,
            "grounded": grounding.grounded,
            "confidence": grounding.confidence,
            "evidence_sources": grounding.evidence_sources,
            "reasoning_trace": grounding.reasoning_trace,
        }
