# ─────────────────────────────────────────────────────────────────────────────
# causal_reasoner.py
# Causal Reasoning Pipeline — Video Understanding Pipeline
# Layer 5: Script & Narrative — causal_relations + narrative reasoning
# ─────────────────────────────────────────────────────────────────────────────
"""
 Causal Reasoning Pipeline using Neo4j-backed knowledge graphs.

 Builds and queries narrative causal graphs to answer "Why" questions:
   "Why does Rose decide to let Jack go?"
   "What causes Jack's death?"
   "How does the iceberg collision lead to the ship's sinking?"

 Key capabilities:
   1. Causal Graph Construction — extract cause-effect pairs from scene descriptions
   2. Multi-hop Reasoning — traverse graph to find indirect causal chains
   3. Counterfactual Reasoning — explore "what if" scenarios
   4. Narrative Explanation Synthesis — generate coherent narrative explanations

 The graph schema follows Layer 5: Script & Narrative → causal_relations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Causal Graph Types ──────────────────────────────────────────────────────────

@dataclass
class CausalTriple:
    """A single cause-effect relationship."""
    cause: str           # Subject/action that causes
    relation: str        # e.g. "leads_to", "enables", "prevents", "triggers"
    effect: str          # Resulting event/state
    scene_id: Optional[str] = None
    movie_id: Optional[str] = None
    evidence: str = ""   # Textual evidence from scene
    confidence: float = 1.0


@dataclass
class CausalChain:
    """A chain of cause-effect relationships forming a narrative path."""
    chain_id: str
    events: List[str] = field(default_factory=list)
    triples: List[CausalTriple] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 1.0
    depth: int = 0


@dataclass
class CausalAnswer:
    """Full answer to a causal reasoning query."""
    query: str
    target_event: str
    causal_explanation: str
    direct_causes: List[str] = field(default_factory=list)
    causal_chain: Optional[CausalChain] = None
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    reasoning_trace: str = ""


# ── Causal Triple Extractor ────────────────────────────────────────────────────

CAUSAL_VERBS = {
    "leads_to", "causes", "results_in", "triggers", "enables",
    "prevents", "forces", "motivates", "creates", "destroys",
    "provokes", "inspires", "discourages", "blocks", "allows",
    "requires", "demands", "compels", "prompts", "initiates",
    "concludes", "produces", "induces", "elicits", "evokes",
}

CAUSAL_PATTERNS = [
    # "X causes Y"
    r"(?P<cause>.+?)\s+(?:causes|caused|led to?|results? in?|triggers?|triggered)\s+(?P<effect>.+?)(?:\.|,|$)",
    # "Because of X, Y"
    r"(?:because of|because|due to|owing to)\s+(?P<cause>.+?),\s+(?P<effect>.+?)(?:\.|,|$)",
    # "X, which leads to Y"
    r"(?P<cause>.+?),?\s+which\s+(?:leads?|led)\s+to\s+(?P<effect>.+?)(?:\.|,|$)",
    # "X is the result of Y"
    r"(?P<effect>.+?)\s+is\s+(?:the\s+)?(?:direct\s+)?result\s+of\s+(?P<cause>.+?)(?:\.|,|$)",
    # "X therefore Y"
    r"(?P<cause>.+?)\s+therefore\s+(?P<effect>.+?)(?:\.|,|$)",
    # "X, so Y"
    r"(?P<cause>.+?),\s+so\s+(?P<effect>.+?)(?:\.|,|$)",
]


class CausalTripleExtractor:
    """
    Extracts cause-effect triples from natural language text.

    Uses both:
      1. Pattern matching (CAUSAL_PATTERNS)
      2. LLM-based extraction (more accurate, used as fallback)
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm_client = llm_client

    def _get_client(self):
        if self._llm_client is None:
            from movierag.generation.universal_client import UniversalLLMClient
            self._llm_client = UniversalLLMClient()
        return self._llm_client

    def extract_from_text(
        self,
        text: str,
        scene_id: Optional[str] = None,
        movie_id: Optional[str] = None,
        use_llm: bool = True,
    ) -> List[CausalTriple]:
        """
        Extract causal triples from text.

        Args:
            text: Scene description, script excerpt, or narrative text.
            scene_id: Optional scene identifier for linking.
            movie_id: Optional movie identifier.
            use_llm: Use LLM extraction (more accurate). If False, use patterns only.

        Returns:
            List of CausalTriple objects.
        """
        import re

        triples: List[CausalTriple] = []

        # Pattern-based extraction
        for pattern in CAUSAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                cause = match.group("cause").strip()
                effect = match.group("effect").strip()
                if cause and effect and len(cause) > 3 and len(effect) > 3:
                    triples.append(CausalTriple(
                        cause=cause,
                        relation="causes",
                        effect=effect,
                        scene_id=scene_id,
                        movie_id=movie_id,
                        evidence=text[:200],
                        confidence=0.7,
                    ))

        # Deduplicate
        seen = set()
        unique_triples = []
        for t in triples:
            key = (t.cause.lower(), t.effect.lower())
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)

        # LLM-based extraction (more thorough)
        if use_llm and len(text) > 50:
            llm_triples = self._extract_with_llm(text, scene_id, movie_id)
            for t in llm_triples:
                key = (t.cause.lower(), t.effect.lower())
                if key not in seen:
                    seen.add(key)
                    unique_triples.append(t)

        return unique_triples

    def _extract_with_llm(
        self,
        text: str,
        scene_id: Optional[str],
        movie_id: Optional[str],
    ) -> List[CausalTriple]:
        """Use LLM to extract causal triples from text."""
        client = self._get_client()

        prompt = f"""Extract all cause-effect relationships from the following text.

Text: {text[:1500]}

Output as JSON array of objects with keys: "cause", "relation", "effect".
Each cause and effect should be a short phrase (5-15 words).
If no clear causal relationships exist, output an empty array [].

Example:
[{{"cause": "Jack saves Rose", "relation": "leads_to", "effect": "Rose falls in love"}},
 {{"cause": "Cal's jealousy", "relation": "triggers", "effect": "confrontation with Jack"}}]
"""
        try:
            response = client.generate_content(
                model=None,
                contents=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
            )

            import json, re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                items = json.loads(json_match.group(0))
                triples = []
                for item in items:
                    if all(k in item for k in ("cause", "effect")):
                        triples.append(CausalTriple(
                            cause=item["cause"],
                            relation=item.get("relation", "causes"),
                            effect=item["effect"],
                            scene_id=scene_id,
                            movie_id=movie_id,
                            evidence=text[:200],
                            confidence=0.9,
                        ))
                return triples
        except Exception as e:
            logger.warning(f"LLM causal extraction failed: {e}")

        return []


# ── Causal Reasoner ────────────────────────────────────────────────────────────

class CausalReasoner:
    """
    Causal reasoning over narrative knowledge graphs.

    Usage:
        reasoner = CausalReasoner(neo4j_store=graph_store)
        answer = reasoner.answer_why("Why does Rose let Jack go?", movie_id="tt0120338")
    """

    def __init__(
        self,
        neo4j_store: Optional[Any] = None,
        triple_extractor: Optional[CausalTripleExtractor] = None,
        scene_retriever: Optional[Callable] = None,
        max_hops: int = 3,
    ):
        """
        Args:
            neo4j_store: Neo4jGraphStore instance for graph queries.
            triple_extractor: CausalTripleExtractor instance.
            scene_retriever: Callable(movie_id, query, k) → List[Dict] for scene retrieval.
            max_hops: Maximum causal chain depth.
        """
        self.neo4j = neo4j_store
        self.extractor = triple_extractor or CausalTripleExtractor()
        self.scene_retriever = scene_retriever
        self.max_hops = max_hops

    def answer_why(
        self,
        query: str,
        movie_id: str,
        context_scenes: Optional[List[Dict[str, Any]]] = None,
    ) -> CausalAnswer:
        """
        Answer a "Why" question using causal graph reasoning.

        Args:
            query: The "Why" question.
            movie_id: Movie identifier.
            context_scenes: Pre-retrieved scene metadata (optional).

        Returns:
            CausalAnswer with explanation and supporting evidence.
        """
        # Step 1: Identify target event from query
        target_event = self._extract_target_event(query)

        # Step 2: Retrieve relevant scene context
        if context_scenes is None and self.scene_retriever:
            context_scenes = self.scene_retriever(movie_id, target_event, k=10)

        # Step 3: Extract causal triples from context
        all_triples: List[CausalTriple] = []
        for scene in (context_scenes or []):
            text = scene.get("description", "") or scene.get("vlm_description", "")
            if text:
                triples = self.extractor.extract_from_text(
                    text,
                    scene_id=scene.get("scene_id"),
                    movie_id=movie_id,
                )
                all_triples.extend(triples)

        # Step 4: Build causal chain
        chain = self._build_causal_chain(target_event, all_triples)

        # Step 5: Synthesize explanation
        explanation = self._synthesize_explanation(query, target_event, chain, all_triples)

        # Step 6: Build reasoning trace
        trace = self._build_reasoning_trace(query, target_event, chain, all_triples)

        return CausalAnswer(
            query=query,
            target_event=target_event,
            causal_explanation=explanation,
            direct_causes=[t.cause for t in all_triples if t.effect in target_event][:5],
            causal_chain=chain,
            supporting_evidence=[
                {"cause": t.cause, "effect": t.effect, "evidence": t.evidence, "scene_id": t.scene_id}
                for t in all_triples[:10]
            ],
            confidence=min(1.0, len(all_triples) * 0.15 + 0.3),
            reasoning_trace=trace,
        )

    # ── Target Event Extraction ───────────────────────────────────────────────

    def _extract_target_event(self, query: str) -> str:
        """Extract the target event from a 'Why' question."""
        import re

        # Remove question words
        query = re.sub(r'^(why|what caused|how come)\s+', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\?$', '', query).strip()

        # Remove trailing "?" and extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        # Limit length
        if len(query) > 200:
            query = query[:200]

        return query

    # ── Causal Chain Building ─────────────────────────────────────────────────

    def _build_causal_chain(
        self,
        target_event: str,
        triples: List[CausalTriple],
    ) -> Optional[CausalChain]:
        """
        Build a causal chain from triples, linking causes to the target event.

        Algorithm:
          1. Find triples where effect matches (or is similar to) target_event
          2. Recursively find causes of those causes
          3. Build ordered chain
        """
        if not triples:
            return None

        chain_id = f"chain_{hash(target_event) % 10000:04d}"
        events = [target_event]
        chain_triples: List[CausalTriple] = []

        # Direct causes
        direct = [t for t in triples if self._events_similar(t.effect, target_event)]
        direct.sort(key=lambda x: x.confidence, reverse=True)

        for t in direct[:5]:
            events.insert(0, t.cause)
            chain_triples.append(t)

        # Extend chain: causes of causes
        current_causes = [t.cause for t in direct]
        seen_causes = set(current_causes)
        depth = 1

        while depth < self.max_hops and current_causes:
            next_causes = []
            for cause in current_causes:
                sub_causes = [t for t in triples if self._events_similar(t.effect, cause) and t.cause not in seen_causes]
                for t in sub_causes[:2]:
                    events.insert(0, t.cause)
                    chain_triples.insert(0, t)
                    next_causes.append(t.cause)
                    seen_causes.add(t.cause)

            current_causes = next_causes
            depth += 1

        return CausalChain(
            chain_id=chain_id,
            events=list(dict.fromkeys(events)),  # dedupe preserving order
            triples=chain_triples,
            depth=depth,
            confidence=float(min(1.0, len(chain_triples) * 0.2 + 0.2)),
        )

    def _events_similar(self, a: str, b: str) -> bool:
        """Check if two event strings refer to the same event."""
        a_lower = a.lower().strip()
        b_lower = b.lower().strip()
        if a_lower == b_lower:
            return True
        # Simple word overlap
        a_words = set(a_lower.split())
        b_words = set(b_lower.split())
        overlap = len(a_words & b_words)
        return overlap >= min(2, len(a_words), len(b_words))

    # ── Explanation Synthesis ─────────────────────────────────────────────────

    def _synthesize_explanation(
        self,
        query: str,
        target_event: str,
        chain: Optional[CausalChain],
        triples: List[CausalTriple],
    ) -> str:
        """Generate a coherent causal explanation using LLM."""
        client = self.extractor._get_client()

        direct_causes = [t.cause for t in triples if self._events_similar(t.effect, target_event)][:5]
        all_causes = [t.cause for t in triples[:10]]

        context = {
            "query": query,
            "target_event": target_event,
            "direct_causes": direct_causes,
            "all_causes": all_causes,
            "chain_depth": chain.depth if chain else 0,
        }

        prompt = f"""You are a film analyst explaining causal relationships in a movie.

Question: {query}

The target event: "{target_event}"

Direct causes identified:
{chr(10).join(f'- {c}' for c in direct_causes)}

Other causal factors:
{chr(10).join(f'- {t.cause} → {t.effect}' for t in triples[:5])}

Write a clear, engaging explanation (3-5 sentences) that:
1. States the immediate cause of the target event
2. Provides 1-2 upstream causes when known
3. Connects causes to the narrative context

Keep it factual and based on the evidence above.
"""
        try:
            explanation = client.generate_content(
                model=None,
                contents=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
            )
            return explanation.strip() if explanation else "Limited causal information available."
        except Exception as e:
            logger.warning(f"Explanation synthesis failed: {e}")
            if direct_causes:
                return f"{target_event} is primarily caused by: {', '.join(direct_causes[:3])}."
            return "Causal chain could not be fully reconstructed."

    # ── Reasoning Trace ───────────────────────────────────────────────────────

    def _build_reasoning_trace(
        self,
        query: str,
        target_event: str,
        chain: Optional[CausalChain],
        triples: List[CausalTriple],
    ) -> str:
        """Build a step-by-step reasoning trace."""
        lines = [
            f"Query: {query}",
            f"Target event: {target_event}",
            f"Causal triples found: {len(triples)}",
        ]
        if chain:
            lines.append(f"Causal chain depth: {chain.depth}")
            lines.append("Causal chain:")
            for i, t in enumerate(chain.triples[:5]):
                lines.append(f"  {i + 1}. {t.cause} --[{t.relation}]--> {t.effect} (conf={t.confidence:.2f})")
        elif triples:
            lines.append("No chain constructed. Direct causes:")
            for t in triples[:3]:
                lines.append(f"  - {t.cause} → {t.effect}")
        else:
            lines.append("No causal relationships found in retrieved context.")

        return "\n".join(lines)

    # ── Neo4j Integration ─────────────────────────────────────────────────────

    def query_graph_causes(
        self,
        movie_id: str,
        target_event: str,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Query Neo4j for causal paths leading to target_event.

        Uses graph traversal to find cause-effect chains.
        """
        if self.neo4j is None:
            return []

        try:
            # Search for events matching the target
            hits = self.neo4j.search(
                f"event:{target_event}",
                top_k=20,
                filters={"movie_id": movie_id},
            )

            causal_paths = []
            for hit in hits:
                # Get incoming causal edges (CAUSES relationship)
                if hasattr(self.neo4j, '_search_character_relationship_remote'):
                    paths = self.neo4j._search_character_relationship_remote(
                        movie_id, "CAUSES", max_hops=max_hops
                    )
                    causal_paths.extend(paths)

            return causal_paths

        except Exception as e:
            logger.warning(f"Neo4j causal query failed: {e}")
            return []

    def __repr__(self) -> str:
        return f"CausalReasoner(max_hops={self.max_hops}, has_neo4j={self.neo4j is not None})"
