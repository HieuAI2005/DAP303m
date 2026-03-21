#!/usr/bin/env python3
"""
answer_generator.py
====================
End-to-end VideoSceneRAG answer generation pipeline.
Connects retrieval (FAISS L3 Knowledge + L0 Visual) to Groq LLM for grounded answers.

Pipeline:
  1. Query → SentenceTransformer encode
  2. L3 Knowledge Index search (top-k chunks)
  3. Optional: L0 Visual search (frame similarity)
  4. Optional: Neo4j graph traversal (character/narrative context)
  5. Evidence aggregation → Groq LLM → Grounded answer with timestamps

Usage:
    python scripts/answer_generator.py --query "Who is the main character in Titanic?"
    python scripts/answer_generator.py --query "Find scenes where someone is driving a car"
    python scripts/answer_generator.py --interactive
    python scripts/answer_generator.py --benchmark  # Run on 20 sample QA queries
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AnswerGen")

INDEX_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "indexes"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "pipeline_output" / "benchmark"


# ── Retrieval ─────────────────────────────────────────────────────────────────

class VideoSceneRetriever:
    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info("Loading indexes...")
        self.knowledge_idx = faiss.read_index(str(INDEX_DIR / "knowledge_videorag.faiss"))
        self.knowledge_meta = json.loads((INDEX_DIR / "knowledge_videorag_meta.json").read_text())
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"  Knowledge index: {self.knowledge_idx.ntotal} vectors")

    def search(self, query: str, k: int = 10, source_filter: str = None) -> list[dict]:
        """Search knowledge index, return top-k chunks with metadata."""
        import numpy as np
        emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.knowledge_idx.search(emb, k * 2)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.knowledge_meta):
                continue
            m = self.knowledge_meta[idx].copy()
            m["score"] = float(score)
            if source_filter and m.get("source") != source_filter:
                continue
            results.append(m)
            if len(results) >= k:
                break
        return results

    def get_chunk_detail(self, chunk_id: str) -> dict:
        """Load full chunk from all_chunks.json by chunk_id."""
        chunks_path = PROJECT_ROOT / "data" / "pipeline_output" / "videorag_chunks" / "all_chunks.json"
        if not hasattr(self, "_chunks_cache"):
            self._chunks_cache = {
                c["chunk_id"]: c
                for c in json.loads(chunks_path.read_text())
            }
        return self._chunks_cache.get(chunk_id, {})


# ── Answer Generation ─────────────────────────────────────────────────────────

class AnswerGenerator:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def generate(self, query: str, evidence_chunks: list[dict]) -> dict:
        """Generate a grounded answer from retrieved chunks."""
        # Build context from top chunks
        context_parts = []
        for i, c in enumerate(evidence_chunks[:5]):
            detail = c.get("_detail", {})
            parts = [f"[Scene {i+1}]"]
            parts.append(f"Movie: {c.get('title', c.get('movie_id', '?'))}")
            if detail.get("start_seconds") is not None:
                parts.append(f"Time: {detail['start_seconds']:.0f}s–{detail.get('end_seconds', '?'):.0f}s")
            if detail.get("description"):
                parts.append(f"Description: {detail['description']}")
            if detail.get("dialogue_text") and detail["dialogue_text"] != "[NO_DIALOGUE]":
                parts.append(f"Dialogue: {detail['dialogue_text'][:200]}")
            if detail.get("characters"):
                parts.append(f"Characters: {', '.join(detail['characters'][:5])}")
            context_parts.append(" | ".join(parts))

        context = "\n".join(context_parts) if context_parts else "No relevant scenes found."

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are VideoSceneRAG, an expert video scene retrieval assistant. "
                        "Answer questions about movies using the retrieved scene evidence. "
                        "Always cite specific scenes with timestamps when available. "
                        "Be concise and factual."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Retrieved scene evidence:\n{context}\n\n"
                        "Answer the query based on the evidence. Cite timestamps."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=400,
        )
        answer = resp.choices[0].message.content.strip()

        # Extract cited scenes
        citations = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "movie_id": c.get("movie_id", ""),
                "title": c.get("title", ""),
                "score": round(c.get("score", 0), 4),
                "start_seconds": c.get("_detail", {}).get("start_seconds"),
            }
            for c in evidence_chunks[:5]
        ]

        return {"query": query, "answer": answer, "citations": citations}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_query(retriever: VideoSceneRetriever, generator: AnswerGenerator,
              query: str, k: int = 10, verbose: bool = True) -> dict:
    # Retrieve
    results = retriever.search(query, k=k)
    # Enrich with full chunk details
    for r in results:
        r["_detail"] = retriever.get_chunk_detail(r.get("chunk_id", ""))

    # Generate
    output = generator.generate(query, results)

    if verbose:
        print(f"\n{'='*65}")
        print(f"Query: {query}")
        print(f"{'='*65}")
        print(f"Answer:\n{output['answer']}")
        print(f"\nTop citations:")
        for c in output["citations"][:3]:
            ts = f"@{c['start_seconds']:.0f}s" if c.get('start_seconds') else ""
            print(f"  [{c['movie_id']}] {c['title']} {ts} (score={c['score']:.3f})")
        print()

    return output


def run_benchmark(retriever: VideoSceneRetriever, generator: AnswerGenerator, n: int = 20):
    """Run on a sample of QA benchmark queries and show answers."""
    qa_path = BENCHMARK_DIR / "qa_benchmark.json"
    if not qa_path.exists():
        logger.error("qa_benchmark.json not found — run build_qa_benchmark.py first")
        return

    import random
    queries = json.loads(qa_path.read_text())["queries"]
    # Sample from T1 (scene description) and T2 (character)
    t1 = [q for q in queries if q["task"] == "scene_description"][:n // 2]
    t2 = [q for q in queries if q["task"] == "character_retrieval"][:n // 2]
    sample = t1 + t2
    random.shuffle(sample)

    results = []
    for q in sample[:n]:
        out = run_query(retriever, generator, q["query_text"], verbose=True)
        out["expected_movie_id"] = q.get("expected_movie_id", "")
        out["task"] = q["task"]
        results.append(out)

    # Save
    out_path = BENCHMARK_DIR / "e2e_answers_sample.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(results)} answers → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    retriever = VideoSceneRetriever()
    generator = AnswerGenerator()

    if args.query:
        run_query(retriever, generator, args.query)

    elif args.interactive:
        print("VideoSceneRAG Interactive Query (Ctrl+C to exit)")
        while True:
            try:
                q = input("\nQuery> ").strip()
                if q:
                    run_query(retriever, generator, q)
            except KeyboardInterrupt:
                print("\nBye!")
                break

    elif args.benchmark:
        run_benchmark(retriever, generator, n=args.n)

    else:
        # Default demo queries
        demo_queries = [
            "Who is the main character in Forrest Gump and what is the movie about?",
            "Find scenes where characters are in a restaurant or dining",
            "What happens in The Godfather when Don Corleone is shot?",
        ]
        for q in demo_queries:
            run_query(retriever, generator, q)


if __name__ == "__main__":
    main()
