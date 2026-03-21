#!/usr/bin/env python3
"""
neo4j_graph_queries.py
=======================
Multi-hop graph reasoning over the Neo4j narrative graph.
Implements L5 narrative traversal queries:

  Q1: Co-appearance — characters appearing in the same scene across movies
  Q2: Character timeline — ordered scenes for a character in a movie
  Q3: Narrative proximity — scenes within N hops of a given scene
  Q4: Cross-movie character overlap — movies sharing common character names
  Q5: Scene before/after — scenes that causally precede or follow a target

Usage:
    python scripts/neo4j_graph_queries.py --demo
    python scripts/neo4j_graph_queries.py --query "Find all scenes with Forrest Gump"
    python scripts/neo4j_graph_queries.py --query "Characters in The Godfather dining scene"
"""
from __future__ import annotations

import argparse
import json
import logging
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
logger = logging.getLogger("GraphQuery")

NEO4J_URI = "bolt://localhost:7688"
NEO4J_USER = "neo4j"
NEO4J_PASS = "movierag123"


def get_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


# ── Graph Query Functions ─────────────────────────────────────────────────────

def q1_character_coappearance(driver, character_name: str, limit: int = 20) -> list[dict]:
    """Find all scenes where a character appears, with co-appearing characters."""
    with driver.session() as s:
        result = s.run("""
            MATCH (c:Character {name: $name})-[:APPEARS_IN]->(scene:GraphNode)
            OPTIONAL MATCH (other:Character)-[:APPEARS_IN]->(scene)
            WHERE other.name <> $name
            RETURN scene.chunk_id AS chunk_id,
                   scene.movie_id AS movie_id,
                   scene.title AS title,
                   scene.start_seconds AS start_s,
                   collect(DISTINCT other.name) AS co_characters
            ORDER BY scene.movie_id, scene.start_seconds
            LIMIT $limit
        """, name=character_name, limit=limit)
        return [dict(r) for r in result]


def q2_character_timeline(driver, character_name: str, movie_id: str = None) -> list[dict]:
    """Ordered scene timeline for a character."""
    with driver.session() as s:
        if movie_id:
            result = s.run("""
                MATCH (c:Character {name: $name})-[:APPEARS_IN]->(scene:GraphNode {movie_id: $movie_id})
                RETURN scene.chunk_id AS chunk_id,
                       scene.movie_id AS movie_id,
                       scene.start_seconds AS start_s,
                       scene.end_seconds AS end_s,
                       scene.description AS description
                ORDER BY scene.start_seconds
            """, name=character_name, movie_id=movie_id)
        else:
            result = s.run("""
                MATCH (c:Character {name: $name})-[:APPEARS_IN]->(scene:GraphNode)
                RETURN scene.chunk_id AS chunk_id,
                       scene.movie_id AS movie_id,
                       scene.start_seconds AS start_s,
                       scene.end_seconds AS end_s,
                       scene.description AS description
                ORDER BY scene.movie_id, scene.start_seconds
                LIMIT 50
            """, name=character_name)
        return [dict(r) for r in result]


def q3_narrative_neighbors(driver, chunk_id: str, hops: int = 2) -> list[dict]:
    """Find scenes within N temporal hops of a target scene."""
    with driver.session() as s:
        result = s.run("""
            MATCH (target:GraphNode {chunk_id: $chunk_id})
            MATCH (neighbor:GraphNode {movie_id: target.movie_id})
            WHERE abs(neighbor.start_seconds - target.start_seconds) <= $window
              AND neighbor.chunk_id <> $chunk_id
            RETURN neighbor.chunk_id AS chunk_id,
                   neighbor.start_seconds AS start_s,
                   neighbor.description AS description,
                   abs(neighbor.start_seconds - target.start_seconds) AS distance
            ORDER BY distance
            LIMIT 10
        """, chunk_id=chunk_id, window=hops * 30)
        return [dict(r) for r in result]


def q4_cross_movie_character_overlap(driver, limit: int = 20) -> list[dict]:
    """Find characters that appear in 2+ movies (cross-movie overlap)."""
    with driver.session() as s:
        result = s.run("""
            MATCH (c:Character)-[:APPEARS_IN]->(scene:GraphNode)
            WITH c.name AS name, collect(DISTINCT scene.movie_id) AS movies
            WHERE size(movies) >= 2
            RETURN name, movies, size(movies) AS movie_count
            ORDER BY movie_count DESC
            LIMIT $limit
        """, limit=limit)
        return [dict(r) for r in result]


def q5_movies_by_character(driver, character_name: str) -> list[dict]:
    """Find all movies and scene counts for a character."""
    with driver.session() as s:
        result = s.run("""
            MATCH (c:Character {name: $name})-[:APPEARS_IN]->(scene:GraphNode)
            RETURN scene.movie_id AS movie_id,
                   scene.title AS title,
                   count(scene) AS scene_count
            ORDER BY scene_count DESC
        """, name=character_name)
        return [dict(r) for r in result]


def q6_most_connected_scenes(driver, movie_id: str = None, limit: int = 10) -> list[dict]:
    """Find scenes with the most character co-appearances."""
    with driver.session() as s:
        if movie_id:
            result = s.run("""
                MATCH (c:Character)-[:APPEARS_IN]->(scene:GraphNode {movie_id: $movie_id})
                WITH scene, count(c) AS char_count
                RETURN scene.chunk_id AS chunk_id,
                       scene.movie_id AS movie_id,
                       scene.start_seconds AS start_s,
                       scene.description AS description,
                       char_count
                ORDER BY char_count DESC
                LIMIT $limit
            """, movie_id=movie_id, limit=limit)
        else:
            result = s.run("""
                MATCH (c:Character)-[:APPEARS_IN]->(scene:GraphNode)
                WITH scene, count(c) AS char_count
                RETURN scene.chunk_id AS chunk_id,
                       scene.movie_id AS movie_id,
                       scene.start_seconds AS start_s,
                       scene.description AS description,
                       char_count
                ORDER BY char_count DESC
                LIMIT $limit
            """, limit=limit)
        return [dict(r) for r in result]


# ── Demo ─────────────────────────────────────────────────────────────────────

def run_demo(driver):
    print("\n" + "=" * 70)
    print("  VideoSceneRAG — Multi-hop Graph Reasoning Demo (Neo4j L5)")
    print("=" * 70)

    # Q1: Character co-appearance
    print("\n[Q1] Character co-appearance: 'Forrest Gump'")
    scenes = q1_character_coappearance(driver, "Forrest Gump", limit=5)
    if not scenes:
        # Try other known characters
        with driver.session() as s:
            r = s.run("MATCH (c:Character) RETURN c.name AS name LIMIT 5")
            sample_chars = [rec["name"] for rec in r]
        print(f"  (no 'Forrest Gump' found — trying: {sample_chars[:3]})")
        if sample_chars:
            scenes = q1_character_coappearance(driver, sample_chars[0], limit=5)
    for sc in scenes[:3]:
        co = ", ".join(sc.get("co_characters", [])[:3])
        print(f"  [{sc.get('movie_id','')}] {sc.get('start_s',0):.0f}s — co-chars: {co}")

    # Q4: Cross-movie character overlap
    print("\n[Q4] Characters appearing in multiple movies:")
    overlaps = q4_cross_movie_character_overlap(driver, limit=5)
    for o in overlaps:
        print(f"  '{o['name']}' → {o['movie_count']} movies: {o['movies'][:3]}")

    # Q6: Most connected scenes (most characters)
    print("\n[Q6] Top scenes by character count (all movies):")
    connected = q6_most_connected_scenes(driver, limit=5)
    for sc in connected:
        print(f"  [{sc.get('movie_id','')}] {sc.get('start_s',0):.0f}s — {sc.get('char_count',0)} chars")

    # Graph stats
    print("\n[Stats] Neo4j graph summary:")
    with driver.session() as s:
        r = s.run("MATCH (n) RETURN labels(n)[0] AS lbl, count(*) AS cnt ORDER BY cnt DESC")
        for rec in r:
            print(f"  {rec['lbl']}: {rec['cnt']:,}")
        r2 = s.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC LIMIT 5")
        print("  Relationships:")
        for rec in r2:
            print(f"    {rec['rel']}: {rec['cnt']:,}")

    print("=" * 70)


# ── LLM-powered graph query ───────────────────────────────────────────────────

def llm_graph_query(driver, natural_query: str) -> str:
    """Answer a natural language query using graph traversal + Groq LLM."""
    import os
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "No GROQ_API_KEY available"

    client = Groq(api_key=api_key)

    # Step 1: Extract entities from query
    entity_resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": (
                f"Extract character names and movie titles from this query: '{natural_query}'\n"
                'Return JSON: {"characters": [...], "movies": [...]}'
            )
        }],
        temperature=0,
        max_tokens=150,
    )
    import re, json as _json
    text = entity_resp.choices[0].message.content
    m = re.search(r'\{.*\}', text, re.DOTALL)
    entities = {"characters": [], "movies": []}
    if m:
        try:
            entities = _json.loads(m.group())
        except Exception:
            pass

    # Step 2: Graph traversal
    evidence = []
    for char in entities.get("characters", [])[:2]:
        scenes = q1_character_coappearance(driver, char, limit=5)
        if scenes:
            evidence.append(f"Character '{char}' appears in: " +
                           "; ".join(f"{s['movie_id']} @{s.get('start_s',0):.0f}s (with {', '.join(s.get('co_characters',[])[:2])})"
                                    for s in scenes[:3]))

    if not evidence:
        connected = q6_most_connected_scenes(driver, limit=3)
        evidence.append("Top scenes: " +
                       "; ".join(f"{s['movie_id']} @{s.get('start_s',0):.0f}s ({s.get('char_count',0)} chars)"
                                for s in connected))

    # Step 3: Generate answer
    context = "\n".join(evidence) if evidence else "No matching graph data found."
    answer_resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": (
                f"Query: {natural_query}\n\n"
                f"Graph evidence:\n{context}\n\n"
                "Answer the query using the graph evidence. Be concise."
            )
        }],
        temperature=0.2,
        max_tokens=300,
    )
    return answer_resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    try:
        driver = get_driver()
        driver.verify_connectivity()
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        return

    if args.demo:
        run_demo(driver)

    if args.query:
        print(f"\nQuery: {args.query}")
        answer = llm_graph_query(driver, args.query)
        print(f"Answer: {answer}")

    driver.close()


if __name__ == "__main__":
    main()
