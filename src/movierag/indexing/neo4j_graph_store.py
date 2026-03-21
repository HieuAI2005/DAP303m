"""
Neo4j-backed graph sync and query utilities for MovieRAG.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

from preprocess_data.config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Sync graph artifacts into Neo4j and expose graph-aware search helpers."""

    _CYTHER_BLOCKLIST = ("create ", "merge ", "delete ", "set ", "remove ", "drop ", "load csv", "foreach ")

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        cfg = Cfg.get_neo4j_config()
        self.uri = uri or cfg["uri"]
        self.user = user or cfg["user"]
        self.password = password or cfg["password"]
        self.database = database or cfg["database"]
        self.driver = None

        if not GraphDatabase:
            logger.warning("Neo4j Python driver is not installed.")
            return

        self._connect(max_attempts=1, delay_seconds=0.0)

    def _connect(self, max_attempts: int = 1, delay_seconds: float = 0.0) -> bool:
        last_error = None
        for attempt in range(max_attempts):
            try:
                driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                driver.verify_connectivity()
                self.driver = driver
                return True
            except Exception as exc:
                last_error = exc
                self.driver = None
                if attempt < max_attempts - 1 and delay_seconds > 0:
                    time.sleep(delay_seconds)
        try:
            raise last_error
        except Exception as exc:
            logger.warning("Neo4j connection unavailable at %s: %s", self.uri, exc)
            return False

    @property
    def is_available(self) -> bool:
        return self.driver is not None

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def sync_movie(self, movie_id: str) -> Dict[str, Any]:
        """Replace and repopulate all graph data for one movie in Neo4j."""
        stats = {
            "movie_id": movie_id,
            "synced": False,
            "movie_nodes": 0,
            "temporal_chunk_nodes": 0,
            "script_scene_nodes": 0,
            "script_subscene_nodes": 0,
            "kg_nodes": 0,
            "relationships": 0,
        }
        if not self.driver:
            self._connect(max_attempts=6, delay_seconds=3.0)
        if not self.driver:
            stats["reason"] = "neo4j_unavailable"
            return stats

        chunks = self._load_json(Cfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json")
        subscenes = self._load_json(
            Cfg.get_script_subscenes_dir() / f"{movie_id}_script_subscenes.json"
        )
        kg_graph = self._load_graph(Cfg.get_index_dir() / f"{movie_id}_kg.graphml")

        if not chunks and not subscenes and kg_graph is None:
            stats["reason"] = "no_graph_artifacts"
            return stats

        canonical_maps = self._build_canonical_maps(
            movie_id, chunks or [], subscenes or [], kg_graph=kg_graph
        )
        payload = self._build_temporal_payload(
            movie_id, chunks or [], subscenes or [], canonical_maps=canonical_maps
        )
        if kg_graph is not None:
            kg_payload = self._build_kg_payload(
                movie_id,
                kg_graph,
                {chunk.get("chunk_id", "") for chunk in chunks or [] if chunk.get("chunk_id")},
                canonical_maps=canonical_maps,
            )
            self._merge_payload(payload, kg_payload)

        with self.driver.session(database=self.database) as session:
            self._ensure_constraints(session)
            execute_write = getattr(session, "execute_write", None)
            if execute_write:
                execute_write(self._delete_movie_subgraph, movie_id)
            else:
                session.write_transaction(self._delete_movie_subgraph, movie_id)
            self._write_payload(session, payload)

        stats["synced"] = True
        stats["movie_nodes"] = len(payload["nodes_by_label"].get("Movie", []))
        stats["temporal_chunk_nodes"] = len(
            payload["nodes_by_label"].get("TemporalChunk", [])
        )
        stats["script_scene_nodes"] = len(
            payload["nodes_by_label"].get("ScriptScene", [])
        )
        stats["script_subscene_nodes"] = len(
            payload["nodes_by_label"].get("ScriptSubscene", [])
        )
        stats["kg_nodes"] = len(payload["nodes_by_label"].get("KGScene", [])) + len(
            payload["nodes_by_label"].get("Location", [])
        ) + len(payload["nodes_by_label"].get("Object", [])) + len(
            payload["nodes_by_label"].get("Entity", [])
        )
        stats["relationships"] = sum(
            len(rows) for rows in payload["relationships_by_type"].values()
        )
        return stats

    def search(
        self, query: str, movie_id: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the graph using Neo4j if available, otherwise use local fallback."""
        if self._looks_like_cypher(query):
            return self._run_readonly_cypher(query, movie_id=movie_id, limit=limit)

        merged_hits: Dict[str, Dict[str, Any]] = {}
        query_request = self._classify_graph_query(query)
        structured_hits: Dict[str, Dict[str, Any]] = {}

        for hit in self._search_structured_local(query_request, movie_id=movie_id, limit=limit):
            self._merge_ranked_hit(structured_hits, hit)
            self._merge_ranked_hit(merged_hits, hit)

        if self.driver:
            try:
                for hit in self._search_structured_remote(
                    query_request, movie_id=movie_id, limit=limit
                ):
                    self._merge_ranked_hit(structured_hits, hit)
                    self._merge_ranked_hit(merged_hits, hit)
            except Exception as exc:
                logger.warning("Neo4j graph search failed, falling back locally: %s", exc)

        if structured_hits:
            ranked_structured = sorted(
                structured_hits.values(),
                key=lambda item: item.get("score", 0.0),
                reverse=True,
            )
            if len(ranked_structured) >= limit:
                return ranked_structured[:limit]

        if self.driver:
            try:
                remote_hits = self._search_remote(
                    query, movie_id=movie_id, limit=max(limit * 3, 12)
                )
                for hit in remote_hits:
                    self._merge_ranked_hit(merged_hits, hit)
            except Exception as exc:
                logger.warning("Neo4j graph lexical search failed: %s", exc)

        local_hits = self._search_local(query, movie_id=movie_id, limit=max(limit * 3, 12))
        for hit in local_hits:
            self._merge_ranked_hit(merged_hits, hit)

        ranked_hits = sorted(
            merged_hits.values(), key=lambda item: item.get("score", 0.0), reverse=True
        )
        return ranked_hits[:limit]

    def search_as_documents(
        self, query: str, movie_id: Optional[str] = None, limit: int = 4
    ) -> List[Dict[str, Any]]:
        """Convert graph hits into pseudo-documents that the text generator can use."""
        hits = self.search(query, movie_id=movie_id, limit=limit)
        documents = []
        for hit in hits:
            node_title = hit.get("title") or hit.get("node_id", "graph_node")
            node_type = hit.get("node_type", "GraphNode")
            heading = hit.get("heading", "")
            location = hit.get("location", "")
            character_names = [name for name in (hit.get("character_names") or []) if name]
            body = (hit.get("body", "") or "").strip()
            if len(body) > 360:
                body = f"{body[:357]}..."
            neighbors = hit.get("neighbors") or []
            neighbor_text = ", ".join(
                f"{n.get('relation', 'RELATED_TO')} -> {n.get('title', n.get('node_id', ''))}"
                for n in neighbors[:5]
                if n.get("title") or n.get("node_id")
            )
            parts = [
                f"Graph node: {node_title}",
                f"Type: {node_type}",
            ]
            if heading:
                parts.append(f"Heading: {heading}")
            if location:
                parts.append(f"Location: {location}")
            if character_names:
                parts.append(f"Characters: {', '.join(character_names)}")
            if body:
                parts.append(f"Evidence: {body}")
            if neighbor_text:
                parts.append(f"Links: {neighbor_text}")

            documents.append(
                {
                    "movie_id": hit.get("movie_id", movie_id or "unknown"),
                    "clip_id": hit.get("node_id", node_title),
                    "text": "\n".join(parts),
                    "score": float(hit.get("score", 0.0)),
                    "metadata": {
                        "category": "moviegraph",
                        "title": hit.get("movie_title")
                        or hit.get("movie_id", movie_id or "unknown"),
                        "node_type": node_type,
                        "graph_source": hit.get("source", "neo4j"),
                        "graph_node_id": hit.get("node_id"),
                        "graph_heading": heading,
                        "graph_location": location,
                        "character_names": character_names,
                        "start_time": hit.get("start_time", ""),
                        "end_time": hit.get("end_time", ""),
                        "chunk_id": hit.get("chunk_id", ""),
                        "graph_question_type": hit.get("question_type", ""),
                    },
                }
            )
        return documents

    @staticmethod
    def format_hits(hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "[no graph hits]"

        lines = []
        for idx, hit in enumerate(hits, start=1):
            title = hit.get("title") or hit.get("node_id", "graph_node")
            node_type = hit.get("node_type", "GraphNode")
            heading = hit.get("heading", "")
            location = hit.get("location", "")
            timerange = ""
            if hit.get("start_time") or hit.get("end_time"):
                timerange = f" | {hit.get('start_time', '')}->{hit.get('end_time', '')}"
            line = f"[{idx}] {node_type} | {title}"
            if heading:
                line += f" | {heading}"
            if location:
                line += f" | {location}"
            line += timerange
            body = (hit.get("body", "") or "").strip()
            if body:
                line += f"\n  {body[:220]}"
            neighbors = hit.get("neighbors") or []
            if neighbors:
                links = ", ".join(
                    f"{n.get('relation', 'RELATED_TO')} -> {n.get('title', n.get('node_id', ''))}"
                    for n in neighbors[:4]
                )
                line += f"\n  Links: {links}"
            lines.append(line)
        return "\n".join(lines)

    def _search_remote(
        self, query: str, movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        tokens = self._tokenize(query)
        if not tokens:
            return []

        candidate_limit = max(limit * 8, 24)
        cypher = """
        MATCH (n:GraphNode)
        WHERE ($movie_id IS NULL OR n.movie_id = $movie_id)
          AND any(token IN $tokens WHERE
              toLower(coalesce(n.name, '')) CONTAINS token OR
              toLower(coalesce(n.heading, '')) CONTAINS token OR
              toLower(coalesce(n.location, '')) CONTAINS token OR
              toLower(coalesce(n.script_primary_heading, '')) CONTAINS token OR
              toLower(coalesce(n.scene_label, '')) CONTAINS token OR
              toLower(coalesce(n.description, '')) CONTAINS token OR
              toLower(coalesce(n.dialogue_text, '')) CONTAINS token OR
              toLower(coalesce(n.text, '')) CONTAINS token
          )
        OPTIONAL MATCH (n)-[r]-(m:GraphNode)
        WHERE $movie_id IS NULL OR m.movie_id = $movie_id
        WITH n, collect(
            DISTINCT {
                relation: type(r),
                node_id: m.id,
                title: coalesce(m.name, m.heading, m.scene_label, m.id),
                node_type: m.node_type
            }
        )[0..8] AS neighbors
        RETURN
            n.id AS node_id,
            labels(n) AS labels,
            n.node_type AS node_type,
            n.movie_id AS movie_id,
            coalesce(n.title, n.name, n.heading, n.scene_label, n.id) AS title,
            coalesce(n.heading, n.script_primary_heading, '') AS heading,
            coalesce(n.location, n.script_location, '') AS location,
            coalesce(n.time_of_day, n.script_time_of_day, '') AS time_of_day,
            coalesce(n.scene_label, '') AS scene_label,
            coalesce(n.start_time, '') AS start_time,
            coalesce(n.end_time, '') AS end_time,
            coalesce(n.chunk_id, '') AS chunk_id,
            coalesce(n.indexable, false) AS indexable,
            coalesce(n.is_canonical_subscene, false) AS is_canonical_subscene,
            coalesce(n.description, n.dialogue_text, n.text, '') AS body,
            neighbors
        LIMIT $candidate_limit
        """
        with self.driver.session(database=self.database) as session:
            rows = [
                dict(record)
                for record in session.run(
                    cypher,
                    movie_id=movie_id,
                    tokens=tokens,
                    candidate_limit=candidate_limit,
                )
            ]

        reranked = self._rerank_hits(query, rows)
        for hit in reranked:
            hit["source"] = "neo4j"
        return reranked[:limit]

    def _search_local(
        self, query: str, movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        tokens = self._tokenize(query)
        if not tokens:
            return []

        candidates = self._build_local_candidates(movie_id)
        reranked = self._rerank_hits(query, candidates)
        for hit in reranked:
            hit["source"] = "local"
        return reranked[:limit]

    def _search_structured_remote(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        if not self.driver or query_request.get("kind") == "generic":
            return []

        kind = query_request.get("kind")
        if kind == "scene_characters":
            return self._search_scene_characters_remote(
                query_request, movie_id=movie_id, limit=limit
            )
        if kind == "scene_transition":
            return self._search_scene_transition_remote(
                query_request, movie_id=movie_id, limit=limit
            )
        if kind == "scene_location":
            return self._search_scene_location_remote(
                query_request, movie_id=movie_id, limit=limit
            )
        if kind == "character_relationship":
            return self._search_character_relationship_remote(
                query_request, movie_id=movie_id, limit=limit
            )
        return []

    def _search_structured_local(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        kind = query_request.get("kind")
        if kind == "scene_characters":
            return self._search_scene_characters_local(
                query_request, movie_id=movie_id, limit=limit
            )
        if kind == "scene_transition":
            return self._search_scene_transition_local(
                query_request, movie_id=movie_id, limit=limit
            )
        if kind == "scene_location":
            return self._search_scene_location_local(
                query_request, movie_id=movie_id, limit=limit
            )
        if kind == "character_relationship":
            return self._search_character_relationship_local(
                query_request, movie_id=movie_id, limit=limit
            )
        return []

    def _run_readonly_cypher(
        self, query: str, movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        if not self.driver:
            return [{"error": "neo4j_unavailable", "query": query}]

        lowered = f" {query.strip().lower()} "
        if any(token in lowered for token in self._CYTHER_BLOCKLIST):
            return [{"error": "write_cypher_blocked", "query": query}]

        with self.driver.session(database=self.database) as session:
            records = session.run(query, movie_id=movie_id, limit=limit)
            rows = []
            for idx, record in enumerate(records):
                row = dict(record)
                row.setdefault("node_id", f"cypher_row_{idx}")
                row.setdefault("node_type", "CypherResult")
                row.setdefault("title", row.get("node_id", f"row_{idx}"))
                row.setdefault("movie_id", movie_id or row.get("movie_id", "unknown"))
                row.setdefault("neighbors", [])
                row.setdefault("body", json.dumps(row, ensure_ascii=False))
                row["score"] = 1.0
                rows.append(row)
            return rows[:limit]

    def _search_scene_characters_remote(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        anchor_tokens = query_request.get("anchor_tokens", [])
        if not self.driver or not anchor_tokens:
            return []

        cypher = """
        MATCH (scene:GraphNode)
        WHERE ($movie_id IS NULL OR scene.movie_id = $movie_id)
          AND scene.node_type IN ['ScriptSubscene', 'ScriptScene', 'TemporalChunk']
          AND all(token IN $anchor_tokens WHERE
              toLower(coalesce(scene.heading, '')) CONTAINS token OR
              toLower(coalesce(scene.location, '')) CONTAINS token OR
              toLower(coalesce(scene.time_of_day, '')) CONTAINS token OR
              toLower(coalesce(scene.script_primary_heading, '')) CONTAINS token OR
              toLower(coalesce(scene.script_location, '')) CONTAINS token OR
              toLower(coalesce(scene.scene_label, '')) CONTAINS token
          )
        OPTIONAL MATCH (char:GraphNode)-[r]-(scene)
        WHERE char.node_type = 'Character'
        WITH scene, collect(DISTINCT char.name)[0..12] AS characters
        RETURN
            scene.id AS node_id,
            scene.node_type AS node_type,
            scene.movie_id AS movie_id,
            coalesce(scene.title, scene.name, scene.heading, scene.scene_label, scene.id) AS title,
            coalesce(scene.heading, scene.script_primary_heading, '') AS heading,
            coalesce(scene.location, scene.script_location, '') AS location,
            coalesce(scene.time_of_day, scene.script_time_of_day, '') AS time_of_day,
            coalesce(scene.start_time, '') AS start_time,
            coalesce(scene.end_time, '') AS end_time,
            coalesce(scene.chunk_id, '') AS chunk_id,
            characters
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            rows = [
                dict(record)
                for record in session.run(
                    cypher,
                    movie_id=movie_id,
                    anchor_tokens=anchor_tokens,
                    limit=max(limit * 3, 6),
                )
            ]

        hits = []
        for row in rows:
            characters = [name for name in row.get("characters", []) if name]
            row["character_names"] = characters
            row["body"] = (
                f"Characters present: {', '.join(characters)}"
                if characters
                else "Characters present: unknown"
            )
            row["score"] = 8.5 + min(1.0, 0.1 * len(characters))
            row["source"] = "graph_structured_remote"
            row["question_type"] = "scene_characters"
            row["neighbors"] = [
                {
                    "relation": "APPEARS_IN",
                    "title": name,
                    "node_id": name,
                    "node_type": "Character",
                }
                for name in characters[:8]
            ]
            hits.append(row)
        return hits[:limit]

    def _search_scene_transition_remote(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        anchor_tokens = query_request.get("anchor_tokens", [])
        if not self.driver or not anchor_tokens:
            return []

        direction = "FOLLOWED_BY" if query_request.get("direction") == "after" else "PRECEDED_BY"
        if direction == "FOLLOWED_BY":
            match_clause = "MATCH (anchor:GraphNode)-[:FOLLOWED_BY]->(target:GraphNode)"
        else:
            match_clause = "MATCH (target:GraphNode)-[:FOLLOWED_BY]->(anchor:GraphNode)"

        cypher = f"""
        {match_clause}
        WHERE ($movie_id IS NULL OR anchor.movie_id = $movie_id)
          AND anchor.node_type = 'TemporalChunk'
          AND target.node_type = 'TemporalChunk'
          AND all(token IN $anchor_tokens WHERE
              toLower(coalesce(anchor.script_primary_heading, '')) CONTAINS token OR
              toLower(coalesce(anchor.script_location, '')) CONTAINS token OR
              toLower(coalesce(anchor.scene_label, '')) CONTAINS token OR
              toLower(coalesce(anchor.description, '')) CONTAINS token
          )
        RETURN
            target.id AS node_id,
            target.node_type AS node_type,
            target.movie_id AS movie_id,
            coalesce(target.title, target.scene_label, target.id) AS title,
            coalesce(target.script_primary_heading, '') AS heading,
            coalesce(target.script_location, '') AS location,
            coalesce(target.script_time_of_day, '') AS time_of_day,
            coalesce(target.start_time, '') AS start_time,
            coalesce(target.end_time, '') AS end_time,
            coalesce(target.chunk_id, '') AS chunk_id,
            coalesce(target.description, target.dialogue_text, '') AS body
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            rows = [
                dict(record)
                for record in session.run(
                    cypher,
                    movie_id=movie_id,
                    anchor_tokens=anchor_tokens,
                    limit=max(limit * 2, 4),
                )
            ]

        for row in rows:
            row["body"] = (
                f"{query_request.get('direction', 'after').title()} scene: "
                f"{row.get('body', '')}"
            ).strip()
            row["score"] = 7.8
            row["source"] = "graph_structured_remote"
            row["question_type"] = "scene_transition"
            row.setdefault("neighbors", [])
        return rows[:limit]

    def _search_scene_characters_local(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        anchor_tokens = query_request.get("anchor_tokens", [])
        if not anchor_tokens:
            return []

        candidates = []
        for candidate in self._build_local_candidates(movie_id):
            if candidate.get("node_type") not in {"ScriptSubscene", "ScriptScene", "TemporalChunk"}:
                continue
            haystack = " ".join(
                self._normalize_field(candidate.get(field, ""))
                for field in ("heading", "location", "time_of_day", "title", "body")
            )
            if not all(token in haystack.split() or token in haystack for token in anchor_tokens):
                continue
            character_names = [
                neighbor.get("title", "")
                for neighbor in (candidate.get("neighbors") or [])
                if neighbor.get("node_type") == "Character" and neighbor.get("title")
            ]
            candidate_copy = dict(candidate)
            candidate_copy["character_names"] = character_names
            candidate_copy["body"] = (
                f"Characters present: {', '.join(character_names)}"
                if character_names
                else candidate_copy.get("body", "")
            )
            candidate_copy["score"] = 8.0 + min(1.0, 0.1 * len(character_names))
            candidate_copy["source"] = "graph_structured_local"
            candidate_copy["question_type"] = "scene_characters"
            candidates.append(candidate_copy)

        return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:limit]

    def _search_scene_transition_local(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        anchor_tokens = query_request.get("anchor_tokens", [])
        if not anchor_tokens:
            return []

        results = []
        movie_ids = [movie_id] if movie_id else [
            path.stem.replace("_chunks", "")
            for path in Cfg.get_temporal_chunks_dir().glob("*_chunks.json")
        ]
        for current_movie_id in movie_ids:
            chunks = self._load_json(
                Cfg.get_temporal_chunks_dir() / f"{current_movie_id}_chunks.json"
            ) or []
            ordered_chunks = sorted(chunks, key=lambda item: item.get("start_seconds", 0.0))
            for index, chunk in enumerate(ordered_chunks):
                haystack = " ".join(
                    self._normalize_field(chunk.get(field, ""))
                    for field in (
                        "script_primary_heading",
                        "script_location",
                        "script_time_of_day",
                        "scene_label",
                        "description",
                    )
                )
                if not all(token in haystack.split() or token in haystack for token in anchor_tokens):
                    continue
                neighbor_index = index + 1 if query_request.get("direction") == "after" else index - 1
                if neighbor_index < 0 or neighbor_index >= len(ordered_chunks):
                    continue
                neighbor = ordered_chunks[neighbor_index]
                results.append(
                    {
                        "node_id": neighbor.get("chunk_id", f"{current_movie_id}_chunk_{neighbor_index:04d}"),
                        "node_type": "TemporalChunk",
                        "movie_id": current_movie_id,
                        "title": neighbor.get("title", current_movie_id),
                        "heading": neighbor.get("script_primary_heading", ""),
                        "location": neighbor.get("script_location", ""),
                        "time_of_day": neighbor.get("script_time_of_day", ""),
                        "start_time": neighbor.get("start_time", ""),
                        "end_time": neighbor.get("end_time", ""),
                        "chunk_id": neighbor.get("chunk_id", ""),
                        "body": f"{query_request.get('direction', 'after').title()} scene: {neighbor.get('description', '')}",
                        "neighbors": [],
                        "score": 7.5,
                        "source": "graph_structured_local",
                        "question_type": "scene_transition",
                    }
                )
        return results[:limit]

    def _search_scene_location_remote(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        anchor_tokens = query_request.get("anchor_tokens", [])
        if not self.driver or not anchor_tokens:
            return []

        cypher = """
        MATCH (scene:GraphNode)
        OPTIONAL MATCH (char:GraphNode)-[r]-(scene)
        WHERE char.node_type = 'Character'
        WITH scene, collect(DISTINCT char.name)[0..12] AS character_names
        WHERE ($movie_id IS NULL OR scene.movie_id = $movie_id)
          AND scene.node_type IN ['ScriptSubscene', 'ScriptScene', 'TemporalChunk']
          AND coalesce(scene.location, scene.script_location, '') <> ''
          AND all(token IN $anchor_tokens WHERE
              toLower(coalesce(scene.heading, '')) CONTAINS token OR
              toLower(coalesce(scene.location, '')) CONTAINS token OR
              toLower(coalesce(scene.script_primary_heading, '')) CONTAINS token OR
              toLower(coalesce(scene.script_location, '')) CONTAINS token OR
              toLower(coalesce(scene.scene_label, '')) CONTAINS token OR
              any(name IN character_names WHERE toLower(coalesce(name, '')) CONTAINS token)
          )
        RETURN
            scene.id AS node_id,
            scene.node_type AS node_type,
            scene.movie_id AS movie_id,
            coalesce(scene.title, scene.name, scene.heading, scene.scene_label, scene.id) AS title,
            coalesce(scene.heading, scene.script_primary_heading, '') AS heading,
            coalesce(scene.location, scene.script_location, '') AS location,
            coalesce(scene.time_of_day, scene.script_time_of_day, '') AS time_of_day,
            coalesce(scene.start_time, '') AS start_time,
            coalesce(scene.end_time, '') AS end_time,
            coalesce(scene.chunk_id, '') AS chunk_id,
            character_names
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            rows = [
                dict(record)
                for record in session.run(
                    cypher,
                    movie_id=movie_id,
                    anchor_tokens=anchor_tokens,
                    limit=max(limit * 3, 6),
                )
            ]

        hits = []
        for row in rows:
            character_names = [name for name in row.get("character_names", []) if name]
            row["character_names"] = character_names
            row["body"] = (
                f"Location: {row.get('location', '')}. "
                f"Scene: {row.get('heading', row.get('title', ''))}."
            ).strip()
            row["score"] = 8.3 + min(0.6, 0.05 * len(character_names))
            row["source"] = "graph_structured_remote"
            row["question_type"] = "scene_location"
            row["neighbors"] = [
                {
                    "relation": "SET_IN",
                    "title": row.get("location", ""),
                    "node_id": row.get("location", ""),
                    "node_type": "Location",
                }
            ]
            hits.append(row)
        return hits[:limit]

    def _search_scene_location_local(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        anchor_tokens = query_request.get("anchor_tokens", [])
        if not anchor_tokens:
            return []

        results = []
        for candidate in self._build_local_candidates(movie_id):
            if candidate.get("node_type") not in {"ScriptSubscene", "ScriptScene", "TemporalChunk"}:
                continue
            if not candidate.get("location"):
                continue
            character_names = [
                neighbor.get("title", "")
                for neighbor in (candidate.get("neighbors") or [])
                if neighbor.get("node_type") == "Character" and neighbor.get("title")
            ]
            haystack = " ".join(
                filter(
                    None,
                    [
                        self._normalize_field(candidate.get("heading", "")),
                        self._normalize_field(candidate.get("location", "")),
                        self._normalize_field(candidate.get("title", "")),
                        self._normalize_field(candidate.get("body", "")),
                        " ".join(self._normalize_field(name) for name in character_names),
                    ],
                )
            )
            if not all(token in haystack.split() or token in haystack for token in anchor_tokens):
                continue

            candidate_copy = dict(candidate)
            candidate_copy["character_names"] = character_names
            candidate_copy["body"] = (
                f"Location: {candidate_copy.get('location', '')}. "
                f"Scene: {candidate_copy.get('heading', candidate_copy.get('title', ''))}."
            ).strip()
            candidate_copy["score"] = 8.0 + min(0.6, 0.05 * len(character_names))
            candidate_copy["source"] = "graph_structured_local"
            candidate_copy["question_type"] = "scene_location"
            candidate_copy.setdefault("neighbors", []).append(
                {
                    "relation": "SET_IN",
                    "title": candidate_copy.get("location", ""),
                    "node_id": candidate_copy.get("location", ""),
                    "node_type": "Location",
                }
            )
            results.append(candidate_copy)

        return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)[:limit]

    def _search_character_relationship_remote(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        entity_groups = query_request.get("entity_groups", [])
        if not self.driver or len(entity_groups) < 2:
            return []

        matched = self._resolve_character_groups_remote(entity_groups[:2], movie_id)
        if len(matched) < 2:
            return []

        left, right = matched[0], matched[1]
        cypher = """
        MATCH (left:GraphNode {id: $left_id})
        MATCH (right:GraphNode {id: $right_id})
        OPTIONAL MATCH (left)-[rel]-(right)
        WITH left, right, collect(DISTINCT type(rel))[0..8] AS direct_relations
        OPTIONAL MATCH (left)-[:APPEARS_IN|MENTIONED_IN_SCRIPT_SCENE|MENTIONED_IN_SCRIPT_SUBSCENE|APPEARS_IN_SUBSCENE]->(scene:GraphNode)
                       <-[:APPEARS_IN|MENTIONED_IN_SCRIPT_SCENE|MENTIONED_IN_SCRIPT_SUBSCENE|APPEARS_IN_SUBSCENE]-(right)
        WHERE scene.node_type IN ['TemporalChunk', 'ScriptScene', 'ScriptSubscene']
        WITH left, right, direct_relations,
             collect(
                DISTINCT {
                    node_id: scene.id,
                    title: coalesce(scene.heading, scene.scene_label, scene.id),
                    relation: 'SHARED_SCENE',
                    node_type: scene.node_type
                }
             )[0..5] AS shared_scenes
        OPTIONAL MATCH p = shortestPath((left)-[*..4]-(right))
        WHERE p IS NULL OR all(node IN nodes(p) WHERE $movie_id IS NULL OR node.movie_id = $movie_id)
        RETURN
            left.id AS left_id,
            left.name AS left_name,
            right.id AS right_id,
            right.name AS right_name,
            direct_relations,
            shared_scenes,
            [node IN CASE WHEN p IS NULL THEN [] ELSE nodes(p) END |
                {
                    node_id: node.id,
                    title: coalesce(node.name, node.heading, node.scene_label, node.id),
                    node_type: node.node_type
                }
            ] AS path_nodes,
            [rel IN CASE WHEN p IS NULL THEN [] ELSE relationships(p) END | type(rel)] AS path_relations
        """
        with self.driver.session(database=self.database) as session:
            row = session.run(
                cypher,
                left_id=left["node_id"],
                right_id=right["node_id"],
                movie_id=movie_id or left.get("movie_id"),
            ).single()

        if not row:
            return []

        row_dict = dict(row)
        direct_relations = [value for value in row_dict.get("direct_relations", []) if value]
        shared_scenes = row_dict.get("shared_scenes", []) or []
        path_nodes = row_dict.get("path_nodes", []) or []
        path_relations = [value for value in (row_dict.get("path_relations", []) or []) if value]
        body_parts = []
        if direct_relations:
            body_parts.append(
                f"Direct relations: {', '.join(sorted(set(direct_relations)))}"
            )
        if shared_scenes:
            shared_scene_titles = [
                scene.get("title", "") for scene in shared_scenes[:4] if scene.get("title")
            ]
            if shared_scene_titles:
                body_parts.append("Shared scenes: " + ", ".join(shared_scene_titles))
        path_summary = self._summarize_graph_path(path_nodes, path_relations)
        if path_summary:
            body_parts.append(f"Graph path: {path_summary}")
        if query_request.get("question_focus") == "why" and path_summary:
            body_parts.append("Explanation: the connection is supported by the shortest graph path and the shared scene evidence above")
        if not body_parts:
            return []

        return [
            {
                "node_id": f"{left['node_id']}__REL__{right['node_id']}",
                "node_type": "CharacterRelation",
                "movie_id": movie_id or left.get("movie_id", "unknown"),
                "title": f"{left['title']} <-> {right['title']}",
                "heading": "",
                "location": "",
                "character_names": [left["title"], right["title"]],
                "body": ". ".join(body_parts),
                "neighbors": shared_scenes
                + self._path_nodes_to_neighbors(path_nodes)
                + [
                    {
                        "relation": relation,
                        "title": right["title"],
                        "node_id": right["node_id"],
                        "node_type": "Character",
                    }
                    for relation in direct_relations[:4]
                ],
                "score": 9.1 if direct_relations else 8.6 if path_summary else 8.2,
                "source": "graph_structured_remote",
                "question_type": "character_relationship"
                if query_request.get("question_focus") != "why"
                else "character_relationship_why",
            }
        ][:limit]

    def _search_character_relationship_local(
        self, query_request: Dict[str, Any], movie_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        entity_groups = query_request.get("entity_groups", [])
        if len(entity_groups) < 2:
            return []

        movie_ids = [movie_id] if movie_id else [
            path.stem.replace("_chunks", "")
            for path in Cfg.get_temporal_chunks_dir().glob("*_chunks.json")
        ]
        results = []

        for current_movie_id in movie_ids:
            character_names = self._collect_local_character_names(current_movie_id)
            matched_names = []
            for group in entity_groups[:2]:
                match = self._match_character_group_local(group, character_names)
                if match:
                    matched_names.append(match)
            if len(matched_names) < 2:
                continue

            left_name, right_name = matched_names[0], matched_names[1]
            left_id = self._character_node_id(current_movie_id, left_name)
            right_id = self._character_node_id(current_movie_id, right_name)
            left_norm = self._normalize_entity_key(left_name)
            right_norm = self._normalize_entity_key(right_name)

            direct_relations = []
            shared_scenes = []
            path_nodes = []
            path_relations = []
            kg_graph = self._load_graph(Cfg.get_index_dir() / f"{current_movie_id}_kg.graphml")
            if kg_graph is not None:
                left_candidates = []
                right_candidates = []
                for node_id, data in kg_graph.nodes(data=True):
                    if self._kg_label(data.get("type", "Entity")) != "Character":
                        continue
                    normalized_name = self._normalize_entity_key(data.get("name", ""))
                    if normalized_name == left_norm:
                        left_candidates.append(node_id)
                    if normalized_name == right_norm:
                        right_candidates.append(node_id)

                for source_id in left_candidates:
                    for target_id in right_candidates:
                        edge_data = kg_graph.get_edge_data(source_id, target_id) or {}
                        if isinstance(edge_data, dict):
                            relation = edge_data.get("relation") or edge_data.get("description")
                            if relation:
                                direct_relations.append(str(relation))
                path_nodes, path_relations = self._find_local_character_path(
                    kg_graph, left_candidates, right_candidates
                )

            chunks = self._load_json(
                Cfg.get_temporal_chunks_dir() / f"{current_movie_id}_chunks.json"
            ) or []
            for chunk in chunks:
                scene_characters = set(chunk.get("characters", []) or [])
                scene_characters.update(chunk.get("script_characters", []) or [])
                if left_name in scene_characters and right_name in scene_characters:
                    shared_scenes.append(
                        {
                            "relation": "SHARED_SCENE",
                            "title": chunk.get("script_primary_heading")
                            or chunk.get("scene_label")
                            or chunk.get("chunk_id", ""),
                            "node_id": chunk.get("chunk_id", ""),
                            "node_type": "TemporalChunk",
                        }
                    )

            body_parts = []
            if direct_relations:
                body_parts.append(
                    f"Direct relations: {', '.join(sorted(set(direct_relations)))}"
                )
            if shared_scenes:
                shared_scene_titles = [
                    scene.get("title", "") for scene in shared_scenes[:4] if scene.get("title")
                ]
                if shared_scene_titles:
                    body_parts.append("Shared scenes: " + ", ".join(shared_scene_titles))
            path_summary = self._summarize_graph_path(path_nodes, path_relations)
            if path_summary:
                body_parts.append(f"Graph path: {path_summary}")
            if query_request.get("question_focus") == "why" and path_summary:
                body_parts.append("Explanation: the connection is justified by the shortest local graph path and overlapping scenes")
            if not body_parts:
                continue

            results.append(
                {
                    "node_id": f"{left_id}__REL__{right_id}",
                    "node_type": "CharacterRelation",
                    "movie_id": current_movie_id,
                    "title": f"{left_name} <-> {right_name}",
                    "heading": "",
                    "location": "",
                    "character_names": [left_name, right_name],
                    "body": ". ".join(body_parts),
                    "neighbors": shared_scenes + self._path_nodes_to_neighbors(path_nodes),
                    "score": 8.7 if direct_relations else 8.2 if path_summary else 7.9,
                    "source": "graph_structured_local",
                    "question_type": "character_relationship"
                    if query_request.get("question_focus") != "why"
                    else "character_relationship_why",
                }
            )

        return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)[:limit]

    @staticmethod
    def _summarize_graph_path(
        path_nodes: List[Dict[str, Any]], path_relations: List[str]
    ) -> str:
        if not path_nodes:
            return ""
        node_titles = [
            str(node.get("title", "") or node.get("node_id", "")).strip()
            for node in path_nodes
            if str(node.get("title", "") or node.get("node_id", "")).strip()
        ]
        if len(node_titles) < 2:
            return ""
        if not path_relations:
            return " <-> ".join(node_titles)
        parts = [node_titles[0]]
        for index, relation in enumerate(path_relations):
            if index + 1 >= len(node_titles):
                break
            parts.append(f"-[{relation}]-")
            parts.append(node_titles[index + 1])
        return " ".join(parts)

    @staticmethod
    def _path_nodes_to_neighbors(path_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        neighbors = []
        for node in path_nodes[1:-1]:
            title = str(node.get("title", "") or node.get("node_id", "")).strip()
            if not title:
                continue
            neighbors.append(
                {
                    "relation": "PATH_STEP",
                    "title": title,
                    "node_id": node.get("node_id", title),
                    "node_type": node.get("node_type", "GraphNode"),
                }
            )
        return neighbors[:6]

    def _find_local_character_path(
        self, kg_graph, left_candidates: List[str], right_candidates: List[str]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        if nx is None or kg_graph is None or not left_candidates or not right_candidates:
            return [], []

        try:
            graph_view = kg_graph.to_undirected()
        except Exception:
            graph_view = kg_graph

        best_path = None
        for source_id in left_candidates:
            for target_id in right_candidates:
                if source_id == target_id:
                    continue
                try:
                    path = nx.shortest_path(graph_view, source=source_id, target=target_id)
                except Exception:
                    continue
                if not path:
                    continue
                if best_path is None or len(path) < len(best_path):
                    best_path = path

        if not best_path or len(best_path) < 2:
            return [], []

        path_nodes = []
        path_relations = []
        for node_id in best_path:
            data = kg_graph.nodes[node_id]
            path_nodes.append(
                {
                    "node_id": node_id,
                    "title": data.get("name")
                    or data.get("title")
                    or data.get("heading")
                    or data.get("scene_label")
                    or node_id,
                    "node_type": self._kg_label(data.get("type", "Entity")),
                }
            )
        for source_id, target_id in zip(best_path, best_path[1:]):
            edge_data = kg_graph.get_edge_data(source_id, target_id) or {}
            relation = edge_data.get("relation") or edge_data.get("description") or "RELATED_TO"
            path_relations.append(self._sanitize_relationship_type(relation))
        return path_nodes, path_relations

    @staticmethod
    def _delete_movie_subgraph(tx, movie_id: str) -> None:
        tx.run("MATCH (n:GraphNode {movie_id: $movie_id}) DETACH DELETE n", movie_id=movie_id)

    @staticmethod
    def _ensure_constraints(session) -> None:
        session.run(
            "CREATE CONSTRAINT graph_node_id IF NOT EXISTS FOR (n:GraphNode) REQUIRE n.id IS UNIQUE"
        )
        session.run(
            "CREATE INDEX graph_node_movie_id IF NOT EXISTS FOR (n:GraphNode) ON (n.movie_id)"
        )

    def _write_payload(self, session, payload: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
        for label, rows in payload["nodes_by_label"].items():
            if not rows:
                continue
            query = (
                f"UNWIND $rows AS row "
                f"MERGE (n:GraphNode:{label} {{id: row.id}}) "
                "SET n += row.props, n.node_type = row.node_type"
            )
            session.run(query, rows=rows).consume()

        for rel_type, rows in payload["relationships_by_type"].items():
            if not rows:
                continue
            query = (
                f"UNWIND $rows AS row "
                "MATCH (src:GraphNode {id: row.source}) "
                "MATCH (dst:GraphNode {id: row.target}) "
                f"MERGE (src)-[r:{rel_type}]->(dst) "
                "SET r += row.props"
            )
            session.run(query, rows=rows).consume()

    def _build_temporal_payload(
        self,
        movie_id: str,
        chunks: List[Dict[str, Any]],
        subscenes: List[Dict[str, Any]],
        canonical_maps: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        payload = {"nodes_by_label": {}, "relationships_by_type": {}}
        movie_node_id = f"MOVIE_{movie_id}"
        movie_title = next(
            (chunk.get("title") for chunk in chunks if chunk.get("title")), movie_id
        )
        canonical_maps = canonical_maps or {}
        self._add_node(
            payload,
            "Movie",
            movie_node_id,
            {
                "movie_id": movie_id,
                "name": movie_title,
                "title": movie_title,
                "source_temporal_graph": True,
            },
        )

        character_lookup: Dict[str, str] = {}
        previous_chunk_id = None
        chunk_character_ids_by_id: Dict[str, List[str]] = {}
        for chunk in sorted(chunks, key=lambda item: item.get("start_seconds", 0.0)):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            chunk_character_ids: List[str] = []
            chunk_location = chunk.get("script_location", "")
            chunk_location_id = self._resolve_canonical_entity_id(
                chunk_location, "Location", canonical_maps
            ) if chunk_location else None

            self._add_node(
                payload,
                "TemporalChunk",
                chunk_id,
                {
                    "movie_id": movie_id,
                    "chunk_id": chunk_id,
                    "title": chunk.get("script_primary_heading", "")
                    or chunk.get("scene_label", "")
                    or chunk.get("title", movie_title),
                    "scene_label": chunk.get("scene_label", ""),
                    "description": chunk.get("description", ""),
                    "situation": chunk.get("situation", ""),
                    "dialogue_text": chunk.get("dialogue_text", ""),
                    "start_time": chunk.get("start_time", ""),
                    "end_time": chunk.get("end_time", ""),
                    "start_seconds": chunk.get("start_seconds"),
                    "end_seconds": chunk.get("end_seconds"),
                    "script_primary_heading": chunk.get("script_primary_heading", ""),
                    "script_location": chunk.get("script_location", ""),
                    "script_time_of_day": chunk.get("script_time_of_day", ""),
                    "script_characters": chunk.get("script_characters", []),
                    "script_headings": chunk.get("script_headings", []),
                    "script_scene_count": chunk.get("script_scene_count"),
                    "dominant_script_scene_uid": (
                        chunk.get("dominant_script_scene_ref", {}) or {}
                    ).get("script_scene_uid", ""),
                    "location_id": chunk_location_id or "",
                    "source_temporal_graph": True,
                },
            )
            if chunk_location_id:
                self._add_node(
                    payload,
                    "Location",
                    chunk_location_id,
                    {
                        "movie_id": movie_id,
                        "name": chunk_location,
                        "title": chunk_location,
                        "source_temporal_graph": True,
                    },
                )
                self._add_relationship(
                    payload,
                    "SET_IN",
                    chunk_id,
                    chunk_location_id,
                    {"movie_id": movie_id},
                )
            self._add_relationship(
                payload,
                "BELONGS_TO",
                chunk_id,
                movie_node_id,
                {"movie_id": movie_id},
            )
            if previous_chunk_id:
                self._add_relationship(
                    payload,
                    "FOLLOWED_BY",
                    previous_chunk_id,
                    chunk_id,
                    {"movie_id": movie_id},
                )
            previous_chunk_id = chunk_id

            for character_name in chunk.get("characters", []) or []:
                if self._should_skip_character_name(character_name):
                    continue
                character_id = self._resolve_canonical_character_id(
                    character_name, canonical_maps
                ) or self._character_node_id(movie_id, character_name)
                character_lookup.setdefault(
                    self._normalize_entity_key(character_name), character_id
                )
                self._add_node(
                    payload,
                    "Character",
                    character_id,
                    {
                        "movie_id": movie_id,
                        "name": character_name,
                        "title": character_name,
                        "source_temporal_graph": True,
                    },
                )
                self._add_relationship(
                    payload,
                    "APPEARS_IN",
                    character_id,
                    chunk_id,
                    {"movie_id": movie_id},
                )
                if character_id not in chunk_character_ids:
                    chunk_character_ids.append(character_id)

            for cast in chunk.get("cast_in_scene", []) or []:
                actor_name = cast.get("actor", "")
                character_name = cast.get("character", "")
                if not actor_name or not character_name:
                    continue
                if self._should_skip_character_name(character_name):
                    continue
                actor_id = f"{movie_id}_{actor_name}".upper()
                character_id = self._resolve_canonical_character_id(
                    character_name, canonical_maps
                ) or self._character_node_id(movie_id, character_name)
                character_lookup.setdefault(
                    self._normalize_entity_key(character_name), character_id
                )
                self._add_node(
                    payload,
                    "Actor",
                    actor_id,
                    {
                        "movie_id": movie_id,
                        "name": actor_name,
                        "title": actor_name,
                        "source_temporal_graph": True,
                    },
                )
                self._add_node(
                    payload,
                    "Character",
                    character_id,
                    {
                        "movie_id": movie_id,
                        "name": character_name,
                        "title": character_name,
                        "source_temporal_graph": True,
                    },
                )
                self._add_relationship(
                    payload,
                    "PLAYS",
                    actor_id,
                    character_id,
                    {"movie_id": movie_id},
                )
                if character_id not in chunk_character_ids:
                    chunk_character_ids.append(character_id)

            dominant_uid = (
                chunk.get("dominant_script_scene_ref", {}) or {}
            ).get("script_scene_uid", "")
            for ref in chunk.get("script_scene_refs", []) or []:
                script_scene_uid = ref.get("script_scene_uid")
                if not script_scene_uid:
                    continue
                script_location = ref.get("location", "")
                script_location_id = self._resolve_canonical_entity_id(
                    script_location, "Location", canonical_maps
                ) if script_location else None
                self._add_node(
                    payload,
                    "ScriptScene",
                    script_scene_uid,
                    {
                        "movie_id": movie_id,
                        "title": ref.get("heading", script_scene_uid),
                        "name": ref.get("heading", script_scene_uid),
                        "heading": ref.get("heading", ""),
                        "location": ref.get("location", ""),
                        "time_of_day": ref.get("time_of_day", ""),
                        "characters": ref.get("characters", []),
                        "scene_num": ref.get("scene_num"),
                        "start_sec": ref.get("start_sec"),
                        "end_sec": ref.get("end_sec"),
                        "anchor_quality": ref.get("anchor_quality", ""),
                        "confidence_score": ref.get("confidence_score"),
                        "anchor_start_sec": ref.get("anchor_start_sec"),
                        "anchor_end_sec": ref.get("anchor_end_sec"),
                        "linear_start_sec": ref.get("linear_start_sec"),
                        "linear_end_sec": ref.get("linear_end_sec"),
                        "location_id": script_location_id or "",
                        "source_script_graph": True,
                    },
                )
                if script_location_id:
                    self._add_node(
                        payload,
                        "Location",
                        script_location_id,
                        {
                            "movie_id": movie_id,
                            "name": script_location,
                            "title": script_location,
                            "source_script_graph": True,
                        },
                    )
                    self._add_relationship(
                        payload,
                        "SET_IN",
                        script_scene_uid,
                        script_location_id,
                        {"movie_id": movie_id},
                    )
                self._add_relationship(
                    payload,
                    "HAS_SCRIPT_SCENE",
                    movie_node_id,
                    script_scene_uid,
                    {"movie_id": movie_id},
                )
                self._add_relationship(
                    payload,
                    "ALIGNS_TO_SCRIPT_SCENE",
                    chunk_id,
                    script_scene_uid,
                    {
                        "movie_id": movie_id,
                        "anchor_quality": ref.get("anchor_quality", ""),
                        "confidence_score": ref.get("confidence_score"),
                        "overlap_seconds": self._coalesce_overlap_seconds(
                            chunk, ref
                        ),
                        "is_dominant": script_scene_uid == dominant_uid,
                    },
                )
                for name in ref.get("characters", []) or []:
                    if self._should_skip_character_name(name):
                        continue
                    character_id = self._resolve_canonical_character_id(
                        name, canonical_maps
                    ) or character_lookup.get(self._normalize_entity_key(name))
                    if character_id:
                        self._add_relationship(
                            payload,
                            "MENTIONED_IN_SCRIPT_SCENE",
                            character_id,
                            script_scene_uid,
                            {"movie_id": movie_id},
                        )

            chunk_character_ids_by_id[chunk_id] = chunk_character_ids

        for subscene in subscenes:
            subscene_id = subscene.get("subscene_id")
            if not subscene_id:
                continue
            script_scene_uid = subscene.get("script_scene_uid", "")
            parent_chunk_id = subscene.get("parent_chunk_id", "")
            subscene_location = subscene.get("script_location", "")
            subscene_location_id = self._resolve_canonical_entity_id(
                subscene_location, "Location", canonical_maps
            ) if subscene_location else None
            self._add_node(
                payload,
                "ScriptSubscene",
                subscene_id,
                {
                    "movie_id": movie_id,
                    "chunk_id": parent_chunk_id,
                    "title": subscene.get("script_heading", subscene_id),
                    "name": subscene.get("script_heading", subscene_id),
                    "heading": subscene.get("script_heading", ""),
                    "location": subscene.get("script_location", ""),
                    "time_of_day": subscene.get("script_time_of_day", ""),
                    "script_characters": subscene.get("script_characters", []),
                    "parent_scene_id": subscene.get("parent_scene_id", ""),
                    "parent_chunk_id": parent_chunk_id,
                    "start_time": subscene.get("start_time", ""),
                    "end_time": subscene.get("end_time", ""),
                    "start_seconds": subscene.get("start_seconds"),
                    "end_seconds": subscene.get("end_seconds"),
                    "anchor_quality": subscene.get("anchor_quality", ""),
                    "confidence_score": subscene.get("confidence_score"),
                    "overlap_seconds": subscene.get("overlap_seconds"),
                    "overlap_ratio_semantic": subscene.get("overlap_ratio_semantic"),
                    "overlap_ratio_script": subscene.get("overlap_ratio_script"),
                    "dialogue_text": subscene.get("dialogue_excerpt", ""),
                    "semantic_description": subscene.get("semantic_description", ""),
                    "semantic_scene_label": subscene.get("semantic_scene_label", ""),
                    "indexable": bool(subscene.get("indexable")),
                    "is_canonical_subscene": bool(subscene.get("is_canonical_subscene")),
                    "location_id": subscene_location_id or "",
                    "source_script_graph": True,
                },
            )
            if subscene_location_id:
                self._add_node(
                    payload,
                    "Location",
                    subscene_location_id,
                    {
                        "movie_id": movie_id,
                        "name": subscene_location,
                        "title": subscene_location,
                        "source_script_graph": True,
                    },
                )
                self._add_relationship(
                    payload,
                    "SET_IN",
                    subscene_id,
                    subscene_location_id,
                    {"movie_id": movie_id},
                )
            self._add_relationship(
                payload,
                "HAS_SCRIPT_SUBSCENE",
                movie_node_id,
                subscene_id,
                {"movie_id": movie_id},
            )
            if parent_chunk_id:
                self._add_relationship(
                    payload,
                    "BELONGS_TO_CHUNK",
                    subscene_id,
                    parent_chunk_id,
                    {"movie_id": movie_id},
                )
            if script_scene_uid:
                self._add_relationship(
                    payload,
                    "DERIVED_FROM",
                    subscene_id,
                    script_scene_uid,
                    {"movie_id": movie_id},
                )
            for name in subscene.get("script_characters", []) or []:
                if self._should_skip_character_name(name):
                    continue
                character_id = self._resolve_canonical_character_id(
                    name, canonical_maps
                ) or character_lookup.get(self._normalize_entity_key(name))
                if character_id:
                    self._add_relationship(
                        payload,
                        "MENTIONED_IN_SCRIPT_SUBSCENE",
                        character_id,
                        subscene_id,
                        {"movie_id": movie_id},
                    )
            for character_id in chunk_character_ids_by_id.get(parent_chunk_id, []):
                self._add_relationship(
                    payload,
                    "APPEARS_IN_SUBSCENE",
                    character_id,
                    subscene_id,
                    {"movie_id": movie_id},
                )

        return payload

    def _build_kg_payload(
        self,
        movie_id: str,
        graph,
        chunk_ids: Iterable[str],
        canonical_maps: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        payload = {"nodes_by_label": {}, "relationships_by_type": {}}
        movie_node_id = f"MOVIE_{movie_id}"
        chunk_ids = set(chunk_ids)
        known_nodes = set()
        canonical_maps = canonical_maps or {}

        for node_id, data in graph.nodes(data=True):
            if data.get("movie_id") != movie_id:
                continue
            node_type = str(data.get("type", "Entity") or "Entity")
            label = self._kg_label(node_type)
            node_name = (
                data.get("name")
                or data.get("title")
                or data.get("heading")
                or data.get("scene_label")
                or ""
            )
            if label == "Character" and self._should_skip_character_name(node_name):
                continue
            canonical_node_id = None
            if label in {"Character", "Location", "Object"}:
                canonical_node_id = self._resolve_canonical_entity_id(
                    data.get("name", ""), label, canonical_maps
                )
            actual_node_id = canonical_node_id or node_id
            scene_idx = self._maybe_int(data.get("scene_idx"))
            chunk_id = data.get("chunk_id", "")
            if not chunk_id and scene_idx is not None:
                candidate_chunk_id = f"{movie_id}_chunk_{scene_idx:04d}"
                if candidate_chunk_id in chunk_ids:
                    chunk_id = candidate_chunk_id

            self._add_node(
                payload,
                label,
                actual_node_id,
                {
                    "movie_id": movie_id,
                    "title": node_name or actual_node_id,
                    "name": data.get("name", ""),
                    "description": data.get("description", ""),
                    "text": data.get("text", ""),
                    "scene_idx": scene_idx,
                    "chunk_id": chunk_id,
                    "source_kg_graph": True,
                },
            )
            known_nodes.add(node_id)
            if actual_node_id != node_id:
                known_nodes.add(actual_node_id)

            if label == "KGScene":
                self._add_relationship(
                    payload,
                    "HAS_KG_SCENE",
                    movie_node_id,
                    actual_node_id,
                    {"movie_id": movie_id},
                )
                if chunk_id:
                    self._add_relationship(
                        payload,
                        "DESCRIBES_CHUNK",
                        actual_node_id,
                        chunk_id,
                        {"movie_id": movie_id},
                    )

        for source_id, target_id, edge_data in graph.edges(data=True):
            if source_id not in known_nodes or target_id not in known_nodes:
                continue
            source_data = graph.nodes[source_id]
            target_data = graph.nodes[target_id]
            canonical_source = source_id
            canonical_target = target_id
            source_label = self._kg_label(source_data.get("type", "Entity"))
            target_label = self._kg_label(target_data.get("type", "Entity"))
            if source_label == "Character" and self._should_skip_character_name(
                source_data.get("name", "")
            ):
                continue
            if target_label == "Character" and self._should_skip_character_name(
                target_data.get("name", "")
            ):
                continue
            if source_label in {"Character", "Location", "Object"}:
                canonical_source = self._resolve_canonical_entity_id(
                    source_data.get("name", ""), source_label, canonical_maps
                ) or source_id
            if target_label in {"Character", "Location", "Object"}:
                canonical_target = self._resolve_canonical_entity_id(
                    target_data.get("name", ""), target_label, canonical_maps
                ) or target_id
            self._add_relationship(
                payload,
                self._sanitize_relationship_type(
                    edge_data.get("relation", "INTERACTS_WITH")
                ),
                canonical_source,
                canonical_target,
                {
                    "movie_id": movie_id,
                    "scene_idx": self._maybe_int(edge_data.get("scene_idx")),
                    "description": edge_data.get("description", ""),
                    "source_kg_graph": True,
                },
            )

        return payload

    @staticmethod
    def _merge_payload(target: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        for label, rows in incoming["nodes_by_label"].items():
            for row in rows:
                Neo4jGraphStore._add_node(
                    target,
                    label,
                    row["id"],
                    row.get("props", {}),
                )
        for rel_type, rows in incoming["relationships_by_type"].items():
            for row in rows:
                Neo4jGraphStore._add_relationship(
                    target,
                    rel_type,
                    row["source"],
                    row["target"],
                    row.get("props", {}),
                )

    @staticmethod
    def _merge_ranked_hit(
        merged_hits: Dict[str, Dict[str, Any]], hit: Dict[str, Any]
    ) -> None:
        node_id = hit.get("node_id", "")
        if not node_id:
            return

        existing = merged_hits.get(node_id)
        if existing is None or float(hit.get("score", 0.0)) > float(
            existing.get("score", 0.0)
        ):
            merged_hits[node_id] = hit
        elif existing.get("source") != hit.get("source"):
            existing["source"] = "hybrid"
            if hit.get("character_names") and not existing.get("character_names"):
                existing["character_names"] = hit.get("character_names")

    @staticmethod
    def _add_node(
        payload: Dict[str, Any], label: str, node_id: str, props: Dict[str, Any]
    ) -> None:
        sanitized_props = Neo4jGraphStore._sanitize_props({"id": node_id, **props})
        payload.setdefault("_node_registry", {})
        registry = payload["_node_registry"]

        existing = registry.get(node_id)
        if existing:
            existing_label = existing["label"]
            existing_row = existing["row"]
            merged_props = Neo4jGraphStore._merge_props(
                existing_row["props"], sanitized_props
            )
            preferred_label = existing_label
            if Neo4jGraphStore._label_priority(label) > Neo4jGraphStore._label_priority(
                existing_label
            ):
                payload["nodes_by_label"][existing_label] = [
                    row
                    for row in payload["nodes_by_label"].get(existing_label, [])
                    if row["id"] != node_id
                ]
                payload["nodes_by_label"].setdefault(label, []).append(existing_row)
                existing["label"] = label
                preferred_label = label

            existing_row["node_type"] = preferred_label
            existing_row["props"] = merged_props
            return

        payload["nodes_by_label"].setdefault(label, [])
        row = {
            "id": node_id,
            "node_type": label,
            "props": sanitized_props,
        }
        payload["nodes_by_label"][label].append(row)
        registry[node_id] = {"label": label, "row": row}

    @staticmethod
    def _add_relationship(
        payload: Dict[str, Any],
        rel_type: str,
        source: str,
        target: str,
        props: Dict[str, Any],
    ) -> None:
        rel_type = Neo4jGraphStore._sanitize_relationship_type(rel_type)
        payload["relationships_by_type"].setdefault(rel_type, [])
        rows = payload["relationships_by_type"][rel_type]
        rel_key = (source, target)
        if any((row["source"], row["target"]) == rel_key for row in rows):
            return
        rows.append(
            {
                "source": source,
                "target": target,
                "props": Neo4jGraphStore._sanitize_props(props),
            }
        )

    @staticmethod
    def _label_priority(label: str) -> int:
        priorities = {
            "Movie": 100,
            "TemporalChunk": 90,
            "ScriptSubscene": 85,
            "ScriptScene": 80,
            "Character": 70,
            "Actor": 65,
            "Location": 60,
            "Object": 55,
            "KGScene": 50,
            "Entity": 10,
        }
        return priorities.get(str(label or ""), 0)

    @staticmethod
    def _load_json(path: Path):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load JSON %s: %s", path, exc)
            return None

    @staticmethod
    def _load_graph(path: Path):
        if not path.exists() or nx is None:
            return None
        try:
            return nx.read_graphml(path)
        except Exception as exc:
            logger.warning("Failed to load GraphML %s: %s", path, exc)
            return None

    @staticmethod
    def _sanitize_props(props: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in props.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    sanitized[key] = value
                else:
                    sanitized[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            else:
                sanitized[key] = str(value)
        return sanitized

    @staticmethod
    def _merge_props(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(existing)
        for key, value in incoming.items():
            if value in ("", [], {}, None):
                continue
            if key not in merged or merged.get(key) in ("", [], {}, None):
                merged[key] = value
            elif isinstance(merged.get(key), list) and isinstance(value, list):
                seen = list(merged[key])
                for item in value:
                    if item not in seen:
                        seen.append(item)
                merged[key] = seen
        return merged

    def _build_canonical_maps(
        self,
        movie_id: str,
        chunks: List[Dict[str, Any]],
        subscenes: List[Dict[str, Any]],
        kg_graph=None,
    ) -> Dict[str, Any]:
        character_names: List[str] = []
        location_names: List[str] = []
        object_names: List[str] = []

        for chunk in chunks:
            character_names.extend(chunk.get("characters", []) or [])
            character_names.extend(chunk.get("script_characters", []) or [])
            character_names.extend(
                cast.get("character", "")
                for cast in (chunk.get("cast_in_scene", []) or [])
                if cast.get("character")
            )
            location_names.extend(
                value
                for value in (
                    chunk.get("script_location", ""),
                    chunk.get("scene_label", ""),
                )
                if value
            )
            for ref in chunk.get("script_scene_refs", []) or []:
                character_names.extend(ref.get("characters", []) or [])
                location_names.extend(
                    value for value in (ref.get("location", ""), ref.get("heading", "")) if value
                )

        for subscene in subscenes:
            character_names.extend(subscene.get("script_characters", []) or [])
            location_names.extend(
                value
                for value in (
                    subscene.get("script_location", ""),
                    subscene.get("script_heading", ""),
                )
                if value
            )

        if kg_graph is not None:
            for _node_id, data in kg_graph.nodes(data=True):
                label = self._kg_label(data.get("type", "Entity"))
                name = (
                    data.get("name")
                    or data.get("title")
                    or data.get("heading")
                    or data.get("scene_label")
                    or ""
                )
                if not name:
                    continue
                if label == "Character":
                    character_names.append(name)
                elif label == "Location":
                    location_names.append(name)
                elif label == "Object":
                    object_names.append(name)

        return {
            "movie_id": movie_id,
            "Character": self._build_entity_canonical_map(
                movie_id, "Character", character_names
            ),
            "Location": self._build_entity_canonical_map(
                movie_id, "Location", location_names
            ),
            "Object": self._build_entity_canonical_map(movie_id, "Object", object_names),
            "script_character_count": sum(
                len(subscene.get("script_characters", []) or []) for subscene in subscenes
            ),
        }

    def _build_entity_canonical_map(
        self, movie_id: str, entity_type: str, names: List[str]
    ) -> Dict[str, Any]:
        exact_map: Dict[str, str] = {}
        token_map: Dict[str, List[str]] = {}
        display_name_by_id: Dict[str, str] = {}
        alias_groups: Dict[str, List[tuple[str, str]]] = {}
        signature_index: Dict[str, List[str]] = {}
        records: List[tuple[str, str, str]] = []

        for name in names:
            normalized = self._normalize_entity_key(name)
            if not normalized or self._is_generic_entity_label(normalized, entity_type):
                continue
            signature = self._canonical_signature(entity_type, normalized)
            records.append((name, normalized, signature))
            signature_index.setdefault(signature, []).append(normalized)

        for name, normalized, signature in records:
            canonical_signature = self._resolve_signature_alias(
                entity_type, signature, signature_index
            )
            alias_groups.setdefault(canonical_signature, []).append((name, normalized))

        for _signature, aliases in alias_groups.items():
            preferred_name = self._pick_preferred_entity_name(
                entity_type, [name for name, _normalized in aliases]
            )
            node_id = self._entity_node_id(movie_id, entity_type, preferred_name)
            display_name_by_id[node_id] = preferred_name

            for alias_name, normalized in aliases:
                existing_id = exact_map.get(normalized)
                if existing_id and existing_id != node_id:
                    current_name = display_name_by_id.get(existing_id, "")
                    if self._entity_name_priority_score(
                        entity_type, preferred_name
                    ) <= self._entity_name_priority_score(entity_type, current_name):
                        node_id = existing_id
                    else:
                        display_name_by_id[node_id] = preferred_name
                exact_map[normalized] = node_id

                alias_signature = self._canonical_signature(entity_type, normalized)
                if alias_signature and alias_signature != normalized:
                    exact_map.setdefault(alias_signature, node_id)

                for token in normalized.split():
                    if len(token) < 3:
                        continue
                    token_map.setdefault(token, [])
                    if node_id not in token_map[token]:
                        token_map[token].append(node_id)

        return {
            "exact": exact_map,
            "token": token_map,
            "display_name_by_id": display_name_by_id,
        }

    @classmethod
    def _canonical_signature(cls, entity_type: str, normalized: str) -> str:
        tokens = [token for token in normalized.split() if token]
        if not tokens:
            return normalized

        if entity_type == "Character":
            ignored = {"MR", "MRS", "MS", "MISS", "DR", "DOCTOR", "THE"}
            tokens = [token for token in tokens if token not in ignored]
            if len(tokens) > 1:
                tokens = [token for token in tokens if len(token) > 1]
        elif entity_type == "Location":
            ignored = {
                "INT",
                "EXT",
                "INTEXT",
                "INTERIOR",
                "EXTERIOR",
                "DAY",
                "NIGHT",
                "MORNING",
                "EVENING",
                "AFTERNOON",
                "DAWN",
                "DUSK",
                "LATE",
                "EARLY",
                "SAME",
                "CONTINUOUS",
                "LATER",
            }
            tokens = [token for token in tokens if token not in ignored]
        elif entity_type == "Object":
            ignored = {"THE", "A", "AN"}
            tokens = [token for token in tokens if token not in ignored]

        return " ".join(tokens) or normalized

    @classmethod
    def _resolve_signature_alias(
        cls,
        entity_type: str,
        signature: str,
        signature_index: Dict[str, List[str]],
    ) -> str:
        signature_tokens = {token for token in signature.split() if token}
        if not signature_tokens:
            return signature

        if entity_type == "Character" and len(signature_tokens) == 1:
            supersets = [
                candidate
                for candidate in signature_index
                if candidate != signature
                and signature_tokens < {token for token in candidate.split() if token}
            ]
            if len(supersets) == 1:
                return supersets[0]

        if entity_type in {"Location", "Object"}:
            supersets = [
                candidate
                for candidate in signature_index
                if candidate != signature
                and signature_tokens <= {token for token in candidate.split() if token}
            ]
            if len(supersets) == 1 and len(supersets[0].split()) <= len(signature_tokens) + 2:
                return supersets[0]

        return signature

    @classmethod
    def _pick_preferred_entity_name(cls, entity_type: str, names: List[str]) -> str:
        if not names:
            return ""
        ranked = sorted(
            names,
            key=lambda value: cls._entity_name_priority_score(entity_type, value),
            reverse=True,
        )
        return ranked[0]

    @classmethod
    def _entity_name_priority_score(cls, entity_type: str, value: str) -> tuple[int, int, int]:
        normalized = cls._normalize_entity_key(value)
        signature = cls._canonical_signature(entity_type, normalized)
        signature_tokens = signature.split()
        value_upper = str(value or "").upper()
        penalty = 0
        if entity_type == "Character":
            if re.search(r"\((?:O\.?\s*S\.?|V\.?\s*O\.?|O\.?\s*C\.?)\)", value_upper):
                penalty -= 5
            if value_upper.startswith(("DR ", "DR.", "MR ", "MR.", "MRS ", "MRS.", "MS ", "MS.")):
                penalty -= 6
        elif entity_type == "Location":
            if any(token in value_upper for token in (" DAY", " NIGHT", " MORNING", " EVENING", " AFTERNOON", " DAWN", " DUSK")):
                penalty -= 2
            if "-" in str(value or ""):
                penalty -= 1
        return (
            len(signature_tokens),
            len(signature),
            penalty + len(str(value or "")),
        )

    def _resolve_canonical_entity_id(
        self,
        name: str,
        entity_type: str,
        canonical_maps: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if not name or not canonical_maps:
            return None

        normalized = self._normalize_entity_key(name)
        if not normalized:
            return None

        entity_map = canonical_maps.get(entity_type, {})
        exact_map = entity_map.get("exact", {})
        if normalized in exact_map:
            return exact_map[normalized]

        signature = self._canonical_signature(entity_type, normalized)
        if signature in exact_map:
            return exact_map[signature]

        token_candidates = set()
        for token in normalized.split():
            if len(token) < 3:
                continue
            for candidate in entity_map.get("token", {}).get(token, []):
                token_candidates.add(candidate)

        if len(token_candidates) == 1:
            return next(iter(token_candidates))
        if len(token_candidates) > 1:
            signature_tokens = set(signature.split())
            display_name_by_id = entity_map.get("display_name_by_id", {})
            scored_candidates = []
            for candidate in token_candidates:
                candidate_signature = self._canonical_signature(
                    entity_type,
                    self._normalize_entity_key(display_name_by_id.get(candidate, "")),
                )
                candidate_tokens = set(candidate_signature.split())
                overlap = len(signature_tokens & candidate_tokens)
                if not overlap:
                    continue
                scored_candidates.append((overlap, len(candidate_tokens), candidate))
            if scored_candidates:
                scored_candidates.sort(reverse=True)
                if len(scored_candidates) == 1 or scored_candidates[0][:2] != scored_candidates[1][:2]:
                    return scored_candidates[0][2]
        return None

    def _collect_local_character_names(self, movie_id: str) -> List[str]:
        preferred_by_key: Dict[str, str] = {}

        def add_name(name: str) -> None:
            normalized = self._normalize_entity_key(name)
            signature = self._canonical_signature("Character", normalized)
            if self._should_skip_character_name(name):
                return
            key = signature or normalized
            current = preferred_by_key.get(key)
            if (
                current is None
                or self._entity_name_priority_score("Character", name)
                > self._entity_name_priority_score("Character", current)
            ):
                preferred_by_key[key] = name

        chunks = self._load_json(Cfg.get_temporal_chunks_dir() / f"{movie_id}_chunks.json") or []
        for chunk in chunks:
            for name in chunk.get("characters", []) or []:
                add_name(name)
            for name in chunk.get("script_characters", []) or []:
                add_name(name)
            for cast in chunk.get("cast_in_scene", []) or []:
                add_name(cast.get("character", ""))
            for ref in chunk.get("script_scene_refs", []) or []:
                for name in ref.get("characters", []) or []:
                    add_name(name)

        subscenes = self._load_json(
            Cfg.get_script_subscenes_dir() / f"{movie_id}_script_subscenes.json"
        ) or []
        for subscene in subscenes:
            for name in subscene.get("script_characters", []) or []:
                add_name(name)

        kg_graph = self._load_graph(Cfg.get_index_dir() / f"{movie_id}_kg.graphml")
        if kg_graph is not None:
            for _node_id, data in kg_graph.nodes(data=True):
                if self._kg_label(data.get("type", "Entity")) != "Character":
                    continue
                add_name(data.get("name", ""))

        return sorted(
            preferred_by_key.values(),
            key=lambda value: self._entity_name_priority_score("Character", value),
            reverse=True,
        )

    def _match_character_group_local(
        self, group: List[str], character_names: List[str]
    ) -> Optional[str]:
        group_tokens = [token for token in group if token]
        if not group_tokens:
            return None

        best_name = None
        best_score = None
        for name in character_names:
            score = self._character_query_match_score(group_tokens, name)
            matched_ratio = score[0]
            matched = score[1]
            if not matched:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_name = name

        if best_score and best_score[0] >= 0.5:
            return best_name
        return None

    def _resolve_character_groups_remote(
        self, entity_groups: List[List[str]], movie_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not self.driver:
            return []

        matched: List[Dict[str, Any]] = []
        cypher = """
        MATCH (char:GraphNode)
        WHERE char.node_type = 'Character'
          AND ($movie_id IS NULL OR char.movie_id = $movie_id)
        WITH char, toLower(coalesce(char.name, char.title, '')) AS char_name
        WITH char, char_name,
             size([token IN $tokens WHERE char_name CONTAINS token]) AS matched_tokens
        WHERE matched_tokens > 0
        RETURN
            char.id AS node_id,
            char.movie_id AS movie_id,
            coalesce(char.name, char.title, char.id) AS title,
            matched_tokens
        ORDER BY matched_tokens DESC, size(split(char_name, ' ')) DESC, size(char_name) DESC
        LIMIT 12
        """

        with self.driver.session(database=self.database) as session:
            for group in entity_groups:
                tokens = [token for token in group if token]
                if not tokens:
                    continue
                rows = [
                    dict(record)
                    for record in session.run(
                        cypher,
                        movie_id=movie_id,
                        tokens=tokens,
                    )
                ]
                if not rows:
                    continue
                scored_rows = []
                for row in rows:
                    title = row.get("title", "")
                    if self._should_skip_character_name(title):
                        continue
                    score = self._character_query_match_score(tokens, title)
                    scored_rows.append((score, row))
                if not scored_rows:
                    continue
                scored_rows.sort(key=lambda item: item[0], reverse=True)
                best_score, best_row = scored_rows[0]
                required_overlap = max(1, min(len(tokens), 2))
                if int(best_row.get("matched_tokens", 0)) < required_overlap or best_score[0] < 0.5:
                    continue
                matched.append(best_row)
        return matched

    def _extract_entity_groups(self, query: str) -> List[List[str]]:
        lowered = str(query or "").lower()
        patterns = [
            r"between\s+(.+?)\s+and\s+(.+)",
            r"how\s+is\s+(.+?)\s+related\s+to\s+(.+)",
            r"how\s+are\s+(.+?)\s+and\s+(.+?)\s+related",
            r"how\s+are\s+(.+?)\s+connected\s+to\s+(.+)",
            r"relationship\s+of\s+(.+?)\s+and\s+(.+)",
            r"relationship\s+between\s+(.+?)\s+and\s+(.+)",
            r"relation\s+between\s+(.+?)\s+and\s+(.+)",
            r"(.+?)\s+and\s+(.+?)\s+relationship",
            r"(.+?)\s+with\s+(.+)",
        ]
        groups: List[List[str]] = []
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            groups = [
                self._tokenize(match.group(1)),
                self._tokenize(match.group(2)),
            ]
            break

        if not groups:
            connectors = (" and ", " with ")
            for connector in connectors:
                if connector not in lowered:
                    continue
                left, right = lowered.split(connector, 1)
                left_tokens = self._tokenize(left)
                right_tokens = self._tokenize(right)
                if left_tokens and right_tokens:
                    groups = [left_tokens, right_tokens]
                    break

        cleaned_groups: List[List[str]] = []
        relationship_fillers = {
            "relationship",
            "related",
            "relation",
            "between",
            "connected",
            "connection",
            "with",
            "and",
            "how",
            "is",
            "are",
            "the",
        }
        for group in groups:
            cleaned = [token for token in group if token not in relationship_fillers]
            if cleaned:
                cleaned_groups.append(cleaned)
        return cleaned_groups

    def _resolve_canonical_character_id(
        self, name: str, canonical_maps: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        return self._resolve_canonical_entity_id(name, "Character", canonical_maps)

    @staticmethod
    def _coalesce_overlap_seconds(
        chunk: Dict[str, Any], scene_ref: Dict[str, Any]
    ) -> float:
        explicit = scene_ref.get("overlap_seconds")
        if explicit is not None:
            try:
                return float(explicit)
            except Exception:
                pass
        try:
            return round(
                max(
                    0.0,
                    min(float(chunk.get("end_seconds", 0.0)), float(scene_ref.get("end_sec", 0.0)))
                    - max(float(chunk.get("start_seconds", 0.0)), float(scene_ref.get("start_sec", 0.0))),
                ),
                3,
            )
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_entity_key(value: str) -> str:
        normalized = re.sub(r"\(.*?\)", " ", str(value or ""))
        normalized = normalized.replace('"', " ")
        normalized = re.sub(r"[^A-Za-z0-9]+", " ", normalized).strip().upper()
        return normalized

    @classmethod
    def _character_node_id(cls, movie_id: str, name: str) -> str:
        return cls._entity_node_id(movie_id, "Character", name)

    @classmethod
    def _entity_node_id(cls, movie_id: str, entity_type: str, name: str) -> str:
        normalized = cls._normalize_entity_key(name)
        if entity_type == "Character":
            return f"{movie_id}_{name}".upper()
        return f"{movie_id}_{entity_type}_{normalized}".upper()

    @staticmethod
    def _is_generic_entity_label(normalized: str, entity_type: str) -> bool:
        generic_tokens = {
            "Character": {
                "MAN",
                "WOMAN",
                "BOY",
                "GIRL",
                "MALE",
                "FEMALE",
                "NARRATOR",
                "VOICE",
                "VISITOR",
                "CLASS",
                "UNKNOWN",
                "PERSON",
                "SOMEONE",
                "SOMEBODY",
                "GUY",
                "LADY",
                "CHILD",
                "KID",
            },
            "Location": {
                "INT",
                "EXT",
                "ROOM",
                "HOUSE",
                "HOME",
                "HALL",
                "HALLWAY",
                "STREET",
                "INTERIOR",
                "EXTERIOR",
                "PLACE",
            },
            "Object": {
                "OBJECT",
                "THING",
                "ITEM",
                "STUFF",
            },
        }
        tokens = normalized.split()
        if not tokens:
            return True
        strong_tokens = [
            token for token in tokens if token not in {"INT", "EXT", "THE", "A", "AN"}
        ]
        blocked = generic_tokens.get(entity_type, set())
        return bool(strong_tokens) and all(token in blocked for token in strong_tokens)

    @staticmethod
    def _is_relational_character_label(normalized: str) -> bool:
        tokens = [token for token in normalized.split() if token and token != "S"]
        if len(tokens) < 2:
            return False
        relational_tokens = {
            "MOTHER",
            "FATHER",
            "MOM",
            "DAD",
            "SON",
            "DAUGHTER",
            "WIFE",
            "HUSBAND",
            "SISTER",
            "BROTHER",
            "TEACHER",
            "DOCTOR",
            "PATIENT",
            "VOICE",
            "NARRATOR",
            "VISITOR",
            "LADY",
            "MAN",
            "WOMAN",
            "BOY",
            "GIRL",
            "PERSON",
        }
        return tokens[-1] in relational_tokens

    def _should_skip_character_name(self, name: str) -> bool:
        normalized = self._normalize_entity_key(name)
        if not normalized:
            return True
        raw_upper = str(name or "").upper()
        if len(normalized.split()) >= 5 and any(punct in raw_upper for punct in ("!", "?", ",")):
            return True
        if re.search(r"\((?:O\.?\s*S\.?|V\.?\s*O\.?|O\.?\s*C\.?)\)", raw_upper):
            return True
        if self._is_generic_entity_label(normalized, "Character"):
            return True
        return self._is_relational_character_label(normalized)

    def _character_query_match_score(self, group_tokens: List[str], name: str) -> tuple[float, int, int, int]:
        normalized_name = self._normalize_field(name)
        raw_upper = str(name or "").upper()
        raw_tokens = normalized_name.split()
        honorific_tokens = {"dr", "doctor", "mr", "mrs", "ms", "miss", "sir", "lady"}
        stage_tokens = {"o", "s", "v", "os", "vo", "oc"}
        clean_tokens = [
            token
            for token in raw_tokens
            if token not in honorific_tokens and token not in stage_tokens and len(token) > 1
        ]
        name_tokens = set(clean_tokens or raw_tokens)
        searchable_name = " ".join(clean_tokens or raw_tokens)
        matched = sum(
            1 for token in group_tokens if token in name_tokens or token in searchable_name
        )
        relational_penalty = -2 if self._is_relational_character_label(self._normalize_entity_key(name)) else 0
        stage_penalty = -3 if re.search(r"\((?:O\.?\s*S\.?|V\.?\s*O\.?|O\.?\s*C\.?)\)", raw_upper) else 0
        honorific_penalty = -1 if raw_upper.startswith(("DR ", "DR.", "MR ", "MR.", "MRS ", "MRS.", "MS ", "MS.")) else 0
        return (
            matched / max(len(group_tokens), 1),
            matched + relational_penalty + stage_penalty + honorific_penalty,
            len(name_tokens),
            len(searchable_name),
        )

    @staticmethod
    def _sanitize_relationship_type(value: str) -> str:
        relation = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "RELATED_TO").upper())
        relation = relation.strip("_") or "RELATED_TO"
        if relation[0].isdigit():
            relation = f"REL_{relation}"
        return relation

    @staticmethod
    def _kg_label(node_type: str) -> str:
        normalized = str(node_type or "Entity")
        if normalized == "SceneChunk":
            return "KGScene"
        if normalized in {"Character", "Location", "Object"}:
            return normalized
        return "Entity"

    @staticmethod
    def _maybe_int(value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except Exception:
            return None

    def _classify_graph_query(self, query: str) -> Dict[str, Any]:
        tokens = self._tokenize(query)
        token_set = set(tokens)
        character_tokens = {"who", "character", "appears", "appear", "present"}
        where_tokens = {"where", "location", "set", "place", "room"}
        after_tokens = {"after", "next", "following", "then"}
        before_tokens = {"before", "previous", "prior"}
        why_tokens = {"why", "reason", "because", "motivation", "motivated", "cause", "caused"}
        path_tokens = {"how", "linked", "link", "path", "through"}
        relationship_tokens = {
            "relationship",
            "related",
            "relation",
            "between",
            "connected",
            "connection",
            "with",
        }
        filler_tokens = {"happens", "happen", "scene", "scenes", "event", "events", "comes", "come"}

        entity_groups = self._extract_entity_groups(query)
        if len(entity_groups) >= 2 and (
            token_set & relationship_tokens or token_set & why_tokens or token_set & path_tokens
        ):
            return {
                "kind": "character_relationship",
                "entity_groups": entity_groups[:2],
                "question_focus": "why" if token_set & why_tokens else "relationship",
            }
        if token_set & character_tokens:
            anchor_tokens = [
                token for token in tokens if token not in character_tokens | filler_tokens
            ]
            return {"kind": "scene_characters", "anchor_tokens": anchor_tokens}
        if token_set & where_tokens:
            anchor_tokens = [
                token
                for token in tokens
                if token not in where_tokens | filler_tokens | {"happened", "happening"}
            ]
            return {"kind": "scene_location", "anchor_tokens": anchor_tokens}
        if token_set & after_tokens:
            anchor_tokens = [
                token for token in tokens if token not in after_tokens | filler_tokens
            ]
            return {
                "kind": "scene_transition",
                "anchor_tokens": anchor_tokens,
                "direction": "after",
            }
        if token_set & before_tokens:
            anchor_tokens = [
                token for token in tokens if token not in before_tokens | filler_tokens
            ]
            return {
                "kind": "scene_transition",
                "anchor_tokens": anchor_tokens,
                "direction": "before",
            }
        return {"kind": "generic", "anchor_tokens": tokens}

    @staticmethod
    def _tokenize(query: str) -> List[str]:
        stopwords = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "of",
            "to",
            "for",
            "is",
            "are",
            "was",
            "were",
            "what",
            "which",
        }
        tokens = re.findall(r"[a-z0-9]+", str(query or "").lower())
        return [token for token in tokens if len(token) >= 2 and token not in stopwords]

    def _rerank_hits(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored_hits = []
        query_norm = " ".join(self._tokenize(query))
        query_tokens = set(self._tokenize(query))
        dialogue_query = any(token in query_tokens for token in {"quote", "dialogue", "line", "said"})
        location_query = any(token in query_tokens for token in {"where", "location", "room", "hall", "kitchen", "basement", "restaurant"})
        character_query = any(token in query_tokens for token in {"who", "character", "appears", "present"})
        relationship_query = any(
            token in query_tokens
            for token in {"relationship", "related", "relation", "connected", "connection", "why", "how"}
        )
        content_tokens = {
            token
            for token in query_tokens
            if token not in {"who", "character", "appears", "present"}
        }

        for hit in hits:
            neighbors = hit.get("neighbors") or []
            neighbor_text = " ".join(
                self._normalize_field(neighbor.get("title", "")) for neighbor in neighbors
            )
            node_type = hit.get("node_type", "")
            script_focus_text = " ".join(
                self._normalize_field(hit.get(field, ""))
                for field in ("heading", "location", "time_of_day")
            ).strip()
            field_text = " ".join(
                self._normalize_field(hit.get(field, ""))
                for field in (
                    "title",
                    "heading",
                    "location",
                    "time_of_day",
                    "scene_label",
                    "body",
                )
            )
            field_text = f"{field_text} {neighbor_text}".strip()
            if not field_text:
                continue

            overlap = sum(1 for token in query_tokens if token in field_text.split())
            contains_bonus = 0.0
            matched_content = 0
            if query_norm and query_norm in field_text:
                contains_bonus += 1.0
            if hit.get("heading") and query_norm in self._normalize_field(hit.get("heading", "")):
                contains_bonus += 1.2
            if hit.get("location") and query_norm in self._normalize_field(hit.get("location", "")):
                contains_bonus += 0.9
            if content_tokens:
                matched_content = sum(
                    1 for token in content_tokens if token in field_text.split()
                )
                if matched_content == len(content_tokens):
                    contains_bonus += 0.90
                elif matched_content >= 2:
                    contains_bonus += 0.45

            score = contains_bonus + (overlap / max(len(query_tokens), 1))
            if node_type == "ScriptSubscene":
                score += 0.45
                if dialogue_query and hit.get("body"):
                    score += 0.35
                if hit.get("is_canonical_subscene"):
                    score += 0.20
                elif hit.get("indexable") is False:
                    score -= 0.10
            elif node_type == "ScriptScene":
                score += 0.35
            elif node_type == "TemporalChunk":
                score += 0.20
            elif node_type == "KGScene":
                score += 0.15
            elif node_type == "CharacterRelation":
                score += 0.85
                if relationship_query:
                    score += 0.75
            if location_query and hit.get("location"):
                score += 0.20
            if node_type in {"ScriptScene", "ScriptSubscene"} and location_query:
                if matched_content >= 1 and any(
                    token in script_focus_text.split() for token in content_tokens
                ):
                    score += 0.55
                if content_tokens and all(
                    token in script_focus_text.split() for token in content_tokens
                ):
                    score += 0.35
            time_tokens = {
                token
                for token in query_tokens
                if token in {"day", "night", "morning", "evening", "afternoon", "dawn"}
            }
            if time_tokens:
                hit_time = self._normalize_field(hit.get("time_of_day", ""))
                hit_heading = self._normalize_field(hit.get("heading", ""))
                if any(token in hit_time or token in hit_heading for token in time_tokens):
                    score += 0.45
            if character_query and node_type in {"ScriptScene", "ScriptSubscene"}:
                character_neighbor_count = sum(
                    1
                    for neighbor in (hit.get("neighbors") or [])
                    if neighbor.get("node_type") == "Character"
                )
                score += min(0.60, 0.15 * character_neighbor_count)
            score += min(0.25, 0.03 * len(hit.get("neighbors") or []))

            hit_copy = dict(hit)
            hit_copy["score"] = round(float(score), 4)
            scored_hits.append(hit_copy)

        return sorted(scored_hits, key=lambda item: item.get("score", 0.0), reverse=True)

    @staticmethod
    def _normalize_field(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    @staticmethod
    def _looks_like_cypher(query: str) -> bool:
        stripped = str(query or "").strip().lower()
        if not stripped:
            return False
        prefixes = ("match ", "optional match ", "with ", "call ", "unwind ", "return ")
        return any(stripped.startswith(prefix) for prefix in prefixes)

    def _build_local_candidates(self, movie_id: Optional[str]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        if movie_id:
            movie_ids = [movie_id]
        else:
            movie_ids = [
                path.stem.replace("_chunks", "")
                for path in Cfg.get_temporal_chunks_dir().glob("*_chunks.json")
            ]

        for current_movie_id in movie_ids:
            chunks = self._load_json(
                Cfg.get_temporal_chunks_dir() / f"{current_movie_id}_chunks.json"
            ) or []
            chunk_lookup = {
                chunk.get("chunk_id", ""): chunk for chunk in chunks if chunk.get("chunk_id")
            }
            seen_script_scenes = set()
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    continue
                neighbors = [
                    {
                        "relation": "APPEARS_IN",
                        "title": name,
                        "node_id": self._character_node_id(current_movie_id, name),
                        "node_type": "Character",
                    }
                    for name in (chunk.get("characters", []) or [])[:6]
                ]
                neighbors.extend(
                    {
                        "relation": "ALIGNS_TO_SCRIPT_SCENE",
                        "title": ref.get("heading", ""),
                        "node_id": ref.get("script_scene_uid", ref.get("heading", "")),
                        "node_type": "ScriptScene",
                    }
                    for ref in (chunk.get("script_scene_refs", []) or [])[:4]
                    if ref.get("heading") or ref.get("script_scene_uid")
                )
                candidates.append(
                    {
                        "node_id": chunk_id,
                        "node_type": "TemporalChunk",
                        "movie_id": current_movie_id,
                        "title": chunk.get("scene_label", chunk_id),
                        "heading": chunk.get("script_primary_heading", ""),
                        "location": chunk.get("script_location", ""),
                        "time_of_day": chunk.get("script_time_of_day", ""),
                        "start_time": chunk.get("start_time", ""),
                        "end_time": chunk.get("end_time", ""),
                        "chunk_id": chunk_id,
                        "body": " ".join(
                            part
                            for part in (
                                chunk.get("description", ""),
                                chunk.get("dialogue_text", ""),
                                chunk.get("situation", ""),
                                " ".join(chunk.get("characters", []) or []),
                                " ".join(chunk.get("script_characters", []) or []),
                            )
                            if part
                        ),
                        "neighbors": neighbors,
                    }
                )

                for ref in chunk.get("script_scene_refs", []) or []:
                    script_scene_uid = ref.get("script_scene_uid")
                    if not script_scene_uid or script_scene_uid in seen_script_scenes:
                        continue
                    seen_script_scenes.add(script_scene_uid)
                    candidates.append(
                        {
                            "node_id": script_scene_uid,
                            "node_type": "ScriptScene",
                            "movie_id": current_movie_id,
                            "title": ref.get("heading", script_scene_uid),
                            "heading": ref.get("heading", ""),
                            "location": ref.get("location", ""),
                            "time_of_day": ref.get("time_of_day", ""),
                            "body": " ".join(
                                part
                                for part in (
                                    ref.get("heading", ""),
                                    ref.get("location", ""),
                                    " ".join(ref.get("characters", []) or []),
                                    ref.get("anchor_quality", ""),
                                )
                                if part
                            ),
                            "neighbors": [
                                {
                                    "relation": "APPEARS_IN",
                                    "title": name,
                                    "node_id": self._character_node_id(current_movie_id, name),
                                    "node_type": "Character",
                                }
                                for name in (ref.get("characters", []) or [])[:6]
                            ],
                        }
                    )

            subscenes = self._load_json(
                Cfg.get_script_subscenes_dir()
                / f"{current_movie_id}_script_subscenes.json"
            ) or []
            for subscene in subscenes:
                subscene_id = subscene.get("subscene_id")
                if not subscene_id:
                    continue
                candidates.append(
                    {
                        "node_id": subscene_id,
                        "node_type": "ScriptSubscene",
                        "movie_id": current_movie_id,
                        "title": subscene.get("script_heading", subscene_id),
                        "heading": subscene.get("script_heading", ""),
                        "location": subscene.get("script_location", ""),
                        "time_of_day": subscene.get("script_time_of_day", ""),
                        "start_time": subscene.get("start_time", ""),
                        "end_time": subscene.get("end_time", ""),
                        "chunk_id": subscene.get("parent_chunk_id", ""),
                        "indexable": bool(subscene.get("indexable")),
                        "is_canonical_subscene": bool(
                            subscene.get("is_canonical_subscene")
                        ),
                        "body": " ".join(
                            part
                            for part in (
                                subscene.get("dialogue_excerpt", ""),
                                subscene.get("semantic_description", ""),
                                " ".join(subscene.get("script_characters", []) or []),
                            )
                            if part
                        ),
                        "neighbors": [
                            {
                                "relation": "DERIVED_FROM",
                                "title": subscene.get("script_heading", ""),
                                "node_id": subscene.get("script_scene_uid", ""),
                                "node_type": "ScriptScene",
                            }
                        ]
                        + [
                            {
                                "relation": "APPEARS_IN",
                                "title": name,
                                "node_id": self._character_node_id(current_movie_id, name),
                                "node_type": "Character",
                            }
                            for name in (subscene.get("script_characters", []) or [])[:6]
                        ]
                        + [
                            {
                                "relation": "APPEARS_IN_SUBSCENE",
                                "title": name,
                                "node_id": self._character_node_id(current_movie_id, name),
                                "node_type": "Character",
                            }
                            for name in (
                                chunk_lookup.get(subscene.get("parent_chunk_id", ""), {}).get("characters", [])
                                or []
                            )[:6]
                        ],
                    }
                )

            kg_graph = self._load_graph(Cfg.get_index_dir() / f"{current_movie_id}_kg.graphml")
            if kg_graph is not None:
                for node_id, data in kg_graph.nodes(data=True):
                    if data.get("movie_id") != current_movie_id:
                        continue
                    neighbors = []
                    for neighbor in list(kg_graph.neighbors(node_id))[:6]:
                        neighbor_data = kg_graph.nodes[neighbor]
                        neighbors.append(
                            {
                                "relation": "RELATED_TO",
                                "title": neighbor_data.get("name")
                                or neighbor_data.get("title")
                                or neighbor,
                                "node_id": neighbor,
                                "node_type": neighbor_data.get("type", "Entity"),
                            }
                        )
                    candidates.append(
                        {
                            "node_id": node_id,
                            "node_type": self._kg_label(data.get("type", "Entity")),
                            "movie_id": current_movie_id,
                            "title": data.get("name", node_id),
                            "heading": data.get("heading", ""),
                            "location": data.get("location", ""),
                            "chunk_id": data.get("chunk_id", ""),
                            "body": " ".join(
                                part
                                for part in (
                                    data.get("description", ""),
                                    data.get("text", ""),
                                )
                                if part
                            ),
                            "neighbors": neighbors,
                        }
                    )

        return candidates
