"""
Knowledge Graph Enrichment Builder

Loads temporal chunks and enriches the NetworkX knowledge graph with
TemporalChunk nodes and relational edges.

Adapted from: scripts/enrich_graph_temporal.py + scripts/build_graph_index.py
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import networkx as nx

from preprocess_data.config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Enrich the knowledge graph with temporal chunk nodes and edges."""

    def __init__(self, graph_path: str = None):
        self.graph_path = Path(graph_path) if graph_path else Cfg.get_graph_path()

    def enrich(self, chunks_path: Path = None, movie_id: str = None) -> Dict:
        """
        Add TemporalChunk nodes + edges to the knowledge graph.

        Edge types:
          (:TemporalChunk)-[:BELONGS_TO]->(:Movie)
          (:Character)-[:APPEARS_IN]->(:TemporalChunk)
          (:TemporalChunk)-[:FOLLOWED_BY]->(:TemporalChunk)
          (:Actor)-[:PLAYS]->(:Character)
        """
        # Load graph
        if self.graph_path.exists():
            G = nx.read_graphml(str(self.graph_path))
            logger.info(
                f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )
        else:
            G = nx.Graph()
            logger.info("Creating new graph")

        # Load chunks
        chunks_path = chunks_path or (Cfg.get_temporal_chunks_dir() / "all_chunks.json")
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = data.get("chunks", []) if isinstance(data, dict) else data

        if movie_id:
            chunks = [c for c in chunks if c["movie_id"] == movie_id]

        # Group by movie for FOLLOWED_BY edges
        movie_chunks: Dict[str, List[Dict]] = {}
        for c in chunks:
            movie_chunks.setdefault(c["movie_id"], []).append(c)
        for mid in movie_chunks:
            movie_chunks[mid].sort(key=lambda c: c.get("start_seconds", 0))

        stats = {"chunks": 0, "movies": 0, "characters": 0, "edges": 0}

        for mid, m_chunks in movie_chunks.items():
            # Movie node
            movie_node = f"MOVIE_{mid}"
            if not G.has_node(movie_node):
                G.add_node(
                    movie_node,
                    type="Movie",
                    name=m_chunks[0].get("title", mid),
                    movie_id=mid,
                )
                stats["movies"] += 1

            prev_id = None
            for chunk in m_chunks:
                cid = chunk["chunk_id"]

                # TemporalChunk node
                if not G.has_node(cid):
                    G.add_node(
                        cid,
                        type="TemporalChunk",
                        movie_id=mid,
                        start_time=chunk.get("start_time", ""),
                        end_time=chunk.get("end_time", ""),
                        description=chunk.get("description", "")[:500],
                        situation=chunk.get("situation", ""),
                        dialogue_text=chunk.get("dialogue_text", "")[:500],
                    )
                    stats["chunks"] += 1

                # BELONGS_TO
                if not G.has_edge(cid, movie_node):
                    G.add_edge(cid, movie_node, relation="BELONGS_TO")
                    stats["edges"] += 1

                # APPEARS_IN: Character → Chunk
                for name in chunk.get("characters", []):
                    char_id = f"{mid}_{name}".upper()
                    if not G.has_node(char_id):
                        G.add_node(char_id, type="Character", name=name, movie_id=mid)
                        stats["characters"] += 1
                    if not G.has_edge(char_id, cid):
                        G.add_edge(char_id, cid, relation="APPEARS_IN")
                        stats["edges"] += 1

                # PLAYS: Actor → Character
                for cast in chunk.get("cast_in_scene", []):
                    actor, char = cast.get("actor", ""), cast.get("character", "")
                    if actor and char:
                        a_id = f"{mid}_{actor}".upper()
                        c_id = f"{mid}_{char}".upper()
                        if not G.has_node(a_id):
                            G.add_node(a_id, type="Actor", name=actor, movie_id=mid)
                        if not G.has_edge(a_id, c_id):
                            G.add_edge(a_id, c_id, relation="PLAYS")
                            stats["edges"] += 1

                # FOLLOWED_BY
                if prev_id and not G.has_edge(prev_id, cid):
                    G.add_edge(prev_id, cid, relation="FOLLOWED_BY")
                    stats["edges"] += 1
                prev_id = cid

        # Save
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G, str(self.graph_path))

        logger.info(
            f"  ✅ Graph enriched: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        logger.info(
            f"     New: {stats['chunks']} chunks, {stats['characters']} characters, {stats['edges']} edges"
        )
        return stats
