"""
Graph Indexer for MovieRAG, implementing VideoRAG's Dual-Channel GraphRAG approach.

Extracts Entities and Relationships from transcripts using Gemini to build
a Semantic Knowledge Graph for high-level reasoning queries.
"""

import os
import json
import logging
import re
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional

from movierag.generation.universal_client import (
    LLMRateLimitError,
    UniversalLLMClient,
    is_rate_limit_error,
)

logger = logging.getLogger(__name__)


class GraphIndexer:
    """
    Indexes textual knowledge from movie transcripts into a Graph structure.
    Uses Gemini to extract Entities and Relationships, creating nodes and edges.
    """

    def __init__(
        self,
        index_dir: str,
        index_name: str = "movie_graph",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Graph Indexer.

        Args:
            index_dir: Directory to store/load index files
            index_name: Base name for index files
            api_key: Google Gemini API key
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_name = index_name
        self.graph_path = self.index_dir / f"{index_name}.graphml"

        # Use the unified client for all LLM interactions
        self.client = UniversalLLMClient()
        self.graph = nx.Graph()
        self._is_loaded = False

    @staticmethod
    def _normalize_key(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", " ", str(value or "")).strip().upper()

    @classmethod
    def _is_generic_entity_name(cls, name: str, entity_type: str) -> bool:
        normalized = cls._normalize_key(name)
        if not normalized:
            return True
        generic_tokens = {
            "Character": {
                "MAN",
                "WOMAN",
                "BOY",
                "GIRL",
                "PERSON",
                "NARRATOR",
                "VOICE",
                "UNKNOWN",
                "VISITOR",
                "CLASS",
                "TEACHER",
                "MOTHER",
                "FATHER",
            },
            "Location": {
                "ROOM",
                "HOUSE",
                "HOME",
                "HALL",
                "HALLWAY",
                "STREET",
                "INTERIOR",
                "EXTERIOR",
            },
            "Object": {"OBJECT", "THING", "ITEM"},
        }
        strong_tokens = [
            token
            for token in normalized.split()
            if token not in {"INT", "EXT", "THE", "A", "AN"}
        ]
        if not strong_tokens:
            return True
        blocked = generic_tokens.get(entity_type, set())
        return all(token in blocked for token in strong_tokens)

    @staticmethod
    def _normalize_relation_label(value: str) -> str:
        relation = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").upper()).strip("_")
        if not relation:
            return "RELATED_TO"
        if relation[0].isdigit():
            relation = f"REL_{relation}"
        return relation

    def _build_scene_hints(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        characters: List[str] = []
        for name in (scene.get("characters", []) or []) + (scene.get("script_characters", []) or []):
            text = str(name or "").strip()
            if text and text not in characters:
                characters.append(text)
        for cast in scene.get("cast_in_scene", []) or []:
            text = str(cast.get("character", "") or "").strip()
            if text and text not in characters:
                characters.append(text)

        locations: List[str] = []
        for value in (
            scene.get("script_location", ""),
            scene.get("scene_label", ""),
            scene.get("script_primary_heading", ""),
        ):
            text = str(value or "").strip()
            if text and text not in locations:
                locations.append(text)

        objects = scene.get("vision_objects", []) or []
        if isinstance(objects, str):
            objects = [part.strip() for part in re.split(r"[,|]", objects) if part.strip()]
        object_names: List[str] = []
        for value in objects:
            text = str(value or "").strip()
            if text and text not in object_names:
                object_names.append(text)

        return {
            "characters": characters,
            "locations": locations,
            "objects": object_names,
        }

    def _best_hint_match(
        self, raw_name: str, candidates: List[str], entity_type: str
    ) -> str:
        normalized = self._normalize_key(raw_name)
        if not normalized or not candidates:
            return ""
        raw_tokens = set(normalized.split())
        best_name = ""
        best_score = None
        for candidate in candidates:
            candidate_norm = self._normalize_key(candidate)
            candidate_tokens = set(candidate_norm.split())
            if not candidate_tokens:
                continue
            overlap = len(raw_tokens & candidate_tokens)
            if not overlap:
                continue
            exact_bonus = 3 if candidate_norm == normalized else 0
            subset_bonus = 2 if raw_tokens <= candidate_tokens or candidate_tokens <= raw_tokens else 0
            score = (overlap + exact_bonus + subset_bonus, len(candidate_tokens), len(candidate_norm))
            if best_score is None or score > best_score:
                best_score = score
                best_name = candidate
        if best_score and best_score[0] >= 2:
            return best_name
        if entity_type == "Character" and len(raw_tokens) == 1:
            for candidate in candidates:
                candidate_norm = self._normalize_key(candidate)
                if normalized in candidate_norm.split():
                    return candidate
        return ""

    def _canonicalize_hint_name(
        self, raw_name: str, entity_type: str, scene_hints: Dict[str, Any]
    ) -> str:
        raw_name = str(raw_name or "").strip()
        if not raw_name:
            return ""
        candidate_pool = scene_hints.get(f"{entity_type.lower()}s", []) or []
        matched = self._best_hint_match(raw_name, candidate_pool, entity_type)
        return matched or raw_name

    def _sanitize_graph_data(
        self, graph_data: Dict[str, Any], scene_hints: Dict[str, Any]
    ) -> Dict[str, Any]:
        sanitized_entities: List[Dict[str, Any]] = []
        entity_lookup: Dict[str, Dict[str, Any]] = {}

        # Seed deterministic hints first so generic LLM aliases collapse onto them.
        for entity_type, values in (
            ("Character", scene_hints.get("characters", []) or []),
            ("Location", scene_hints.get("locations", []) or []),
            ("Object", scene_hints.get("objects", []) or []),
        ):
            for value in values:
                canonical_name = str(value or "").strip()
                if not canonical_name or self._is_generic_entity_name(canonical_name, entity_type):
                    continue
                key = f"{entity_type}:{self._normalize_key(canonical_name)}"
                if key not in entity_lookup:
                    entity = {
                        "name": canonical_name,
                        "type": entity_type,
                        "description": "",
                    }
                    entity_lookup[key] = entity
                    sanitized_entities.append(entity)

        for ent in graph_data.get("entities", []) or []:
            raw_name = str(ent.get("name", "") or "").strip()
            entity_type = str(ent.get("type", "Object") or "Object").title()
            if entity_type not in {"Character", "Location", "Object"}:
                entity_type = "Object"
            canonical_name = self._canonicalize_hint_name(raw_name, entity_type, scene_hints)
            if not canonical_name or self._is_generic_entity_name(canonical_name, entity_type):
                continue
            key = f"{entity_type}:{self._normalize_key(canonical_name)}"
            entity = entity_lookup.get(key)
            description = str(ent.get("description", "") or "").strip()
            if entity is None:
                entity = {
                    "name": canonical_name,
                    "type": entity_type,
                    "description": description,
                }
                entity_lookup[key] = entity
                sanitized_entities.append(entity)
            elif description and not entity.get("description"):
                entity["description"] = description

        valid_entity_names = {
            self._normalize_key(entity["name"]): entity["name"] for entity in sanitized_entities
        }
        sanitized_relationships: List[Dict[str, Any]] = []
        seen_edges = set()
        for rel in graph_data.get("relationships", []) or []:
            source = self._canonicalize_hint_name(
                rel.get("source", ""), "Character", scene_hints
            ) or str(rel.get("source", "") or "").strip()
            target = self._canonicalize_hint_name(
                rel.get("target", ""), "Character", scene_hints
            ) or str(rel.get("target", "") or "").strip()
            source_key = self._normalize_key(source)
            target_key = self._normalize_key(target)
            if source_key not in valid_entity_names or target_key not in valid_entity_names:
                continue
            if source_key == target_key:
                continue
            relation = self._normalize_relation_label(
                rel.get("relation", rel.get("description", "RELATED_TO"))
            )
            edge_key = tuple(sorted((source_key, target_key))) + (relation,)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            sanitized_relationships.append(
                {
                    "source": valid_entity_names[source_key],
                    "target": valid_entity_names[target_key],
                    "relation": relation,
                    "description": str(rel.get("description", "") or "").strip(),
                }
            )

        return {
            "entities": sanitized_entities,
            "relationships": sanitized_relationships,
        }

    def extract_entities_and_relations(
        self, text: str, scene_hints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Use LLM to extract Graph Data from a text chunk."""
        if not self.client:
            return {"entities": [], "relationships": []}

        scene_hints = scene_hints or {}
        hint_lines = []
        if scene_hints.get("characters"):
            hint_lines.append(
                "Known canonical characters for this scene: "
                + ", ".join(scene_hints["characters"][:10])
            )
        if scene_hints.get("locations"):
            hint_lines.append(
                "Location hints: " + ", ".join(scene_hints["locations"][:6])
            )
        if scene_hints.get("objects"):
            hint_lines.append("Object hints: " + ", ".join(scene_hints["objects"][:8]))
        hints_block = "\n".join(hint_lines)

        prompt = f"""
        Extract key entities (Characters, Locations, Objects) and their relationships from the following movie transcript.
        Return ONLY a JSON object with this exact structure:
        {{
            "entities": [
                {{"name": "Entity Name", "type": "Character/Location/Object", "description": "Brief description"}}
            ],
            "relationships": [
                {{"source": "Entity 1", "target": "Entity 2", "relation": "SHORT_RELATION_LABEL", "description": "How they relate or interact"}}
            ]
        }}

        Rules:
        - Prefer canonical names from the provided hints whenever possible.
        - Omit generic labels such as MAN, WOMAN, BOY, GIRL, VOICE, ROOM, HOUSE unless they can be resolved to a named entity from hints.
        - Do not invent actors as entities unless they are explicitly acting inside the story world.
        - Keep relation labels short and reusable, for example: INTERACTS_WITH, CONFRONTS, HELPS, THREATENS, PROTECTS, LOCATED_IN.

        Hints:
        {hints_block}
        
        Transcript:
        {text[:4000]}
        """
        try:
            # UniversalLLMClient exposes .models.generate_content (direct or via mock)
            # Use 'kimi' (Groq) by default per user request to limit Gemini
            response = self.client.models.generate_content(
                model="kimi",
                contents=prompt,
                response_mime_type="application/json"
            )
            
            text = response.text if hasattr(response, "text") else str(response)
            
            # Clean up JSON if not perfectly returned (common with Groq/Kimi fallback)
            import re
            json_str = text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                
            return json.loads(json_str)
        except Exception as e:
            if is_rate_limit_error(e):
                raise LLMRateLimitError(str(e)) from e
            logger.error(f"Failed to extract graph data: {e}")
            return {"entities": [], "relationships": []}

    def build_cross_modal_index(self, scene_data: List[Dict[str, Any]]) -> None:
        """
        Build KG from fused visual and textual scene data.
        Args:
            scene_data: List of dicts with {scene_idx, visual_description, transcript, movie_id}
        """
        logger.info(f"🕸️ Building Cross-Modal Knowledge Graph from {len(scene_data)} scenes...")

        if self.graph_path.exists() and not self._is_loaded:
            try:
                self.load()
                logger.info(
                    "↻ Resuming KG build with %s existing nodes.",
                    self.graph.number_of_nodes(),
                )
            except Exception as exc:
                logger.warning(f"Could not resume existing KG graph: {exc}")

        for scene in scene_data:
            scene_idx = scene["scene_idx"]
            scene_node_id = f"{scene.get('movie_id', 'unknown')}_SCENE_{scene_idx}"
            if self.graph.has_node(scene_node_id):
                continue
            chunk_id = scene.get("chunk_id", "")
            visual_desc = scene.get("visual_description", "")
            transcript = scene.get("transcript", "")
            movie_id = scene.get("movie_id", "unknown")
            scene_label = scene.get("scene_label", "")
            script_heading = scene.get("script_primary_heading", "")
            script_location = scene.get("script_location", "")
            scene_hints = self._build_scene_hints(scene)
            
            # Fuse knowledge
            fused_text = f"SCENE {scene_idx} ANALYSIS:\n"
            if scene_label:
                fused_text += f"SCENE_LABEL: {scene_label}\n"
            if script_heading or script_location:
                fused_text += (
                    f"SCRIPT_CONTEXT: {script_heading} | {script_location}\n"
                )
            if visual_desc:
                fused_text += f"VISUAL: {visual_desc}\n"
            if scene.get("vision_setting"):
                fused_text += f"VISUAL_SETTING: {scene.get('vision_setting')}\n"
            if scene.get("vision_actions"):
                fused_text += f"VISUAL_ACTIONS: {scene.get('vision_actions')}\n"
            if scene.get("screenplay_context_excerpt"):
                fused_text += (
                    f"SCREENPLAY_CONTEXT: {scene.get('screenplay_context_excerpt')}\n"
                )
            if scene.get("description"):
                fused_text += f"SEMANTIC_DESCRIPTION: {scene.get('description')}\n"
            if scene.get("situation"):
                fused_text += f"SITUATION: {scene.get('situation')}\n"
            if transcript:
                fused_text += f"DIALOGUE: {transcript}\n"
            if scene_hints.get("characters"):
                fused_text += (
                    "KNOWN_CHARACTERS: "
                    + ", ".join(scene_hints["characters"][:10])
                    + "\n"
                )
            if scene_hints.get("locations"):
                fused_text += (
                    "KNOWN_LOCATIONS: "
                    + ", ".join(scene_hints["locations"][:6])
                    + "\n"
                )
            if scene_hints.get("objects"):
                fused_text += (
                    "KNOWN_OBJECTS: "
                    + ", ".join(scene_hints["objects"][:8])
                    + "\n"
                )
                
            # Extract Graph Data
            try:
                graph_data = self.extract_entities_and_relations(
                    fused_text, scene_hints=scene_hints
                )
            except LLMRateLimitError:
                self.save()
                raise
            graph_data = self._sanitize_graph_data(graph_data, scene_hints)
            
            # 1. Add Scene Node
            self.graph.add_node(
                scene_node_id,
                type="SceneChunk",
                text=fused_text,
                movie_id=movie_id,
                scene_idx=scene_idx,
                chunk_id=chunk_id,
                scene_label=scene_label,
                script_heading=script_heading,
                script_location=script_location,
                characters=scene_hints.get("characters", []),
                locations=scene_hints.get("locations", []),
            )

            for character_name in scene_hints.get("characters", []):
                ent_id = f"{movie_id}_{character_name}".upper()
                if not self.graph.has_node(ent_id):
                    self.graph.add_node(
                        ent_id,
                        name=character_name,
                        type="Character",
                        description="",
                        movie_id=movie_id,
                    )
                self.graph.add_edge(ent_id, scene_node_id, relation="APPEARS_IN")

            for location_name in scene_hints.get("locations", [])[:1]:
                location_id = f"{movie_id}_LOCATION_{self._normalize_key(location_name).replace(' ', '_')}"
                if not self.graph.has_node(location_id):
                    self.graph.add_node(
                        location_id,
                        name=location_name,
                        type="Location",
                        description="",
                        movie_id=movie_id,
                    )
                self.graph.add_edge(location_id, scene_node_id, relation="SET_IN")
            
            # 2. Add Entities & Relations
            for ent in graph_data.get("entities", []):
                ent_name = ent.get("name", "Unknown")
                ent_type = ent.get("type", "Unknown")
                if ent_type == "Character":
                    ent_id = f"{movie_id}_{ent_name}".upper()
                else:
                    ent_id = f"{movie_id}_{ent_type}_{self._normalize_key(ent_name).replace(' ', '_')}"
                
                if not self.graph.has_node(ent_id):
                    self.graph.add_node(
                        ent_id,
                        name=ent_name,
                        type=ent_type,
                        description=ent.get("description", ""),
                        movie_id=movie_id
                    )
                
                # Relation to Scene
                scene_relation = "SET_IN" if ent_type == "Location" else "APPEARS_IN"
                self.graph.add_edge(ent_id, scene_node_id, relation=scene_relation)
                
            for rel in graph_data.get("relationships", []):
                src_name = rel.get("source")
                tgt_name = rel.get("target")
                src = None
                tgt = None
                for node_id, data in self.graph.nodes(data=True):
                    if data.get("movie_id") != movie_id:
                        continue
                    if self._normalize_key(data.get("name", "")) == self._normalize_key(src_name):
                        src = node_id
                    if self._normalize_key(data.get("name", "")) == self._normalize_key(tgt_name):
                        tgt = node_id
                    if src and tgt:
                        break
                
                if self.graph.has_node(src) and self.graph.has_node(tgt):
                    self.graph.add_edge(
                        src, tgt, 
                        relation=rel.get("relation", self._normalize_relation_label(rel.get("description", "INTERACTS"))),
                        description=rel.get("description", ""),
                        scene_idx=scene_idx
                    )

            self.save()

        self.save()
        logger.info(f"✅ KG build complete: {self.graph.number_of_nodes()} nodes.")

    def save(self):
        """Save networkx graph to disk."""
        sanitized_graph = nx.Graph()
        for node_id, data in self.graph.nodes(data=True):
            sanitized_graph.add_node(
                node_id,
                **{key: value for key, value in data.items() if value is not None},
            )
        for source, target, data in self.graph.edges(data=True):
            sanitized_graph.add_edge(
                source,
                target,
                **{key: value for key, value in data.items() if value is not None},
            )
        nx.write_graphml(sanitized_graph, str(self.graph_path))
        logger.info(f"Saved Graph index to {self.graph_path}")

    def load(self) -> bool:
        """Load networkx graph from disk."""
        if not self.graph_path.exists():
            logger.warning(f"Graph file not found at {self.graph_path}")
            return False
        self.graph = nx.read_graphml(str(self.graph_path))
        self._is_loaded = True
        logger.info(f"Loaded Graph index with {self.graph.number_of_nodes()} nodes")
        return True

    def ensure_loaded(self):
        if not self._is_loaded:
            if not self.load():
                raise RuntimeError("Graph index not found. Build index first.")

    def search(
        self, query: str, k: int = 5, movie_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Graph-based retrieval.
        1. Extract entities from query
        2. Match to graph nodes
        3. Traverse 1-hop to find heavily connected SceneChunks
        """
        self.ensure_loaded()
        if not self.client:
            logger.warning("No LLM client for query entity extraction.")
            return []

        # 1. Extract entities from the user's query
        prompt = f"Extract the key entities (Characters, Locations, Objects) from this query. Return ONLY a JSON list of strings. Query: {query}"
        try:
            # Use 'kimi' (Groq)
            resp = self.client.models.generate_content(
                model="kimi",
                contents=prompt,
                response_mime_type="application/json"
            )
            text = resp.text if hasattr(resp, "text") else str(resp)
            
            import re
            json_str = text
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                
            query_entities = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to extract entities from query: {e}")
            return []

        # 2. Find matching nodes in the graph
        found_nodes = []
        for ent in query_entities:
            ent_lower = ent.lower()
            for node, data in self.graph.nodes(data=True):
                if data.get("type") != "SceneChunk":
                    name = str(data.get("name", "")).lower()
                    if ent_lower in name or name in ent_lower:
                        if movie_id and data.get("movie_id") != movie_id:
                            continue
                        found_nodes.append(node)

        # 3. Traverse 1 hop to get SceneChunks
        chunk_scores = {}
        for node in found_nodes:
            for neighbor in self.graph.neighbors(node):
                if self.graph.nodes[neighbor].get("type") == "SceneChunk":
                    # Weight by number of query entities connected to this chunk
                    chunk_scores[neighbor] = chunk_scores.get(neighbor, 0) + 1

        # Sort chunks by number of entity matches
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        results = []
        for chunk_id, score in sorted_chunks:
            data = self.graph.nodes[chunk_id]
            results.append(
                {
                    "clip_id": chunk_id,
                    "text": data.get("text", ""),
                    "movie_id": data.get("movie_id", ""),
                    "score": score,
                }
            )

        return results
