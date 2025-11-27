"""
RAG-Only Agent - Baseline 2 (Mem0-style)

Graph-based memory with entity relationships.
Retrieval-focused, no consolidation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass, field
import time
import re

from .base_agent import BaseMemoryAgent, Memory, RetrievalResult, AgentFactory


@dataclass
class Entity:
    """An entity extracted from memories."""
    name: str
    entity_type: str
    mentions: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class Relationship:
    """A relationship between two entities."""
    source: str
    target: str
    relation_type: str
    strength: float = 1.0
    evidence: List[str] = field(default_factory=list)


class RAGOnlyAgent(BaseMemoryAgent):
    """
    Baseline: RAG with graph-based memory (Mem0-style).

    - Extracts entities and relationships from memories
    - Graph-based retrieval combining semantic + graph signals
    - No consolidation, no forgetting
    - No tiered memory structure
    """

    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.memories: List[Memory] = []
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.entity_to_memories: Dict[str, Set[str]] = {}  # entity -> memory IDs
        self.embedding_dim = config.get("embedding_dim", 384) if config else 384
        self._embedding_model = None
        self._embeddings: Dict[str, np.ndarray] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                self._embedding_model = "random"

        if self._embedding_model == "random":
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.embedding_dim).astype(np.float32)
        else:
            return self._embedding_model.encode(text, convert_to_numpy=True)

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Simple entity extraction using patterns.
        Returns list of (entity_name, entity_type) tuples.
        """
        entities = []

        # Extract code-related entities
        # Function names
        for match in re.finditer(r'\b(def|function|func)\s+(\w+)', text):
            entities.append((match.group(2), "function"))

        # Class names
        for match in re.finditer(r'\bclass\s+(\w+)', text):
            entities.append((match.group(1), "class"))

        # Variable assignments
        for match in re.finditer(r'\b(\w+)\s*=\s*', text):
            name = match.group(1)
            if name not in ['if', 'for', 'while', 'return', 'True', 'False']:
                entities.append((name, "variable"))

        # File paths
        for match in re.finditer(r'[\w/]+\.(py|js|ts|md|txt|json|yaml)', text):
            entities.append((match.group(0), "file"))

        # Error types
        for match in re.finditer(r'(\w+Error|\w+Exception)', text):
            entities.append((match.group(1), "error"))

        # Technical terms (capitalized words that might be concepts)
        for match in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text):
            entities.append((match.group(1), "concept"))

        return entities

    def _extract_relationships(self, text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """
        Simple relationship extraction.
        Returns list of (source, relation, target) tuples.
        """
        relationships = []
        entity_names = {e[0] for e in entities}

        # Co-occurrence relationships
        words = text.split()
        for i, word in enumerate(words):
            if word in entity_names:
                # Look for nearby entities
                window = words[max(0, i-5):min(len(words), i+6)]
                for other in window:
                    if other in entity_names and other != word:
                        relationships.append((word, "related_to", other))

        # "uses" relationships
        for match in re.finditer(r'(\w+)\s+(?:uses?|calls?|imports?)\s+(\w+)', text):
            if match.group(1) in entity_names and match.group(2) in entity_names:
                relationships.append((match.group(1), "uses", match.group(2)))

        # "returns" relationships
        for match in re.finditer(r'(\w+)\s+returns?\s+(\w+)', text):
            if match.group(1) in entity_names:
                relationships.append((match.group(1), "returns", match.group(2)))

        return relationships

    def store(self, content: str, metadata: Dict[str, Any] = None) -> Memory:
        """Store memory and update knowledge graph."""
        memory_id = self._generate_memory_id(content)

        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            memory_type="rag",
            metadata=metadata or {},
            significance=0.5,
            access_count=0,
            last_accessed=None
        )

        self.memories.append(memory)
        self._embeddings[memory_id] = self._get_embedding(content)

        # Extract and store entities
        extracted_entities = self._extract_entities(content)
        for entity_name, entity_type in extracted_entities:
            if entity_name in self.entities:
                self.entities[entity_name].mentions += 1
                self.entities[entity_name].last_seen = datetime.now()
            else:
                self.entities[entity_name] = Entity(
                    name=entity_name,
                    entity_type=entity_type
                )

            # Link entity to memory
            if entity_name not in self.entity_to_memories:
                self.entity_to_memories[entity_name] = set()
            self.entity_to_memories[entity_name].add(memory_id)

        # Extract and store relationships
        extracted_rels = self._extract_relationships(content, extracted_entities)
        for source, rel_type, target in extracted_rels:
            self.relationships.append(Relationship(
                source=source,
                target=target,
                relation_type=rel_type,
                strength=1.0,
                evidence=[memory_id]
            ))

        memory.metadata["entities"] = [e[0] for e in extracted_entities]
        return memory

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Retrieve using hybrid semantic + graph signals."""
        start_time = time.time()

        if not self.memories:
            return RetrievalResult(
                memories=[],
                query=query,
                retrieval_time_ms=0,
                metadata={"reason": "no_memories"}
            )

        query_embedding = self._get_embedding(query)
        query_entities = self._extract_entities(query)
        query_entity_names = {e[0] for e in query_entities}

        # Score each memory
        scored_memories = []
        for memory in self.memories:
            # Semantic similarity score
            mem_embedding = self._embeddings.get(memory.id)
            if mem_embedding is not None:
                semantic_score = np.dot(query_embedding, mem_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(mem_embedding) + 1e-8
                )
            else:
                semantic_score = 0.0

            # Graph-based score: entity overlap
            memory_entities = set(memory.metadata.get("entities", []))
            entity_overlap = len(query_entity_names & memory_entities)
            graph_score = entity_overlap / (len(query_entity_names) + 1)

            # Combined score (70% semantic, 30% graph)
            combined_score = 0.7 * semantic_score + 0.3 * graph_score

            scored_memories.append((memory, combined_score, semantic_score, graph_score))

        # Sort and take top-k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_memories[:k]

        # Update access counts
        retrieved_memories = []
        for memory, combined, semantic, graph in top_k:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            memory.metadata["last_scores"] = {
                "combined": float(combined),
                "semantic": float(semantic),
                "graph": float(graph)
            }
            retrieved_memories.append(memory)

        retrieval_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=retrieved_memories,
            query=query,
            retrieval_time_ms=retrieval_time,
            metadata={
                "total_memories": len(self.memories),
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "query_entities": list(query_entity_names),
                "scores": [(m.id, c) for m, c, _, _ in top_k]
            }
        )

    def consolidate(self) -> Dict[str, Any]:
        """No consolidation for RAG-only baseline."""
        return {
            "consolidated": False,
            "reason": "rag_only_baseline",
            "memories_count": len(self.memories),
            "entities_count": len(self.entities),
            "relationships_count": len(self.relationships)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get memory and graph statistics."""
        total_content_size = sum(len(m.content) for m in self.memories)
        embedding_size = len(self._embeddings) * self.embedding_dim * 4

        return {
            "agent_type": "rag_only",
            "total_memories": len(self.memories),
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": list(set(e.entity_type for e in self.entities.values())),
            "total_content_chars": total_content_size,
            "embedding_size_bytes": embedding_size,
            "avg_entities_per_memory": len(self.entities) / len(self.memories) if self.memories else 0,
            "consolidation_enabled": False,
            "forgetting_enabled": False,
            "tiers": ["rag"]
        }

    def clear(self) -> None:
        """Clear all memories and graph."""
        self.memories = []
        self.entities = {}
        self.relationships = []
        self.entity_to_memories = {}
        self._embeddings = {}
        self.action_history = []


# Register with factory
AgentFactory.register("rag_only", RAGOnlyAgent)
