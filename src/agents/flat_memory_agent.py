"""
Flat Memory Agent - Baseline 1

Simple vector database with no tiers, no consolidation, no forgetting.
All memories are stored and retrieved equally.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import time

from .base_agent import BaseMemoryAgent, Memory, RetrievalResult, AgentFactory


@dataclass
class VectorMemory(Memory):
    """Memory with embedding vector."""
    embedding: Optional[np.ndarray] = None


class FlatMemoryAgent(BaseMemoryAgent):
    """
    Baseline: Flat vector memory with no consolidation.

    - All memories stored equally (no tiers)
    - Simple cosine similarity retrieval
    - No forgetting, no consolidation
    - No salience or emotional scoring
    """

    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.memories: List[VectorMemory] = []
        self.embedding_dim = config.get("embedding_dim", 384) if config else 384
        self._embedding_model = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        Uses sentence-transformers if available, else random (for testing).
        """
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                # Fallback to random embeddings for testing
                self._embedding_model = "random"

        if self._embedding_model == "random":
            # Deterministic random based on text hash for consistency
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.embedding_dim).astype(np.float32)
        else:
            return self._embedding_model.encode(text, convert_to_numpy=True)

    def store(self, content: str, metadata: Dict[str, Any] = None) -> Memory:
        """Store a memory with its embedding."""
        memory_id = self._generate_memory_id(content)
        embedding = self._get_embedding(content)

        memory = VectorMemory(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            memory_type="flat",  # No tier distinction
            metadata=metadata or {},
            significance=0.5,  # All equal
            access_count=0,
            last_accessed=None,
            embedding=embedding
        )

        self.memories.append(memory)
        return memory

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Retrieve top-k memories by cosine similarity."""
        start_time = time.time()

        if not self.memories:
            return RetrievalResult(
                memories=[],
                query=query,
                retrieval_time_ms=0,
                metadata={"reason": "no_memories"}
            )

        query_embedding = self._get_embedding(query)

        # Calculate cosine similarities
        similarities = []
        for memory in self.memories:
            if memory.embedding is not None:
                sim = np.dot(query_embedding, memory.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-8
                )
                similarities.append((memory, sim))

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # Update access counts
        retrieved_memories = []
        for memory, sim in top_k:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            memory.metadata["last_similarity"] = float(sim)
            retrieved_memories.append(memory)

        retrieval_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=retrieved_memories,
            query=query,
            retrieval_time_ms=retrieval_time,
            metadata={
                "total_memories": len(self.memories),
                "similarities": [s for _, s in top_k]
            }
        )

    def consolidate(self) -> Dict[str, Any]:
        """No consolidation for flat memory baseline."""
        return {
            "consolidated": False,
            "reason": "flat_memory_baseline",
            "memories_count": len(self.memories)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_size = sum(len(m.content) for m in self.memories)
        embedding_size = len(self.memories) * self.embedding_dim * 4  # float32

        return {
            "agent_type": "flat_memory",
            "total_memories": len(self.memories),
            "total_content_chars": total_size,
            "embedding_size_bytes": embedding_size,
            "total_size_bytes": total_size + embedding_size,
            "avg_memory_length": total_size / len(self.memories) if self.memories else 0,
            "consolidation_enabled": False,
            "forgetting_enabled": False,
            "tiers": ["flat"]  # Single tier
        }

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []
        self.action_history = []


# Register with factory
AgentFactory.register("flat_memory", FlatMemoryAgent)
