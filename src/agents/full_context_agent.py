"""
Full Context Agent - Baseline 3 (Upper Bound)

Keeps entire conversation history, no retrieval needed.
Expensive but comprehensive - serves as upper bound.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import time

from .base_agent import BaseMemoryAgent, Memory, RetrievalResult, AgentFactory


class FullContextAgent(BaseMemoryAgent):
    """
    Upper Bound Baseline: Full context window.

    - Stores all interactions in chronological order
    - "Retrieval" returns everything (no filtering)
    - No forgetting, no consolidation
    - Expensive: O(n) context growth

    This establishes the upper bound on what's achievable
    if you have unlimited context window.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.memories: List[Memory] = []
        self.max_context_tokens = config.get("max_context_tokens", 100000) if config else 100000
        self._approx_tokens = 0

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4

    def store(self, content: str, metadata: Dict[str, Any] = None) -> Memory:
        """Store memory in chronological order."""
        memory_id = self._generate_memory_id(content)
        tokens = self._estimate_tokens(content)

        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            memory_type="full_context",
            metadata=metadata or {},
            significance=1.0,  # Everything is kept
            access_count=0,
            last_accessed=None
        )
        memory.metadata["tokens"] = tokens

        self.memories.append(memory)
        self._approx_tokens += tokens

        return memory

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Return all memories (full context).
        The 'k' parameter is ignored - we return everything.
        """
        start_time = time.time()

        # Mark all as accessed
        for memory in self.memories:
            memory.access_count += 1
            memory.last_accessed = datetime.now()

        retrieval_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=self.memories.copy(),  # Return all
            query=query,
            retrieval_time_ms=retrieval_time,
            metadata={
                "total_memories": len(self.memories),
                "total_tokens": self._approx_tokens,
                "retrieval_type": "full_context",
                "note": "All memories returned regardless of query"
            }
        )

    def get_context_string(self) -> str:
        """Get full context as a single string."""
        parts = []
        for i, memory in enumerate(self.memories):
            timestamp = memory.timestamp.strftime("%Y-%m-%d %H:%M")
            parts.append(f"[{i+1}] ({timestamp}) {memory.content}")
        return "\n\n".join(parts)

    def consolidate(self) -> Dict[str, Any]:
        """No consolidation - keep everything."""
        return {
            "consolidated": False,
            "reason": "full_context_baseline",
            "memories_count": len(self.memories),
            "total_tokens": self._approx_tokens
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        total_chars = sum(len(m.content) for m in self.memories)

        return {
            "agent_type": "full_context",
            "total_memories": len(self.memories),
            "total_tokens": self._approx_tokens,
            "total_chars": total_chars,
            "max_context_tokens": self.max_context_tokens,
            "context_utilization": self._approx_tokens / self.max_context_tokens,
            "consolidation_enabled": False,
            "forgetting_enabled": False,
            "tiers": ["full_context"],
            "note": "Upper bound baseline - unlimited storage"
        }

    def is_context_full(self) -> bool:
        """Check if context window is approaching limit."""
        return self._approx_tokens >= self.max_context_tokens * 0.9

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = []
        self._approx_tokens = 0
        self.action_history = []


# Register with factory
AgentFactory.register("full_context", FullContextAgent)
