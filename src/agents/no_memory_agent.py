"""
No Memory Agent - Baseline 4 (Lower Bound)

Fresh context each session - no persistence.
Establishes lower bound on performance.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import time

from .base_agent import BaseMemoryAgent, Memory, RetrievalResult, AgentFactory


class NoMemoryAgent(BaseMemoryAgent):
    """
    Lower Bound Baseline: No persistent memory.

    - Each session starts fresh
    - "Store" is a no-op
    - "Retrieve" returns empty
    - Represents agent with no memory capability

    This establishes the lower bound - how well can an agent
    do with zero memory between sessions?
    """

    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self._current_session_memories: List[Memory] = []
        self._total_discarded = 0

    def store(self, content: str, metadata: Dict[str, Any] = None) -> Memory:
        """
        Store in current session only.
        Memory is discarded when session ends.
        """
        memory_id = self._generate_memory_id(content)

        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            memory_type="ephemeral",
            metadata=metadata or {},
            significance=0.0,  # Will be forgotten
            access_count=0,
            last_accessed=None
        )

        self._current_session_memories.append(memory)
        return memory

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Retrieve from current session only.
        Cross-session retrieval returns empty.
        """
        start_time = time.time()

        # Only return current session memories (simulating within-session context)
        # In practice, this would be limited to recent conversation turns
        recent = self._current_session_memories[-k:] if self._current_session_memories else []

        retrieval_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=recent,
            query=query,
            retrieval_time_ms=retrieval_time,
            metadata={
                "session_memories": len(self._current_session_memories),
                "cross_session_available": False,
                "total_discarded": self._total_discarded,
                "note": "Only current session context available"
            }
        )

    def start_session(self) -> int:
        """Start new session, clearing previous memories."""
        # Count what we're discarding
        self._total_discarded += len(self._current_session_memories)

        # Clear session memories
        self._current_session_memories = []

        return super().start_session()

    def consolidate(self) -> Dict[str, Any]:
        """No consolidation - nothing persists."""
        return {
            "consolidated": False,
            "reason": "no_memory_baseline",
            "session_memories": len(self._current_session_memories),
            "total_discarded": self._total_discarded
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get (lack of) memory statistics."""
        session_chars = sum(len(m.content) for m in self._current_session_memories)

        return {
            "agent_type": "no_memory",
            "current_session_memories": len(self._current_session_memories),
            "current_session_chars": session_chars,
            "total_discarded": self._total_discarded,
            "persistent_memories": 0,
            "consolidation_enabled": False,
            "forgetting_enabled": True,  # Everything is forgotten!
            "tiers": [],  # No tiers
            "note": "Lower bound baseline - no cross-session memory"
        }

    def clear(self) -> None:
        """Clear current session."""
        self._total_discarded += len(self._current_session_memories)
        self._current_session_memories = []
        self.action_history = []


# Register with factory
AgentFactory.register("no_memory", NoMemoryAgent)
