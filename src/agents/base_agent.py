"""
Base Agent Class for Memory Consolidation Experiments

All agents (consolidation and baselines) inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import hashlib


@dataclass
class Memory:
    """A single memory unit."""
    id: str
    content: str
    timestamp: datetime
    memory_type: str  # working, episodic, semantic, procedural
    metadata: Dict[str, Any] = field(default_factory=dict)
    significance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "significance": self.significance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }


@dataclass
class ActionOutcome:
    """Record of an action and its outcome."""
    action_type: str
    action_description: str
    expected_result: str
    actual_result: str
    success_score: float  # 0.0 to 1.0
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    memories: List[Memory]
    query: str
    retrieval_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMemoryAgent(ABC):
    """
    Abstract base class for all memory agents in the experiment.

    Defines the interface that all agents must implement:
    - store(): Store a new memory
    - retrieve(): Retrieve relevant memories for a query
    - consolidate(): Run consolidation (no-op for baselines)
    - get_stats(): Get memory statistics
    """

    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.created_at = datetime.now()
        self.action_history: List[ActionOutcome] = []
        self._session_id = 0

    @abstractmethod
    def store(self, content: str, metadata: Dict[str, Any] = None) -> Memory:
        """Store a new memory. Returns the created Memory object."""
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Retrieve top-k relevant memories for a query."""
        pass

    @abstractmethod
    def consolidate(self) -> Dict[str, Any]:
        """
        Run memory consolidation.
        Returns stats about what was consolidated.
        Baselines return empty dict (no consolidation).
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories (for experiment reset)."""
        pass

    def start_session(self) -> int:
        """Start a new session, returns session ID."""
        self._session_id += 1
        return self._session_id

    def record_action(self, action: ActionOutcome) -> None:
        """Record an action outcome for learning."""
        self.action_history.append(action)

    def get_action_history(self, action_type: Optional[str] = None) -> List[ActionOutcome]:
        """Get action history, optionally filtered by type."""
        if action_type:
            return [a for a in self.action_history if a.action_type == action_type]
        return self.action_history

    def _generate_memory_id(self, content: str) -> str:
        """Generate unique ID for a memory."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{self.agent_id}:{timestamp}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def export_state(self) -> Dict[str, Any]:
        """Export agent state for persistence."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "session_id": self._session_id,
            "stats": self.get_stats(),
            "action_history_count": len(self.action_history)
        }


class AgentFactory:
    """Factory for creating memory agents."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, agent_class: type) -> None:
        """Register an agent class."""
        cls._registry[name] = agent_class

    @classmethod
    def create(cls, agent_type: str, agent_id: str, config: Dict[str, Any] = None) -> BaseMemoryAgent:
        """Create an agent by type name."""
        if agent_type not in cls._registry:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(cls._registry.keys())}")
        return cls._registry[agent_type](agent_id, config)

    @classmethod
    def available_agents(cls) -> List[str]:
        """List available agent types."""
        return list(cls._registry.keys())
