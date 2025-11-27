"""
Memory Agent Implementations

Provides all agent types for the memory consolidation experiments:
- ConsolidationAgent: Experimental agent with multi-tier memory
- FlatMemoryAgent: Baseline - simple vector store
- RAGOnlyAgent: Baseline - graph-based like Mem0
- FullContextAgent: Upper bound - unlimited context
- NoMemoryAgent: Lower bound - no persistence
"""

from .base_agent import (
    BaseMemoryAgent,
    Memory,
    ActionOutcome,
    RetrievalResult,
    AgentFactory
)

from .consolidation_agent import ConsolidationAgent
from .flat_memory_agent import FlatMemoryAgent
from .rag_only_agent import RAGOnlyAgent
from .full_context_agent import FullContextAgent
from .no_memory_agent import NoMemoryAgent

__all__ = [
    # Base classes
    "BaseMemoryAgent",
    "Memory",
    "ActionOutcome",
    "RetrievalResult",
    "AgentFactory",
    # Agent implementations
    "ConsolidationAgent",
    "FlatMemoryAgent",
    "RAGOnlyAgent",
    "FullContextAgent",
    "NoMemoryAgent",
]

# Verify all agents are registered
_expected_agents = ["consolidation", "flat_memory", "rag_only", "full_context", "no_memory"]
_registered = AgentFactory.available_agents()

for agent_type in _expected_agents:
    if agent_type not in _registered:
        raise ImportError(f"Agent '{agent_type}' not registered with AgentFactory")
