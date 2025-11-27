"""
Consolidation Agent - Experimental Agent

Multi-tier memory with sleep-inspired consolidation.
This is the agent we're testing against the baselines.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import time
import math

from .base_agent import BaseMemoryAgent, Memory, RetrievalResult, ActionOutcome, AgentFactory


@dataclass
class TieredMemory(Memory):
    """Memory with tier-specific attributes."""
    embedding: Optional[np.ndarray] = None
    # Salience/emotion
    valence: float = 0.0  # -1 (negative) to +1 (positive)
    arousal: float = 0.0  # 0 (calm) to 1 (excited)
    salience: float = 0.5  # 0 (unimportant) to 1 (critical)
    # Forgetting curve
    strength: float = 1.0  # Decays over time without access
    decay_rate: float = 0.1  # Ebbinghaus k parameter
    # Consolidation
    consolidated: bool = False
    consolidated_from: List[str] = field(default_factory=list)


@dataclass
class SemanticConcept:
    """A semantic concept extracted from episodic memories."""
    id: str
    concept: str
    definition: str
    confidence: float
    source_episodes: List[str]
    related_concepts: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProceduralSkill:
    """A procedural skill learned from experiences."""
    id: str
    name: str
    category: str
    steps: List[str]
    success_rate: float = 0.5
    execution_count: int = 0
    avg_execution_time: float = 0.0
    preconditions: List[str] = field(default_factory=list)


@dataclass
class CausalLink:
    """A causal relationship between entities."""
    cause_id: str
    effect_id: str
    relationship_type: str  # direct, indirect, contributory, preventive
    strength: float
    observations: int = 1


class ConsolidationAgent(BaseMemoryAgent):
    """
    Experimental Agent: Multi-tier memory with consolidation.

    Memory Tiers:
    - Working Memory: TTL-based, high-access items
    - Episodic Memory: Time-bound experiences with significance
    - Semantic Memory: Timeless concepts/principles
    - Procedural Memory: Skills with execution tracking

    Consolidation Features:
    - Pattern extraction: Episodic → Semantic
    - Skill extraction: Repeated actions → Procedural
    - Causal discovery: Action outcomes → Causal chains
    - Forgetting curve: Ebbinghaus decay with retrieval boost
    """

    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        config = config or {}

        # Memory tiers
        self.working_memory: List[TieredMemory] = []
        self.episodic_memory: List[TieredMemory] = []
        self.semantic_memory: List[SemanticConcept] = []
        self.procedural_memory: List[ProceduralSkill] = []

        # Causal model
        self.causal_links: List[CausalLink] = []

        # Configuration
        self.working_memory_ttl = config.get("working_memory_ttl_minutes", 60)
        self.consolidation_interval_hours = config.get("consolidation_interval_hours", 4)
        self.forgetting_decay_rate = config.get("forgetting_decay_rate", 0.1)
        self.min_pattern_frequency = config.get("min_pattern_frequency", 3)
        self.salience_threshold = config.get("salience_threshold", 0.7)

        # Embedding
        self.embedding_dim = config.get("embedding_dim", 384)
        self._embedding_model = None

        # Tracking
        self.last_consolidation = datetime.now()
        self.consolidation_history: List[Dict[str, Any]] = []

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

    def _calculate_salience(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate salience score based on content and context."""
        salience = 0.5  # Base

        # Boost for explicit importance markers
        if metadata.get("important", False):
            salience += 0.3
        if metadata.get("error", False):
            salience += 0.2
        if metadata.get("success", False):
            salience += 0.1

        # Boost for certain keywords
        high_salience_keywords = ["error", "bug", "fix", "important", "critical", "solution", "learned"]
        content_lower = content.lower()
        for keyword in high_salience_keywords:
            if keyword in content_lower:
                salience += 0.05

        return min(1.0, salience)

    def store(self, content: str, metadata: Dict[str, Any] = None) -> Memory:
        """Store memory in appropriate tier."""
        metadata = metadata or {}
        memory_id = self._generate_memory_id(content)
        embedding = self._get_embedding(content)
        salience = self._calculate_salience(content, metadata)

        # Determine initial tier based on content type
        memory_type = metadata.get("tier", "working")
        if memory_type not in ["working", "episodic"]:
            memory_type = "working"

        memory = TieredMemory(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            memory_type=memory_type,
            metadata=metadata,
            significance=salience,
            access_count=0,
            last_accessed=None,
            embedding=embedding,
            valence=metadata.get("valence", 0.0),
            arousal=metadata.get("arousal", 0.0),
            salience=salience,
            strength=1.0,
            decay_rate=self.forgetting_decay_rate
        )

        if memory_type == "working":
            self.working_memory.append(memory)
        else:
            self.episodic_memory.append(memory)

        return memory

    def _apply_forgetting(self) -> int:
        """
        Apply Ebbinghaus forgetting curve to memories.
        strength = e^(-kt) where k is decay rate, t is time since last access.
        Returns number of memories decayed below threshold.
        """
        forgotten = 0
        current_time = datetime.now()

        for memory in self.episodic_memory:
            if memory.last_accessed:
                time_since_access = (current_time - memory.last_accessed).total_seconds() / 3600  # hours
            else:
                time_since_access = (current_time - memory.timestamp).total_seconds() / 3600

            # Ebbinghaus decay
            memory.strength = math.exp(-memory.decay_rate * time_since_access)

            # Salience modulates decay (high salience = slower decay)
            memory.strength *= (0.5 + 0.5 * memory.salience)

            if memory.strength < 0.1:
                forgotten += 1

        return forgotten

    def _boost_retrieval(self, memory: TieredMemory) -> None:
        """Boost memory strength on retrieval (spacing effect)."""
        memory.access_count += 1
        memory.last_accessed = datetime.now()
        # Boost strength, but not above 1.0
        memory.strength = min(1.0, memory.strength + 0.2)
        # Reduce decay rate for frequently accessed memories
        memory.decay_rate *= 0.95

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Retrieve from all tiers with salience-weighted scoring.
        Priority: Semantic > Procedural > Episodic > Working
        """
        start_time = time.time()
        query_embedding = self._get_embedding(query)

        # Collect candidates from all tiers
        candidates: List[Tuple[Memory, float, str]] = []

        # Semantic memory (highest priority)
        for concept in self.semantic_memory:
            concept_text = f"{concept.concept}: {concept.definition}"
            concept_embedding = self._get_embedding(concept_text)
            sim = np.dot(query_embedding, concept_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(concept_embedding) + 1e-8
            )
            # Boost semantic memories
            score = sim * 1.3 * concept.confidence
            # Create a pseudo-memory for uniform interface
            pseudo_mem = Memory(
                id=concept.id,
                content=concept_text,
                timestamp=concept.created_at,
                memory_type="semantic",
                metadata={"concept": concept.concept, "confidence": concept.confidence}
            )
            candidates.append((pseudo_mem, score, "semantic"))

        # Procedural memory
        for skill in self.procedural_memory:
            skill_text = f"Skill: {skill.name} ({skill.category})\nSteps: {'; '.join(skill.steps)}"
            skill_embedding = self._get_embedding(skill_text)
            sim = np.dot(query_embedding, skill_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(skill_embedding) + 1e-8
            )
            # Weight by success rate
            score = sim * 1.2 * skill.success_rate
            pseudo_mem = Memory(
                id=skill.id,
                content=skill_text,
                timestamp=datetime.now(),
                memory_type="procedural",
                metadata={"skill": skill.name, "success_rate": skill.success_rate}
            )
            candidates.append((pseudo_mem, score, "procedural"))

        # Episodic memory
        for memory in self.episodic_memory:
            if memory.embedding is not None:
                sim = np.dot(query_embedding, memory.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-8
                )
                # Weight by salience and strength (forgetting)
                score = sim * memory.salience * memory.strength
                candidates.append((memory, score, "episodic"))

        # Working memory (recent, high access priority)
        for memory in self.working_memory:
            if memory.embedding is not None:
                sim = np.dot(query_embedding, memory.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-8
                )
                # Recency boost for working memory
                age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
                recency_boost = max(0.5, 1.0 - age_hours / 24)
                score = sim * recency_boost
                candidates.append((memory, score, "working"))

        # Sort by score and take top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = candidates[:k]

        # Boost retrieved memories
        retrieved_memories = []
        for memory, score, tier in top_k:
            memory.metadata["retrieval_score"] = float(score)
            memory.metadata["retrieval_tier"] = tier
            if isinstance(memory, TieredMemory):
                self._boost_retrieval(memory)
            retrieved_memories.append(memory)

        retrieval_time = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=retrieved_memories,
            query=query,
            retrieval_time_ms=retrieval_time,
            metadata={
                "working_count": len(self.working_memory),
                "episodic_count": len(self.episodic_memory),
                "semantic_count": len(self.semantic_memory),
                "procedural_count": len(self.procedural_memory),
                "tier_distribution": {tier: sum(1 for _, _, t in top_k if t == tier)
                                       for tier in ["working", "episodic", "semantic", "procedural"]}
            }
        )

    def _promote_working_to_episodic(self) -> int:
        """Promote high-access working memories to episodic."""
        promoted = 0
        cutoff = datetime.now() - timedelta(minutes=self.working_memory_ttl)

        to_remove = []
        for memory in self.working_memory:
            # Promote if: accessed multiple times OR explicitly important
            should_promote = (
                memory.access_count >= 3 or
                memory.salience >= self.salience_threshold or
                memory.metadata.get("important", False)
            )

            # Or if TTL expired but high value
            ttl_expired = memory.timestamp < cutoff
            if ttl_expired and memory.access_count >= 2:
                should_promote = True

            if should_promote:
                memory.memory_type = "episodic"
                self.episodic_memory.append(memory)
                to_remove.append(memory)
                promoted += 1
            elif ttl_expired:
                # TTL expired and not important - just remove
                to_remove.append(memory)

        for memory in to_remove:
            self.working_memory.remove(memory)

        return promoted

    def _extract_patterns(self) -> int:
        """Extract patterns from episodic memories to create semantic concepts."""
        extracted = 0

        # Group episodic memories by similarity
        if len(self.episodic_memory) < self.min_pattern_frequency:
            return 0

        # Simple pattern extraction: find repeated content themes
        # In production, this would use LLM-based extraction
        content_clusters: Dict[str, List[TieredMemory]] = {}

        for memory in self.episodic_memory:
            # Extract key phrases (simple approach)
            words = memory.content.lower().split()
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if phrase not in content_clusters:
                    content_clusters[phrase] = []
                content_clusters[phrase].append(memory)

        # Create concepts from frequent patterns
        for phrase, memories in content_clusters.items():
            if len(memories) >= self.min_pattern_frequency:
                # Check if concept already exists
                existing = [c for c in self.semantic_memory if phrase in c.concept.lower()]
                if not existing:
                    concept = SemanticConcept(
                        id=f"concept_{len(self.semantic_memory)}",
                        concept=phrase.title(),
                        definition=f"Pattern observed in {len(memories)} episodes",
                        confidence=min(1.0, len(memories) / 10),
                        source_episodes=[m.id for m in memories]
                    )
                    self.semantic_memory.append(concept)
                    extracted += 1

                    # Mark source memories as consolidated
                    for memory in memories:
                        memory.consolidated = True
                        memory.consolidated_from.append(concept.id)

        return extracted

    def _extract_skills(self) -> int:
        """Extract procedural skills from repeated successful actions."""
        extracted = 0

        # Group successful actions by type
        action_groups: Dict[str, List[ActionOutcome]] = {}
        for action in self.action_history:
            if action.success_score >= 0.7:
                if action.action_type not in action_groups:
                    action_groups[action.action_type] = []
                action_groups[action.action_type].append(action)

        # Create skills from frequent successful actions
        for action_type, actions in action_groups.items():
            if len(actions) >= self.min_pattern_frequency:
                # Check if skill already exists
                existing = [s for s in self.procedural_memory if s.name == action_type]
                if not existing:
                    avg_success = sum(a.success_score for a in actions) / len(actions)
                    skill = ProceduralSkill(
                        id=f"skill_{len(self.procedural_memory)}",
                        name=action_type,
                        category="learned",
                        steps=[actions[0].action_description],  # Simplified
                        success_rate=avg_success,
                        execution_count=len(actions)
                    )
                    self.procedural_memory.append(skill)
                    extracted += 1

        return extracted

    def _discover_causal_links(self) -> int:
        """Discover causal relationships from action sequences."""
        discovered = 0

        # Look for action sequences where one consistently precedes another
        if len(self.action_history) < 2:
            return 0

        # Simple: consecutive successful actions might be causally linked
        for i in range(len(self.action_history) - 1):
            action_a = self.action_history[i]
            action_b = self.action_history[i + 1]

            # If both successful and close in time
            time_gap = (action_b.timestamp - action_a.timestamp).total_seconds()
            if time_gap < 300 and action_a.success_score >= 0.7 and action_b.success_score >= 0.7:
                # Check if link already exists
                existing = [l for l in self.causal_links
                           if l.cause_id == action_a.action_type and l.effect_id == action_b.action_type]
                if existing:
                    existing[0].observations += 1
                    existing[0].strength = min(1.0, existing[0].strength + 0.1)
                else:
                    link = CausalLink(
                        cause_id=action_a.action_type,
                        effect_id=action_b.action_type,
                        relationship_type="direct",
                        strength=0.5
                    )
                    self.causal_links.append(link)
                    discovered += 1

        return discovered

    def consolidate(self) -> Dict[str, Any]:
        """
        Run full consolidation cycle (sleep-like).

        1. Promote working → episodic
        2. Extract patterns → semantic
        3. Extract skills → procedural
        4. Discover causal links
        5. Apply forgetting curve
        6. Compress old memories
        """
        start_time = time.time()
        results = {
            "consolidated": True,
            "timestamp": datetime.now().isoformat(),
            "before": {
                "working": len(self.working_memory),
                "episodic": len(self.episodic_memory),
                "semantic": len(self.semantic_memory),
                "procedural": len(self.procedural_memory)
            }
        }

        # 1. Promote working to episodic
        promoted = self._promote_working_to_episodic()
        results["promoted_to_episodic"] = promoted

        # 2. Extract patterns to semantic
        patterns = self._extract_patterns()
        results["patterns_extracted"] = patterns

        # 3. Extract skills to procedural
        skills = self._extract_skills()
        results["skills_extracted"] = skills

        # 4. Discover causal links
        causal = self._discover_causal_links()
        results["causal_links_discovered"] = causal

        # 5. Apply forgetting
        forgotten = self._apply_forgetting()
        results["memories_decayed"] = forgotten

        # 6. Remove very weak memories (compression)
        before_compression = len(self.episodic_memory)
        self.episodic_memory = [m for m in self.episodic_memory if m.strength >= 0.1]
        compressed = before_compression - len(self.episodic_memory)
        results["memories_compressed"] = compressed

        results["after"] = {
            "working": len(self.working_memory),
            "episodic": len(self.episodic_memory),
            "semantic": len(self.semantic_memory),
            "procedural": len(self.procedural_memory)
        }
        results["consolidation_time_ms"] = (time.time() - start_time) * 1000

        self.last_consolidation = datetime.now()
        self.consolidation_history.append(results)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        total_episodic_chars = sum(len(m.content) for m in self.episodic_memory)
        total_working_chars = sum(len(m.content) for m in self.working_memory)

        avg_episodic_strength = (
            sum(m.strength for m in self.episodic_memory) / len(self.episodic_memory)
            if self.episodic_memory else 0
        )
        avg_episodic_salience = (
            sum(m.salience for m in self.episodic_memory) / len(self.episodic_memory)
            if self.episodic_memory else 0
        )

        return {
            "agent_type": "consolidation",
            "tiers": {
                "working": len(self.working_memory),
                "episodic": len(self.episodic_memory),
                "semantic": len(self.semantic_memory),
                "procedural": len(self.procedural_memory)
            },
            "total_memories": (
                len(self.working_memory) + len(self.episodic_memory) +
                len(self.semantic_memory) + len(self.procedural_memory)
            ),
            "causal_links": len(self.causal_links),
            "total_chars": total_episodic_chars + total_working_chars,
            "avg_episodic_strength": avg_episodic_strength,
            "avg_episodic_salience": avg_episodic_salience,
            "consolidation_enabled": True,
            "forgetting_enabled": True,
            "consolidations_run": len(self.consolidation_history),
            "last_consolidation": self.last_consolidation.isoformat(),
            "config": {
                "working_memory_ttl": self.working_memory_ttl,
                "consolidation_interval": self.consolidation_interval_hours,
                "forgetting_decay_rate": self.forgetting_decay_rate
            }
        }

    def clear(self) -> None:
        """Clear all memories."""
        self.working_memory = []
        self.episodic_memory = []
        self.semantic_memory = []
        self.procedural_memory = []
        self.causal_links = []
        self.action_history = []
        self.consolidation_history = []


# Register with factory
AgentFactory.register("consolidation", ConsolidationAgent)
