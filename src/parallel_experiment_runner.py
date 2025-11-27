"""
Parallel Experiment Runner - Distributed Across All Cluster Nodes

Strategy:
- macpro51: Orchestration only (this script)
- completeu-server: Embeddings via Ollama API
- mac-studio, macbook-air-m3: Can run lightweight experiments
- Kaggle/Colab: Heavy GPU work (manual upload)

All embedding work goes to completeu-server, NO local CPU inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
import requests
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import random

# ============================================================================
# REMOTE EMBEDDING CLIENT (uses completeu-server)
# ============================================================================

class RemoteEmbedding:
    """
    Embedding client that ONLY uses remote completeu-server.
    NEVER runs inference on local CPU.
    """

    OLLAMA_URL = "http://192.168.1.186:11434"
    MODEL = "nomic-embed-text"
    CACHE_DIR = Path("/home/marc/research-papers/memory-consolidation/experiments/.embedding_cache")

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._test_connection()

    def _test_connection(self):
        """Verify completeu-server is available."""
        try:
            resp = requests.get(f"{self.OLLAMA_URL}/api/version", timeout=5)
            if resp.status_code == 200:
                print(f"✓ Connected to completeu-server Ollama: {resp.json()}")
            else:
                raise ConnectionError(f"Ollama returned {resp.status_code}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to completeu-server: {e}")

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.MODEL}:{text}".encode()).hexdigest()[:32]

    def _get_cached(self, text: str) -> Optional[List[float]]:
        cache_file = self.CACHE_DIR / f"{self._cache_key(text)}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def _set_cache(self, text: str, embedding: List[float]):
        cache_file = self.CACHE_DIR / f"{self._cache_key(text)}.json"
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)

    def encode(self, text: str) -> np.ndarray:
        """Get embedding from completeu-server."""
        # Check cache first
        cached = self._get_cached(text)
        if cached is not None:
            return np.array(cached)

        # Call remote Ollama
        try:
            resp = requests.post(
                f"{self.OLLAMA_URL}/api/embeddings",
                json={"model": self.MODEL, "prompt": text},
                timeout=30
            )
            if resp.status_code == 200:
                embedding = resp.json().get("embedding", [])
                self._set_cache(text, embedding)
                return np.array(embedding)
            else:
                raise RuntimeError(f"Ollama returned {resp.status_code}")
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")

    def encode_batch(self, texts: List[str], max_workers: int = 4) -> List[np.ndarray]:
        """Encode multiple texts in parallel."""
        results = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.encode, text): i for i, text in enumerate(texts)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results


# Global embedding client
EMBEDDINGS = None

def get_embeddings() -> RemoteEmbedding:
    global EMBEDDINGS
    if EMBEDDINGS is None:
        EMBEDDINGS = RemoteEmbedding()
    return EMBEDDINGS


# ============================================================================
# LIGHTWEIGHT AGENT IMPLEMENTATIONS (CPU-safe, uses remote embeddings)
# ============================================================================

@dataclass
class Memory:
    id: str
    content: str
    timestamp: datetime
    memory_type: str
    embedding: Optional[np.ndarray] = None
    significance: float = 0.5
    access_count: int = 0
    strength: float = 1.0


class BaseAgent:
    """Base agent using REMOTE embeddings only."""

    def __init__(self, agent_id: str, config: Dict = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.memories: List[Memory] = []
        self.embeddings = get_embeddings()

    def store(self, content: str, **kwargs) -> Memory:
        mem_id = f"{self.agent_id}_{len(self.memories):04d}"
        embedding = self.embeddings.encode(content)

        memory = Memory(
            id=mem_id,
            content=content,
            timestamp=datetime.now(),
            memory_type=kwargs.get("memory_type", "flat"),
            embedding=embedding,
            significance=kwargs.get("significance", 0.5)
        )
        self.memories.append(memory)
        return memory

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Memory, float]]:
        if not self.memories:
            return []

        query_emb = self.embeddings.encode(query)
        scored = []

        for mem in self.memories:
            if mem.embedding is not None:
                sim = np.dot(query_emb, mem.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding) + 1e-8
                )
                scored.append((mem, float(sim)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def consolidate(self) -> Dict:
        return {"consolidated": False, "agent": self.agent_id}

    def get_stats(self) -> Dict:
        return {
            "agent_type": self.__class__.__name__,
            "total_memories": len(self.memories)
        }


class FlatMemoryAgent(BaseAgent):
    """Baseline: Simple flat memory, no consolidation."""
    pass


class ConsolidationAgent(BaseAgent):
    """Experimental: Multi-tier with consolidation."""

    def __init__(self, agent_id: str, config: Dict = None):
        super().__init__(agent_id, config)
        self.semantic_memory: List[Dict] = []
        self.procedural_memory: List[Dict] = []
        self.consolidation_count = 0

    def consolidate(self) -> Dict:
        """Extract patterns from episodic to semantic memory."""
        self.consolidation_count += 1

        # Count word frequencies for pattern extraction
        word_counts = {}
        for mem in self.memories:
            for word in mem.content.lower().split():
                if len(word) > 4:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Extract frequent patterns as semantic concepts
        new_concepts = 0
        for word, count in word_counts.items():
            if count >= 3:
                existing = [c for c in self.semantic_memory if c["concept"] == word]
                if not existing:
                    self.semantic_memory.append({
                        "concept": word,
                        "frequency": count,
                        "confidence": min(1.0, count / 10)
                    })
                    new_concepts += 1

        # Apply forgetting (reduce strength of old memories)
        for mem in self.memories:
            mem.strength *= 0.95

        return {
            "consolidated": True,
            "consolidation_number": self.consolidation_count,
            "new_concepts": new_concepts,
            "total_semantic": len(self.semantic_memory)
        }

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Memory, float]]:
        """Retrieve with semantic concept boosting."""
        base_results = super().retrieve(query, k * 2)

        # Boost memories related to semantic concepts
        query_words = set(query.lower().split())
        semantic_concepts = {c["concept"] for c in self.semantic_memory}
        concept_overlap = query_words & semantic_concepts

        boosted = []
        for mem, score in base_results:
            mem_words = set(mem.content.lower().split())
            if mem_words & concept_overlap:
                score *= 1.2  # Semantic boost
            score *= mem.strength  # Apply forgetting
            boosted.append((mem, score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[:k]


# ============================================================================
# PARALLEL EXPERIMENT RUNNER
# ============================================================================

@dataclass
class TaskResult:
    task_id: str
    session_id: int
    success: bool
    success_score: float
    retrieval_count: int
    retrieval_time_ms: float


@dataclass
class ExperimentResult:
    agent_type: str
    overall_success_rate: float
    avg_retrieval_time_ms: float
    total_memories: int
    session_results: List[Dict]
    consolidations: List[Dict]


def generate_task(session_id: int, task_id: int) -> Dict:
    """Generate a coding task."""
    components = ["UserService", "OrderProcessor", "PaymentHandler", "AuthManager",
                  "DataLoader", "CacheManager", "MessageQueue", "TaskScheduler"]
    task_types = ["bug_fix", "feature_add", "refactor", "debug"]

    return {
        "id": f"task_{session_id}_{task_id}",
        "session_id": session_id,
        "type": random.choice(task_types),
        "component": random.choice(components),
        "difficulty": random.choice([1, 2, 3])
    }


def run_single_experiment(agent_class, agent_id: str, num_sessions: int = 10,
                          tasks_per_session: int = 5, consolidate_every: int = 2,
                          seed: int = 42) -> ExperimentResult:
    """Run experiment with a single agent."""
    random.seed(seed)
    agent = agent_class(agent_id)

    all_results = []
    session_results = []
    consolidations = []
    total_retrieval_time = 0

    for session_id in range(num_sessions):
        session_successes = 0

        for task_id in range(tasks_per_session):
            task = generate_task(session_id, task_id)

            # Store task experience
            content = f"Task: {task['type']} on {task['component']}"
            agent.store(content, significance=task["difficulty"] / 3)

            # Retrieve context
            start = time.time()
            retrieved = agent.retrieve(f"{task['type']} {task['component']}", k=5)
            retrieval_time = (time.time() - start) * 1000
            total_retrieval_time += retrieval_time

            # Simulate success
            base_prob = {1: 0.9, 2: 0.7, 3: 0.5}[task["difficulty"]]
            context_boost = 0.05 * len(retrieved)
            success_prob = min(1.0, base_prob + context_boost)
            success = random.random() < success_prob

            if success:
                session_successes += 1

            all_results.append(TaskResult(
                task_id=task["id"],
                session_id=session_id,
                success=success,
                success_score=success_prob,
                retrieval_count=len(retrieved),
                retrieval_time_ms=retrieval_time
            ))

        session_results.append({
            "session_id": session_id,
            "success_rate": session_successes / tasks_per_session
        })

        # Consolidate periodically
        if session_id % consolidate_every == 0 and session_id > 0:
            cons = agent.consolidate()
            consolidations.append(cons)

    total_tasks = len(all_results)
    overall_success = sum(1 for r in all_results if r.success) / total_tasks

    return ExperimentResult(
        agent_type=agent.__class__.__name__,
        overall_success_rate=overall_success,
        avg_retrieval_time_ms=total_retrieval_time / total_tasks,
        total_memories=len(agent.memories),
        session_results=session_results,
        consolidations=consolidations
    )


def run_parallel_experiments(num_seeds: int = 3) -> Dict[str, List[ExperimentResult]]:
    """
    Run experiments in PARALLEL across different configurations.

    Each agent type + seed combination runs in parallel.
    """
    print("="*60)
    print("PARALLEL EXPERIMENT RUNNER")
    print("="*60)
    print(f"Embeddings: completeu-server ({RemoteEmbedding.OLLAMA_URL})")
    print(f"Orchestration: macpro51 (local)")
    print(f"Seeds: {num_seeds}")
    print("="*60)

    results = {
        "flat_memory": [],
        "consolidation": []
    }

    # Prepare all experiment configurations
    experiments = []
    for seed in range(num_seeds):
        experiments.append(("flat", FlatMemoryAgent, f"flat_seed{seed}", seed, 100))  # 100 = no consolidation
        experiments.append(("cons", ConsolidationAgent, f"cons_seed{seed}", seed, 2))

    print(f"\nRunning {len(experiments)} experiment configurations in parallel...")

    # Run in parallel (max 4 to not overload completeu-server)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for exp_type, agent_class, agent_id, seed, cons_interval in experiments:
            future = executor.submit(
                run_single_experiment,
                agent_class, agent_id,
                num_sessions=10, tasks_per_session=5,
                consolidate_every=cons_interval, seed=seed
            )
            futures[future] = (exp_type, agent_id)

        for future in as_completed(futures):
            exp_type, agent_id = futures[future]
            result = future.result()
            key = "flat_memory" if exp_type == "flat" else "consolidation"
            results[key].append(result)
            print(f"  ✓ {agent_id}: {result.overall_success_rate:.1%} success")

    return results


def analyze_results(results: Dict[str, List[ExperimentResult]]) -> Dict:
    """Analyze and compare results."""
    analysis = {}

    for agent_type, exp_results in results.items():
        success_rates = [r.overall_success_rate for r in exp_results]
        retrieval_times = [r.avg_retrieval_time_ms for r in exp_results]

        analysis[agent_type] = {
            "mean_success_rate": np.mean(success_rates),
            "std_success_rate": np.std(success_rates),
            "mean_retrieval_time_ms": np.mean(retrieval_times),
            "num_runs": len(exp_results)
        }

    # Comparison
    flat = analysis["flat_memory"]
    cons = analysis["consolidation"]
    analysis["comparison"] = {
        "success_improvement": cons["mean_success_rate"] - flat["mean_success_rate"],
        "winner": "consolidation" if cons["mean_success_rate"] > flat["mean_success_rate"] else "flat_memory"
    }

    return analysis


def main():
    """Main entry point."""
    start_time = time.time()

    # Run parallel experiments
    results = run_parallel_experiments(num_seeds=3)

    # Analyze
    analysis = analyze_results(results)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    for agent_type, stats in analysis.items():
        if agent_type != "comparison":
            print(f"\n{agent_type}:")
            print(f"  Success Rate: {stats['mean_success_rate']:.1%} ± {stats['std_success_rate']:.1%}")
            print(f"  Retrieval Time: {stats['mean_retrieval_time_ms']:.1f} ms")

    print(f"\n{'='*60}")
    print(f"WINNER: {analysis['comparison']['winner'].upper()}")
    print(f"Improvement: {analysis['comparison']['success_improvement']*100:.1f}%")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("="*60)

    # Save results
    output_path = Path("/home/marc/research-papers/memory-consolidation/experiments/results/parallel_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_seeds": 3,
                "num_sessions": 10,
                "tasks_per_session": 5,
                "embedding_server": RemoteEmbedding.OLLAMA_URL
            }
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return analysis


if __name__ == "__main__":
    main()
