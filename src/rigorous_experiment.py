"""
Rigorous Reproducible Experiments with Statistical Proofs

This module implements statistically rigorous experiments that:
1. Use fixed seeds for full reproducibility
2. Run sufficient trials for statistical power (n≥10)
3. Compute confidence intervals (95% CI)
4. Perform hypothesis testing (Welch's t-test)
5. Calculate effect sizes (Cohen's d)
6. Save all parameters for exact reproduction

All compute offloaded to completeu-server. NEVER use local CPU for inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import json
import time
import requests
import numpy as np
from scipy import stats
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import random


# ============================================================================
# CONFIGURATION - All parameters for reproducibility
# ============================================================================

@dataclass
class ExperimentConfig:
    """Complete configuration for reproducible experiments."""
    # Experiment identity
    experiment_name: str = "consolidation_vs_flat"
    experiment_version: str = "1.0.0"

    # Reproducibility
    base_seed: int = 42
    num_seeds: int = 10  # Number of independent runs

    # Task parameters
    num_sessions: int = 30
    tasks_per_session: int = 15

    # Agent parameters
    consolidation_interval: int = 3  # Consolidate every N sessions
    forgetting_threshold: float = 0.4  # Minimum strength to keep memory (was 0.3 - too lenient)
    decay_rate_base: float = 0.6  # Base decay rate (was 0.8 - too lenient per Gemini recommendation)
    semantic_boost: float = 1.3  # Retrieval boost for semantic memories
    procedural_boost: float = 1.4  # Retrieval boost for procedural memories

    # Task generation
    recurring_pattern_prob: float = 0.3  # Probability of recurring task

    # Infrastructure
    embedding_server: str = "http://192.168.1.186:11434"
    embedding_model: str = "qwen3-embedding:8b-fp16"  # SOTA embedding model (MTEB leader)
    llm_model: str = "mistral:7b-instruct-v0.3-fp16"  # Fast model for semantic summaries
    use_llm_summaries: bool = True  # Enable LLM-generated semantic summaries (Option B)
    use_llm_evaluation: bool = True  # Enable LLM-based task evaluation (Option D)
    max_workers: int = 4

    # LLM Temperature settings (0.0=deterministic, 0.3=consistent, 0.7=creative)
    summary_temperature: float = 0.3  # For semantic concept summaries
    evaluation_temperature: float = 0.1  # For task evaluation (low = deterministic)

    # Ollama Cloud settings (faster inference via ollama.com API)
    use_ollama_cloud: bool = False  # Use Ollama Cloud instead of local server
    ollama_cloud_url: str = "https://ollama.com"  # Ollama Cloud API URL
    ollama_cloud_api_key: str = ""  # Set via OLLAMA_API_KEY env var if empty
    ollama_cloud_model: str = "minimax-m2"  # Cloud model to use (fast, not a thinking model)

    # Custom classifier evaluation (Option E) - deterministic, trained locally
    use_classifier_evaluation: bool = False  # Use trained classifier instead of LLM
    classifier_training_mode: bool = False  # Collect training data using LLM as oracle
    classifier_training_samples: int = 500  # Samples to collect for training
    classifier_model_path: str = "experiments/classifier_model.pkl"

    # Statistical parameters
    confidence_level: float = 0.95

    # Enhanced consolidation settings (v2.0 - True Sleep-Inspired)
    enable_memory_replay: bool = True  # Re-evaluate key memories during consolidation
    enable_concept_updates: bool = True  # Allow refining existing concepts
    enable_failure_learning: bool = True  # Learn from failures (anti-procedures)
    enable_causal_tracking: bool = True  # Track retrieval→success chains
    enable_temporal_decay: bool = True  # Recency-weighted forgetting
    replay_top_k: int = 10  # Number of memories to replay per consolidation
    failure_threshold: float = 0.3  # Create anti-procedure if success_rate < this
    access_boost_multiplier: float = 1.2  # Multiplicative boost for accessed memories
    recency_half_life_sessions: int = 10  # Sessions until recency factor halves

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_hash(self) -> str:
        """Generate unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ============================================================================
# REMOTE EMBEDDING (completeu-server only)
# ============================================================================

class RemoteEmbedding:
    """Embedding client using ONLY remote completeu-server."""

    CACHE_DIR = Path("/home/marc/research-papers/memory-consolidation/experiments/.embedding_cache")

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.url = config.embedding_server
        self.model = config.embedding_model
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._verify_connection()

    def _verify_connection(self):
        try:
            resp = requests.get(f"{self.url}/api/version", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Server returned {resp.status_code}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to {self.url}: {e}")

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()[:32]

    def encode(self, text: str) -> np.ndarray:
        # Check cache
        cache_file = self.CACHE_DIR / f"{self._cache_key(text)}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return np.array(json.load(f))

        # Remote call
        resp = requests.post(
            f"{self.url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30
        )
        if resp.status_code == 200:
            embedding = resp.json().get("embedding", [])
            with open(cache_file, 'w') as f:
                json.dump(embedding, f)
            return np.array(embedding)
        raise RuntimeError(f"Embedding failed: {resp.status_code}")


class RemoteLLM:
    """LLM client using remote Ollama server or Ollama Cloud for semantic summarization."""

    CACHE_DIR = Path("/home/marc/research-papers/memory-consolidation/experiments/.llm_cache")

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.use_cloud = getattr(config, 'use_ollama_cloud', False)

        if self.use_cloud:
            # Ollama Cloud configuration
            self.url = getattr(config, 'ollama_cloud_url', 'https://ollama.com')
            self.model = getattr(config, 'ollama_cloud_model', 'llama3.2:3b')
            # API key from config or environment
            self.api_key = getattr(config, 'ollama_cloud_api_key', '') or os.environ.get('OLLAMA_API_KEY', '')
            if not self.api_key:
                raise ValueError("Ollama Cloud requires API key. Set OLLAMA_API_KEY env var or config.ollama_cloud_api_key")
        else:
            # Local server configuration
            self.url = config.embedding_server  # Same server has LLM
            self.model = getattr(config, 'llm_model', 'llama3.2:3b')  # Fast model for summaries
            self.api_key = None

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}:{prompt}".encode()).hexdigest()[:32]

    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
        """Generate text using remote LLM or Ollama Cloud.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            temperature: LLM temperature (0.0=deterministic, 0.3=consistent, 0.7=creative)
        """
        # Check cache (include temperature in cache key for different results)
        cache_key = hashlib.sha256(f"{self.model}:{temperature}:{prompt}".encode()).hexdigest()[:32]
        cache_file = self.CACHE_DIR / f"{cache_key}.txt"
        if cache_file.exists():
            with open(cache_file) as f:
                return f.read()

        try:
            if self.use_cloud:
                # Ollama Cloud API (uses /api/chat endpoint)
                headers = {"Authorization": f"Bearer {self.api_key}"}
                resp = requests.post(
                    f"{self.url}/api/chat",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        }
                    },
                    timeout=30  # Cloud is faster
                )
                if resp.status_code == 200:
                    result = resp.json().get("message", {}).get("content", "").strip()
                    with open(cache_file, 'w') as f:
                        f.write(result)
                    return result
            else:
                # Local Ollama server API (uses /api/generate endpoint)
                resp = requests.post(
                    f"{self.url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        }
                    },
                    timeout=60
                )
                if resp.status_code == 200:
                    result = resp.json().get("response", "").strip()
                    with open(cache_file, 'w') as f:
                        f.write(result)
                    return result
        except Exception as e:
            # Fallback to template if LLM fails
            return None
        return None

    def summarize_pattern(self, pattern: str, memories: List['Memory'],
                          success_rate: float = None) -> str:
        """Generate a rich semantic summary from related memories."""
        # Extract unique outcomes and contexts
        successes = [m for m in memories if 'success' in m.content.lower()]
        failures = [m for m in memories if 'failed' in m.content.lower()]

        # Build context for LLM
        sample_memories = memories[:5]  # Sample to keep prompt short
        memory_texts = [m.content for m in sample_memories]

        prompt = f"""Summarize this recurring pattern for an AI agent's memory:

Pattern: {pattern}
Total occurrences: {len(memories)}
Successes: {len(successes)} | Failures: {len(failures)}
Success rate: {success_rate:.0%} if success_rate else 'unknown'

Sample experiences:
{chr(10).join(f'- {t}' for t in memory_texts)}

Write a ONE PARAGRAPH summary (2-3 sentences) that captures:
1. What this pattern involves
2. When it tends to succeed vs fail
3. Key insight for handling similar tasks

Summary:"""

        # Use summary temperature for creative/flexible summarization
        result = self.generate(prompt, temperature=self.config.summary_temperature)
        if result:
            return f"[Learned] {pattern}: {result}"
        else:
            # Fallback template if LLM unavailable
            return f"Concept: {pattern} - {len(memories)} occurrences, {success_rate:.0%} success rate"

    def replay_memory(self, memory: 'Memory', related_memories: List['Memory']) -> Dict:
        """Replay a memory during consolidation to extract deeper insights.

        This is the core of sleep-inspired consolidation - replaying experiences
        to strengthen connections and extract knowledge.
        """
        # Build context from related memories
        related_context = "\n".join([f"- {m.content}" for m in related_memories[:3]])

        prompt = f"""You are helping an AI agent consolidate its memories during a "sleep" phase.

MEMORY TO REPLAY:
{memory.content}

RELATED MEMORIES:
{related_context if related_context else "None"}

Analyze this memory and extract:
1. KEY_INSIGHT: What is the most important actionable insight? (one sentence)
2. CAUSAL: What pattern or condition led to this outcome?
3. SHOULD_STRENGTHEN: Should this memory be strengthened? (yes/no)
4. IMPORTANCE: Rate importance 1-10 for future similar tasks

Respond in exactly this format:
KEY_INSIGHT: [your insight]
CAUSAL: [pattern → outcome]
SHOULD_STRENGTHEN: [yes/no]
IMPORTANCE: [1-10]"""

        result = self.generate(prompt, max_tokens=200, temperature=0.2)

        if result:
            # Parse structured response
            lines = result.strip().split('\n')
            parsed = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    parsed[key.strip().upper()] = value.strip()

            return {
                'insight': parsed.get('KEY_INSIGHT', ''),
                'causal': parsed.get('CAUSAL', ''),
                'strengthen': parsed.get('SHOULD_STRENGTHEN', '').lower() == 'yes',
                'importance': int(parsed.get('IMPORTANCE', '5').split()[0]) if parsed.get('IMPORTANCE', '5').split()[0].isdigit() else 5
            }

        return {'insight': '', 'causal': '', 'strengthen': False, 'importance': 5}

    def extract_actionable_knowledge(self, pattern: str, memories: List['Memory'],
                                      success_rate: float) -> str:
        """Extract actionable knowledge for better retrieval (not just statistics).

        Instead of "Concept: X (observed N times)", produce:
        "When debugging MessageQueue, check connection timeout first - works 85% of time"
        """
        successes = [m for m in memories if 'success' in m.content.lower()]
        failures = [m for m in memories if 'failed' in m.content.lower()]

        prompt = f"""Extract ACTIONABLE knowledge from these experiences:

PATTERN: {pattern}
SUCCESS RATE: {success_rate:.0%}
SUCCESSES: {len(successes)} | FAILURES: {len(failures)}

SAMPLE EXPERIENCES:
{chr(10).join(f'- {m.content}' for m in memories[:5])}

Write ONE actionable sentence that would help someone facing a similar task.
Format: "When [situation], [action] - [expected result]"

Example good output:
"When debugging MessageQueue errors, check connection timeout settings first - resolves 85% of issues"

Your actionable knowledge:"""

        result = self.generate(prompt, max_tokens=100, temperature=0.3)

        if result and len(result) > 10:
            return f"[ACTION] {result.strip()}"
        else:
            # Fallback with some structure
            action = pattern.replace('_', ' ')
            return f"[ACTION] When handling {action}: success rate {success_rate:.0%} based on {len(memories)} experiences"

    def generate_anti_procedure(self, pattern: str, memories: List['Memory'],
                                 failure_rate: float) -> str:
        """Generate an anti-procedure - what NOT to do based on failures."""
        failures = [m for m in memories if 'failed' in m.content.lower()]

        prompt = f"""Generate a WARNING based on past failures:

PATTERN: {pattern}
FAILURE RATE: {failure_rate:.0%}

FAILED ATTEMPTS:
{chr(10).join(f'- {m.content}' for m in failures[:5])}

Write ONE warning sentence to prevent repeating this mistake.
Format: "AVOID: [what not to do] - [why it fails]"

Your warning:"""

        result = self.generate(prompt, max_tokens=80, temperature=0.3)

        if result and len(result) > 10:
            return f"[WARNING] {result.strip()}"
        else:
            return f"[WARNING] {pattern} has {failure_rate:.0%} failure rate - approach with caution"

    def refine_concept(self, existing_concept: 'Memory', new_memories: List['Memory']) -> str:
        """Refine an existing concept with new information (not one-and-done)."""
        prompt = f"""Update this existing knowledge with new information:

EXISTING KNOWLEDGE:
{existing_concept.content}

NEW EXPERIENCES:
{chr(10).join(f'- {m.content}' for m in new_memories[:5])}

Write an UPDATED version that incorporates the new information.
Keep the same format but add/modify insights based on new data.

Updated knowledge:"""

        result = self.generate(prompt, max_tokens=150, temperature=0.3)

        if result and len(result) > 10:
            return result.strip()
        else:
            return existing_concept.content  # Keep original if refinement fails

    def evaluate_task(self, task: Dict, memories: List['Memory']) -> float:
        """Use LLM to evaluate if task can be solved with given memories.

        Returns a confidence score 0.0-1.0 based on LLM's assessment.
        This tests the ACTUAL UTILITY of memory organization for task solving.
        """
        # Build task description
        task_desc = f"Task: {task['type']} involving {task['component']}"
        if task.get('error_type'):
            task_desc += f" (error: {task['error_type']})"
        task_desc += f"\nDifficulty: {task['difficulty']}/3"

        # Build memory context (limit to top 5 for prompt size)
        memory_context = []
        for mem in memories[:5]:
            memory_context.append(f"- [{mem.memory_type}] {mem.content}")

        context_str = "\n".join(memory_context) if memory_context else "No relevant memories found."

        prompt = f"""You are evaluating whether an AI agent can solve a task given its retrieved memories.

TASK:
{task_desc}

RETRIEVED MEMORIES:
{context_str}

Based on the memories, rate how likely the agent is to solve this task correctly.
Consider:
1. Do the memories contain relevant information?
2. Are there patterns or procedures that apply?
3. Is there enough context to make good decisions?

Respond with ONLY a number from 0-100 representing success probability.
- 0-30: Memories are irrelevant or misleading
- 30-50: Partial relevance, might help
- 50-70: Good context, reasonable chance
- 70-90: Highly relevant, likely to succeed
- 90-100: Perfect context, almost certain success

Your rating (just the number):"""

        # Use low temperature for deterministic evaluation
        result = self.generate(prompt, max_tokens=10, temperature=self.config.evaluation_temperature)

        if result:
            # Parse number from response
            try:
                # Extract first number from response
                import re
                numbers = re.findall(r'\d+', result)
                if numbers:
                    score = int(numbers[0])
                    return min(100, max(0, score)) / 100.0
            except:
                pass

        # Fallback to base probability if LLM fails
        return {1: 0.7, 2: 0.5, 3: 0.3}.get(task.get("difficulty", 2), 0.5)


# Global LLM client
_LLM = None

def get_llm(config: ExperimentConfig) -> RemoteLLM:
    global _LLM
    if _LLM is None:
        _LLM = RemoteLLM(config)
    return _LLM


# Global embeddings client
_EMBEDDINGS = None

def get_embeddings(config: ExperimentConfig) -> RemoteEmbedding:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = RemoteEmbedding(config)
    return _EMBEDDINGS


# ============================================================================
# CLASSIFIER EVALUATOR (Option E) - Deterministic, trained locally
# ============================================================================

class ClassifierEvaluator:
    """Custom classifier for deterministic task evaluation.

    Instead of using LLM (which has temperature variance), this trains a
    logistic regression classifier on (task_features, memory_features) → success.

    Training data is collected during a training phase, then the classifier
    is used for evaluation - providing 100% deterministic results.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.training_data: List[Tuple[np.ndarray, float]] = []
        self.model = None
        self.is_trained = False
        self.embeddings = get_embeddings(config)
        # Feature dimensions: task_embed(1536) + memory_pooled(1536) + meta_features(10)
        self.feature_dim = None

    def _encode_task(self, task: Dict) -> np.ndarray:
        """Encode task as embedding vector."""
        task_text = f"Task: {task['type']} {task['component']} difficulty:{task['difficulty']}"
        if task.get('error_type'):
            task_text += f" error:{task['error_type']}"
        return self.embeddings.embed(task_text)

    def _pool_memories(self, memories: List['Memory']) -> np.ndarray:
        """Pool memory embeddings into single vector (mean pooling)."""
        if not memories:
            # Return zeros if no memories
            sample_dim = 1536  # Default embedding dimension
            return np.zeros(sample_dim)

        embeddings = []
        for mem in memories[:10]:  # Limit to top 10 memories
            if mem.embedding is not None:
                embeddings.append(mem.embedding)
            else:
                emb = self.embeddings.embed(mem.content)
                if emb is not None:
                    embeddings.append(emb)

        if not embeddings:
            return np.zeros(1536)

        return np.mean(embeddings, axis=0)

    def _extract_features(self, task: Dict, memories: List['Memory'],
                          similarity_scores: List[float] = None) -> np.ndarray:
        """Extract feature vector for classifier input."""
        # Task embedding
        task_emb = self._encode_task(task)
        if task_emb is None:
            task_emb = np.zeros(1536)

        # Pooled memory embedding
        mem_emb = self._pool_memories(memories)

        # Meta features (hand-crafted)
        meta = np.array([
            task.get('difficulty', 2) / 3.0,  # Normalized difficulty
            len(memories) / 10.0,  # Memory count (normalized)
            np.mean(similarity_scores) if similarity_scores else 0.5,  # Avg similarity
            np.max(similarity_scores) if similarity_scores else 0.5,  # Max similarity
            sum(1 for m in memories if m.memory_type == 'semantic') / max(len(memories), 1),
            sum(1 for m in memories if m.memory_type == 'procedural') / max(len(memories), 1),
            sum(1 for m in memories if m.memory_type == 'episodic') / max(len(memories), 1),
            sum(1 for m in memories if task.get('type', '') in m.content) / max(len(memories), 1),
            sum(1 for m in memories if task.get('component', '') in m.content) / max(len(memories), 1),
            sum(1 for m in memories if 'success' in m.content.lower()) / max(len(memories), 1),
        ])

        # Concatenate all features
        return np.concatenate([task_emb, mem_emb, meta])

    def collect_training_sample(self, task: Dict, memories: List['Memory'],
                                 similarity_scores: List[float], label: float):
        """Collect a training sample during training phase.

        Args:
            task: Task dictionary
            memories: Retrieved memories
            similarity_scores: Cosine similarity scores for each memory
            label: Ground truth success (0.0 to 1.0)
        """
        if self.is_trained:
            return  # Don't collect if already trained

        features = self._extract_features(task, memories, similarity_scores)
        self.training_data.append((features, label))

    def train(self):
        """Train the classifier on collected samples."""
        if len(self.training_data) < 50:
            print(f"[Classifier] Not enough training data ({len(self.training_data)} samples)")
            return False

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            import pickle

            X = np.array([f for f, _ in self.training_data])
            y = np.array([l > 0.5 for _, l in self.training_data])  # Binary classification

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Calculate training accuracy
            train_acc = self.model.score(X_scaled, y)
            print(f"[Classifier] Trained on {len(self.training_data)} samples, accuracy: {train_acc:.2%}")

            # Save model
            model_path = Path(self.config.classifier_model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
            print(f"[Classifier] Model saved to {model_path}")

            return True

        except Exception as e:
            print(f"[Classifier] Training failed: {e}")
            return False

    def load(self) -> bool:
        """Load pre-trained classifier from disk."""
        try:
            import pickle
            model_path = Path(self.config.classifier_model_path)
            if not model_path.exists():
                return False

            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
                print(f"[Classifier] Loaded model from {model_path}")
                return True
        except Exception as e:
            print(f"[Classifier] Failed to load: {e}")
            return False

    def evaluate(self, task: Dict, memories: List['Memory'],
                 similarity_scores: List[float] = None) -> float:
        """Evaluate task success probability using trained classifier.

        Returns:
            Success probability 0.0 to 1.0 (deterministic)
        """
        if not self.is_trained or self.model is None:
            # Fallback to base probability
            return {1: 0.7, 2: 0.5, 3: 0.3}.get(task.get("difficulty", 2), 0.5)

        features = self._extract_features(task, memories, similarity_scores or [0.5] * len(memories))
        X = self.scaler.transform(features.reshape(1, -1))

        # Get probability estimate
        prob = self.model.predict_proba(X)[0, 1]  # Probability of class 1 (success)
        return float(prob)


# Global classifier
_CLASSIFIER = None

def get_classifier(config: ExperimentConfig) -> ClassifierEvaluator:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = ClassifierEvaluator(config)
        _CLASSIFIER.load()  # Try to load pre-trained model
    return _CLASSIFIER


# ============================================================================
# MEMORY AND AGENT IMPLEMENTATIONS
# ============================================================================

@dataclass
class Memory:
    id: str
    content: str
    timestamp: float
    memory_type: str
    embedding: Optional[np.ndarray] = None
    significance: float = 0.5
    access_count: int = 0
    strength: float = 1.0
    session_id: int = 0
    tags: List[str] = field(default_factory=list)
    # Enhanced consolidation fields (v2.0)
    causal_links: List[str] = field(default_factory=list)  # IDs of memories that led to success
    led_to_success: bool = False  # Whether this memory led to task success
    replay_count: int = 0  # How many times replayed during consolidation
    version: int = 1  # For tracking concept updates


class FlatMemoryAgent:
    """Baseline: Simple flat memory, no consolidation."""

    def __init__(self, agent_id: str, config: ExperimentConfig):
        self.agent_id = agent_id
        self.config = config
        self.memories: List[Memory] = []
        self.embeddings = get_embeddings(config)
        self.total_stored = 0

    def store(self, content: str, session_id: int, significance: float = 0.5,
              tags: List[str] = None, success: bool = None) -> Memory:
        mem_id = f"{self.agent_id}_{self.total_stored:04d}"
        self.total_stored += 1
        embedding = self.embeddings.encode(content)

        memory = Memory(
            id=mem_id,
            content=content,
            timestamp=time.time(),
            memory_type="experience",
            embedding=embedding,
            significance=significance,
            session_id=session_id,
            tags=tags or []
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
        return {"consolidated": False, "total_memories": len(self.memories)}

    def get_memory_count(self) -> int:
        return len(self.memories)

    def get_stats(self) -> Dict:
        return {
            "total_memories": len(self.memories),
            "episodic": len(self.memories),
            "semantic": 0,
            "procedural": 0
        }


class ConsolidationAgent:
    """Experimental: Multi-tier memory with sleep-like consolidation."""

    def __init__(self, agent_id: str, config: ExperimentConfig):
        self.agent_id = agent_id
        self.config = config
        self.episodic: List[Memory] = []
        self.semantic: List[Memory] = []
        self.procedural: List[Memory] = []
        self.embeddings = get_embeddings(config)
        self.llm = get_llm(config) if config.use_llm_summaries else None
        self.total_stored = 0
        self.consolidation_count = 0

        self.pattern_counts: Dict[str, int] = {}
        self.success_patterns: Dict[str, List[bool]] = {}
        self.pattern_memories: Dict[str, List[Memory]] = {}  # Track memories per pattern for LLM summaries

    def store(self, content: str, session_id: int, significance: float = 0.5,
              tags: List[str] = None, success: bool = None) -> Memory:
        mem_id = f"{self.agent_id}_{self.total_stored:04d}"
        self.total_stored += 1
        embedding = self.embeddings.encode(content)

        memory = Memory(
            id=mem_id,
            content=content,
            timestamp=time.time(),
            memory_type="experience",
            embedding=embedding,
            significance=significance,
            session_id=session_id,
            tags=tags or []
        )
        self.episodic.append(memory)

        for tag in (tags or []):
            self.pattern_counts[tag] = self.pattern_counts.get(tag, 0) + 1
            if success is not None:
                if tag not in self.success_patterns:
                    self.success_patterns[tag] = []
                self.success_patterns[tag].append(success)
            # Track memories per pattern for LLM summarization
            if tag not in self.pattern_memories:
                self.pattern_memories[tag] = []
            self.pattern_memories[tag].append(memory)

        return memory

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Memory, float]]:
        if not self.episodic and not self.semantic and not self.procedural:
            return []

        query_emb = self.embeddings.encode(query)
        scored = []

        # Episodic (base score with decay)
        for mem in self.episodic:
            if mem.embedding is not None:
                sim = np.dot(query_emb, mem.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding) + 1e-8
                )
                scored.append((mem, float(sim) * mem.strength))

        # Semantic (boosted)
        for mem in self.semantic:
            if mem.embedding is not None:
                sim = np.dot(query_emb, mem.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding) + 1e-8
                )
                scored.append((mem, float(sim) * self.config.semantic_boost))

        # Procedural (most boosted)
        for mem in self.procedural:
            if mem.embedding is not None:
                sim = np.dot(query_emb, mem.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding) + 1e-8
                )
                scored.append((mem, float(sim) * self.config.procedural_boost))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def consolidate(self) -> Dict:
        """Enhanced sleep-inspired consolidation with all 7 fixes:
        1. Memory replay during consolidation
        2. Concept updates (not one-and-done)
        3. Failure learning (anti-procedures)
        4. Actionable knowledge extraction
        5. Causal relationship tracking
        6. Temporal weighting to decay
        7. Improved spaced repetition boost (multiplicative)
        """
        self.consolidation_count += 1
        current_session = max((m.session_id for m in self.episodic), default=0)

        stats = {
            "consolidated": True,
            "cycle": self.consolidation_count,
            "memories_before": len(self.episodic),
            "memories_forgotten": 0,
            "concepts_extracted": 0,
            "concepts_updated": 0,
            "procedures_learned": 0,
            "anti_procedures_learned": 0,
            "memories_replayed": 0,
            "memories_strengthened": 0
        }

        # =====================================================================
        # PHASE 0: MEMORY REPLAY (the core of sleep consolidation)
        # Select top memories and re-evaluate them with LLM
        # =====================================================================
        if self.config.enable_memory_replay and self.llm and self.episodic:
            # Select top memories by significance and led_to_success
            replay_candidates = sorted(
                self.episodic,
                key=lambda m: (m.led_to_success, m.significance, m.access_count),
                reverse=True
            )[:self.config.replay_top_k]

            for mem in replay_candidates:
                # Find related memories
                related = [m for m in self.episodic
                          if m.id != mem.id and any(t in m.tags for t in mem.tags)][:3]

                replay_result = self.llm.replay_memory(mem, related)
                mem.replay_count += 1
                stats["memories_replayed"] += 1

                # Strengthen if LLM recommends
                if replay_result.get('strengthen', False):
                    mem.strength = min(1.0, mem.strength + 0.2)
                    mem.significance = min(1.0, mem.significance + 0.1)
                    stats["memories_strengthened"] += 1

                # Store insight as causal link metadata
                if replay_result.get('causal') and self.config.enable_causal_tracking:
                    # Add causal insight to memory
                    if replay_result['causal'] not in mem.causal_links:
                        mem.causal_links.append(replay_result['causal'])

        # =====================================================================
        # PHASE 1: TEMPORAL FORGETTING (recency-weighted)
        # Recent memories decay slower, old unreinforced memories decay faster
        # =====================================================================
        surviving = []
        for mem in self.episodic:
            # Calculate recency factor (1.0 for recent, decays over time)
            sessions_ago = current_session - mem.session_id
            if self.config.enable_temporal_decay:
                recency_factor = 0.5 ** (sessions_ago / self.config.recency_half_life_sessions)
            else:
                recency_factor = 1.0

            # Base decay modified by significance and recency
            decay_rate = self.config.decay_rate_base + (0.25 * mem.significance * recency_factor)
            mem.strength *= decay_rate

            # IMPROVED SPACED REPETITION: Multiplicative boost instead of additive
            if mem.access_count > 0:
                boost = self.config.access_boost_multiplier ** mem.access_count
                mem.strength = min(1.0, mem.strength * boost)
                mem.access_count = 0  # Reset after applying boost

            # Boost for memories that led to success (causal tracking)
            if self.config.enable_causal_tracking and mem.led_to_success:
                mem.strength = min(1.0, mem.strength * 1.1)

            if mem.strength > self.config.forgetting_threshold:
                surviving.append(mem)
            else:
                stats["memories_forgotten"] += 1

        self.episodic = surviving

        # =====================================================================
        # PHASE 2: PATTERN EXTRACTION with ACTIONABLE KNOWLEDGE
        # Create new concepts OR update existing ones
        # =====================================================================
        for pattern, count in self.pattern_counts.items():
            if count >= 3:
                success_rate = None
                if pattern in self.success_patterns and len(self.success_patterns[pattern]) >= 3:
                    success_rate = sum(self.success_patterns[pattern]) / len(self.success_patterns[pattern])

                existing = [m for m in self.semantic if pattern in m.tags]

                if existing and self.config.enable_concept_updates:
                    # UPDATE existing concept (not one-and-done)
                    concept = existing[0]
                    recent_mems = [m for m in self.pattern_memories.get(pattern, [])
                                  if m.session_id >= current_session - 3]

                    if recent_mems and self.llm:
                        refined_content = self.llm.refine_concept(concept, recent_mems)
                        if refined_content != concept.content:
                            concept.content = refined_content
                            concept.embedding = self.embeddings.encode(refined_content)
                            concept.version += 1
                            concept.timestamp = time.time()
                            stats["concepts_updated"] += 1

                elif not existing:
                    # CREATE new concept with ACTIONABLE knowledge
                    if self.llm and pattern in self.pattern_memories:
                        pattern_mems = self.pattern_memories[pattern]
                        # Use actionable knowledge extraction instead of simple summary
                        concept_text = self.llm.extract_actionable_knowledge(
                            pattern, pattern_mems, success_rate or 0.5
                        )
                    else:
                        concept_text = f"Concept: {pattern} (observed {count} times)"
                        if success_rate is not None:
                            concept_text += f" - success rate: {success_rate:.0%}"

                    embedding = self.embeddings.encode(concept_text)
                    concept = Memory(
                        id=f"{self.agent_id}_concept_{len(self.semantic)}",
                        content=concept_text,
                        timestamp=time.time(),
                        memory_type="concept",
                        embedding=embedding,
                        significance=0.8,
                        strength=1.0,
                        tags=[pattern]
                    )
                    self.semantic.append(concept)
                    stats["concepts_extracted"] += 1

        # =====================================================================
        # PHASE 3: PROCEDURE LEARNING (successes)
        # =====================================================================
        for pattern, successes in self.success_patterns.items():
            if len(successes) >= 5:
                success_rate = sum(successes) / len(successes)
                if success_rate >= 0.7:
                    existing = [m for m in self.procedural if pattern in m.tags and 'WARNING' not in m.content]
                    if not existing:
                        if self.llm and pattern in self.pattern_memories:
                            procedure_text = self.llm.extract_actionable_knowledge(
                                pattern, self.pattern_memories[pattern], success_rate
                            )
                        else:
                            procedure_text = f"Procedure: {pattern} works ({success_rate:.0%} success, n={len(successes)})"

                        embedding = self.embeddings.encode(procedure_text)
                        procedure = Memory(
                            id=f"{self.agent_id}_proc_{len(self.procedural)}",
                            content=procedure_text,
                            timestamp=time.time(),
                            memory_type="procedure",
                            embedding=embedding,
                            significance=0.9,
                            strength=1.0,
                            tags=[pattern]
                        )
                        self.procedural.append(procedure)
                        stats["procedures_learned"] += 1

        # =====================================================================
        # PHASE 4: ANTI-PROCEDURE LEARNING (failures)
        # Learn what NOT to do - equally valuable as successes
        # =====================================================================
        if self.config.enable_failure_learning:
            for pattern, successes in self.success_patterns.items():
                if len(successes) >= 5:
                    failure_rate = 1.0 - (sum(successes) / len(successes))
                    if failure_rate >= (1.0 - self.config.failure_threshold):
                        # High failure rate - create anti-procedure
                        existing_warning = [m for m in self.procedural
                                           if pattern in m.tags and 'WARNING' in m.content]
                        if not existing_warning:
                            if self.llm and pattern in self.pattern_memories:
                                anti_proc_text = self.llm.generate_anti_procedure(
                                    pattern, self.pattern_memories[pattern], failure_rate
                                )
                            else:
                                anti_proc_text = f"[WARNING] {pattern} has {failure_rate:.0%} failure rate"

                            embedding = self.embeddings.encode(anti_proc_text)
                            anti_proc = Memory(
                                id=f"{self.agent_id}_antiproc_{len(self.procedural)}",
                                content=anti_proc_text,
                                timestamp=time.time(),
                                memory_type="anti_procedure",
                                embedding=embedding,
                                significance=0.85,
                                strength=1.0,
                                tags=[pattern, "warning"]
                            )
                            self.procedural.append(anti_proc)
                            stats["anti_procedures_learned"] += 1

        stats["memories_after"] = len(self.episodic)
        stats["total_semantic"] = len(self.semantic)
        stats["total_procedural"] = len(self.procedural)

        return stats

    def get_memory_count(self) -> int:
        return len(self.episodic) + len(self.semantic) + len(self.procedural)

    def get_stats(self) -> Dict:
        return {
            "total_memories": self.get_memory_count(),
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "procedural": len(self.procedural)
        }


# ============================================================================
# TASK GENERATION (deterministic)
# ============================================================================

COMPONENTS = ["UserService", "OrderProcessor", "PaymentHandler", "AuthManager",
              "DataLoader", "CacheManager", "MessageQueue", "TaskScheduler",
              "NotificationService", "ReportGenerator", "SearchEngine", "Analytics"]

TASK_TYPES = ["bug_fix", "feature_add", "refactor", "debug", "optimize", "test"]

RECURRING_PATTERNS = [
    ("bug_fix", "AuthManager"),
    ("optimize", "CacheManager"),
    ("debug", "MessageQueue"),
    ("feature_add", "UserService"),
]


def generate_tasks(config: ExperimentConfig, seed: int) -> List[List[Dict]]:
    """Generate all tasks for all sessions (deterministic)."""
    rng = random.Random(seed)
    all_sessions = []

    for session_id in range(config.num_sessions):
        tasks = []
        for task_id in range(config.tasks_per_session):
            if rng.random() < config.recurring_pattern_prob and RECURRING_PATTERNS:
                task_type, component = rng.choice(RECURRING_PATTERNS)
            else:
                task_type = rng.choice(TASK_TYPES)
                component = rng.choice(COMPONENTS)

            difficulty = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]

            tasks.append({
                "id": f"task_{seed}_{session_id}_{task_id}",
                "session_id": session_id,
                "type": task_type,
                "component": component,
                "difficulty": difficulty,
                "pattern_key": f"{task_type}_{component}"
            })
        all_sessions.append(tasks)

    return all_sessions


def compute_success_probability(task: Dict, retrieved: List[Tuple[Memory, float]],
                                 config: ExperimentConfig = None, llm: RemoteLLM = None,
                                 classifier: ClassifierEvaluator = None) -> float:
    """Compute task success probability based on context.

    IMPORTANT: This function is AGNOSTIC to memory type.
    The advantage of consolidation should come from:
    1. Better retrieval (concepts have cleaner embeddings, less noise)
    2. More relevant matches (procedures capture patterns)
    3. Less irrelevant clutter (forgetting removes old noise)

    We DO NOT give bonuses based on memory_type - that would be circular logic.

    Evaluation modes (in priority order):
    1. Classifier evaluation (Option E) - deterministic, trained locally
    2. LLM evaluation (Option D) - uses LLM with temperature control
    3. Heuristic evaluation - fallback
    """
    memories = [mem for mem, _ in retrieved]
    similarity_scores = [score for _, score in retrieved]

    # Use classifier evaluation if enabled and trained (Option E)
    if config and getattr(config, 'use_classifier_evaluation', False) and classifier:
        if classifier.is_trained:
            return classifier.evaluate(task, memories, similarity_scores)

    # Use LLM evaluation if enabled (Option D)
    if config and getattr(config, 'use_llm_evaluation', False) and llm:
        llm_result = llm.evaluate_task(task, memories)

        # Collect training data if in training mode (Option E)
        if config and getattr(config, 'classifier_training_mode', False) and classifier:
            classifier.collect_training_sample(task, memories, similarity_scores, llm_result)

        return llm_result

    # Fallback to heuristic evaluation
    base_prob = {1: 0.7, 2: 0.5, 3: 0.3}[task["difficulty"]]

    if not retrieved:
        return base_prob

    pattern_key = task["pattern_key"]

    # Compute relevance purely from embedding similarity scores
    # Higher similarity = more relevant context = better success
    total_similarity = 0.0
    relevant_count = 0

    for mem, score in retrieved:
        # Score is already the cosine similarity - use it directly
        total_similarity += score

        # Count relevant memories based on content match, NOT memory_type
        if pattern_key in mem.tags or task["type"] in mem.content or task["component"] in mem.content:
            relevant_count += 1

    # Average similarity of retrieved memories (0 to ~1)
    avg_similarity = total_similarity / len(retrieved) if retrieved else 0

    # Relevance boost: having memories that match the task pattern
    relevance_boost = min(0.15, 0.03 * relevant_count)

    # Similarity boost: higher quality retrieval (cleaner embeddings = higher scores)
    # This is the FAIR test - consolidated memories should naturally have better embeddings
    similarity_boost = min(0.15, avg_similarity * 0.2)

    return min(0.95, base_prob + relevance_boost + similarity_boost)


# ============================================================================
# SINGLE EXPERIMENT RUN
# ============================================================================

@dataclass
class RunResult:
    """Result of a single experiment run."""
    seed: int
    agent_type: str
    overall_success_rate: float
    avg_retrieval_score: float
    final_memory_count: int
    final_episodic: int
    final_semantic: int
    final_procedural: int
    session_success_rates: List[float]
    consolidation_stats: List[Dict]


def run_single_experiment(agent_class, config: ExperimentConfig, seed: int) -> RunResult:
    """Run a single experiment with a specific seed."""
    # Generate deterministic tasks
    all_sessions = generate_tasks(config, seed)

    # Create agent
    agent_id = f"{agent_class.__name__}_{seed}"
    agent = agent_class(agent_id, config)

    # Initialize evaluation components
    llm = get_llm(config) if getattr(config, 'use_llm_evaluation', False) else None
    # Initialize classifier if using it for evaluation OR collecting training data
    classifier = get_classifier(config) if (
        getattr(config, 'use_classifier_evaluation', False) or
        getattr(config, 'classifier_training_mode', False)
    ) else None

    # Run experiment
    rng = random.Random(seed + 1000)  # Different seed for success sampling
    all_successes = []
    all_retrieval_scores = []
    session_success_rates = []
    consolidation_stats = []

    for session_id, tasks in enumerate(all_sessions):
        session_successes = []

        for task in tasks:
            query = f"{task['type']} {task['component']}"
            retrieved = agent.retrieve(query, k=5)

            # Compute success probability using configured evaluation method
            success_prob = compute_success_probability(task, retrieved, config, llm, classifier)
            success = rng.random() < success_prob
            session_successes.append(success)
            all_successes.append(success)

            if retrieved:
                avg_score = sum(s for _, s in retrieved) / len(retrieved)
                all_retrieval_scores.append(avg_score)
                for mem, _ in retrieved:
                    mem.access_count += 1

            content = f"Task: {task['type']} on {task['component']} - {'success' if success else 'failed'}"
            tags = [task["pattern_key"], task["type"], task["component"]]
            agent.store(content, session_id, significance=task["difficulty"]/3,
                       tags=tags, success=success)

        session_success_rates.append(sum(session_successes) / len(session_successes))

        # Consolidate periodically
        if (session_id + 1) % config.consolidation_interval == 0 and session_id > 0:
            stats = agent.consolidate()
            consolidation_stats.append(stats)

    # Get final stats
    agent_stats = agent.get_stats()

    return RunResult(
        seed=seed,
        agent_type=agent_class.__name__,
        overall_success_rate=sum(all_successes) / len(all_successes),
        avg_retrieval_score=sum(all_retrieval_scores) / len(all_retrieval_scores) if all_retrieval_scores else 0,
        final_memory_count=agent_stats["total_memories"],
        final_episodic=agent_stats["episodic"],
        final_semantic=agent_stats["semantic"],
        final_procedural=agent_stats["procedural"],
        session_success_rates=session_success_rates,
        consolidation_stats=consolidation_stats
    )


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

@dataclass
class StatisticalResult:
    """Complete statistical analysis results."""
    # Descriptive statistics
    n: int
    mean: float
    std: float
    stderr: float
    ci_lower: float
    ci_upper: float
    min_val: float
    max_val: float
    median: float

    # Individual values for reproducibility
    values: List[float]


@dataclass
class HypothesisTest:
    """Results of hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    effect_interpretation: str
    is_significant: bool
    confidence_level: float


def compute_statistics(values: List[float], confidence: float = 0.95) -> StatisticalResult:
    """Compute descriptive statistics with confidence intervals."""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    stderr = std / np.sqrt(n)

    # Confidence interval using t-distribution
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    ci_lower = mean - t_crit * stderr
    ci_upper = mean + t_crit * stderr

    return StatisticalResult(
        n=n,
        mean=float(mean),
        std=float(std),
        stderr=float(stderr),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        min_val=float(np.min(values)),
        max_val=float(np.max(values)),
        median=float(np.median(values)),
        values=values
    )


def cohens_d(group1: List[float], group2: List[float]) -> Tuple[float, str]:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return float(d), interpretation


def perform_hypothesis_test(group1: List[float], group2: List[float],
                           confidence: float = 0.95) -> HypothesisTest:
    """Perform Welch's t-test (unequal variance t-test)."""
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    # Effect size
    d, interpretation = cohens_d(group1, group2)

    alpha = 1 - confidence
    is_significant = p_value < alpha

    return HypothesisTest(
        test_name="Welch's t-test",
        statistic=float(t_stat),
        p_value=float(p_value),
        effect_size=d,
        effect_interpretation=interpretation,
        is_significant=is_significant,
        confidence_level=confidence
    )


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

@dataclass
class ExperimentResults:
    """Complete experiment results with statistical analysis."""
    config: Dict
    config_hash: str
    timestamp: str

    # Raw results
    flat_results: List[Dict]
    consolidation_results: List[Dict]

    # Statistical analysis
    flat_stats: Dict
    consolidation_stats: Dict

    # Hypothesis test
    hypothesis_test: Dict

    # Summary
    summary: Dict


def run_rigorous_experiment(config: ExperimentConfig = None) -> ExperimentResults:
    """Run complete rigorous experiment with statistical analysis."""
    if config is None:
        config = ExperimentConfig()

    print("=" * 70)
    print("RIGOROUS REPRODUCIBLE EXPERIMENT")
    print("=" * 70)
    print(f"Config Hash: {config.get_hash()}")
    print(f"Seeds: {config.num_seeds} (base: {config.base_seed})")
    print(f"Sessions: {config.num_sessions}, Tasks/session: {config.tasks_per_session}")
    print(f"Embedding Server: {config.embedding_server}")
    print(f"Confidence Level: {config.confidence_level:.0%}")
    print("=" * 70)

    # Verify connection
    try:
        get_embeddings(config)
        print("✓ Connected to embedding server")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        raise

    # Prepare experiments
    experiments = []
    for i in range(config.num_seeds):
        seed = config.base_seed + i
        experiments.append(("flat", FlatMemoryAgent, seed))
        experiments.append(("cons", ConsolidationAgent, seed))

    print(f"\nRunning {len(experiments)} experiments ({config.num_seeds} seeds × 2 agents)...")

    flat_results = []
    consolidation_results = []

    # Run in parallel
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {}
        for exp_type, agent_class, seed in experiments:
            future = executor.submit(run_single_experiment, agent_class, config, seed)
            futures[future] = (exp_type, seed)

        completed = 0
        for future in as_completed(futures):
            exp_type, seed = futures[future]
            completed += 1
            try:
                result = future.result()
                if exp_type == "flat":
                    flat_results.append(result)
                else:
                    consolidation_results.append(result)
                print(f"  [{completed}/{len(experiments)}] {exp_type}_seed{seed}: "
                      f"{result.overall_success_rate:.1%} success")
            except Exception as e:
                print(f"  [{completed}/{len(experiments)}] {exp_type}_seed{seed}: FAILED - {e}")

    # Train classifier if in training mode
    if getattr(config, 'classifier_training_mode', False):
        classifier = get_classifier(config)
        if classifier and len(classifier.training_data) > 0:
            print(f"\n[Classifier] Collected {len(classifier.training_data)} training samples")
            if classifier.train():
                print("[Classifier] Training complete! Model saved.")
            else:
                print("[Classifier] Training failed - not enough samples or error occurred")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    flat_success_rates = [r.overall_success_rate for r in flat_results]
    cons_success_rates = [r.overall_success_rate for r in consolidation_results]

    flat_stats = compute_statistics(flat_success_rates, config.confidence_level)
    cons_stats = compute_statistics(cons_success_rates, config.confidence_level)

    print(f"\nFlat Memory (n={flat_stats.n}):")
    print(f"  Mean: {flat_stats.mean:.1%} ± {flat_stats.std:.1%}")
    print(f"  95% CI: [{flat_stats.ci_lower:.1%}, {flat_stats.ci_upper:.1%}]")
    print(f"  Range: [{flat_stats.min_val:.1%}, {flat_stats.max_val:.1%}]")

    print(f"\nConsolidation (n={cons_stats.n}):")
    print(f"  Mean: {cons_stats.mean:.1%} ± {cons_stats.std:.1%}")
    print(f"  95% CI: [{cons_stats.ci_lower:.1%}, {cons_stats.ci_upper:.1%}]")
    print(f"  Range: [{cons_stats.min_val:.1%}, {cons_stats.max_val:.1%}]")

    # Hypothesis test
    test = perform_hypothesis_test(cons_success_rates, flat_success_rates, config.confidence_level)

    print(f"\n{'=' * 70}")
    print("HYPOTHESIS TEST")
    print("=" * 70)
    print(f"H0: Consolidation success rate = Flat memory success rate")
    print(f"H1: Consolidation success rate ≠ Flat memory success rate")
    print(f"\nTest: {test.test_name}")
    print(f"t-statistic: {test.statistic:.4f}")
    print(f"p-value: {test.p_value:.6f}")
    print(f"Effect size (Cohen's d): {test.effect_size:.3f} ({test.effect_interpretation})")
    print(f"\nResult: {'REJECT H0' if test.is_significant else 'FAIL TO REJECT H0'}")

    if test.is_significant:
        print(f"✓ The difference is statistically significant (p < {1-config.confidence_level:.2f})")
    else:
        print(f"✗ The difference is NOT statistically significant (p ≥ {1-config.confidence_level:.2f})")

    # Effect summary
    improvement = cons_stats.mean - flat_stats.mean
    improvement_pct = (cons_stats.mean - flat_stats.mean) / flat_stats.mean * 100

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Absolute Improvement: {improvement:.1%} ({improvement_pct:+.1f}% relative)")
    print(f"Effect Size: {test.effect_interpretation} (d = {test.effect_size:.3f})")
    print(f"Statistical Significance: {'Yes' if test.is_significant else 'No'} (p = {test.p_value:.6f})")

    # Prepare results
    results = ExperimentResults(
        config=config.to_dict(),
        config_hash=config.get_hash(),
        timestamp=datetime.now().isoformat(),
        flat_results=[asdict(r) for r in flat_results],
        consolidation_results=[asdict(r) for r in consolidation_results],
        flat_stats=asdict(flat_stats),
        consolidation_stats=asdict(cons_stats),
        hypothesis_test=asdict(test),
        summary={
            "absolute_improvement": float(improvement),
            "relative_improvement_pct": float(improvement_pct),
            "effect_size": test.effect_size,
            "effect_interpretation": test.effect_interpretation,
            "p_value": test.p_value,
            "is_significant": test.is_significant,
            "winner": "consolidation" if improvement > 0 and test.is_significant else "flat_memory" if improvement < 0 and test.is_significant else "no_significant_difference"
        }
    )

    # Save results
    output_dir = Path("/home/marc/research-papers/memory-consolidation/experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"rigorous_experiment_{config.get_hash()}.json"

    # Convert to serializable dict (avoid circular references)
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, (int, float, str, type(None))):
            return obj
        else:
            return str(obj)

    serializable_results = to_serializable(asdict(results))
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 70)

    return results


def generate_reproducibility_report(results: ExperimentResults) -> str:
    """Generate a reproducibility report."""
    report = []
    report.append("=" * 70)
    report.append("REPRODUCIBILITY REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("To reproduce this experiment, use the following configuration:")
    report.append("")
    report.append("```python")
    report.append("from rigorous_experiment import ExperimentConfig, run_rigorous_experiment")
    report.append("")
    report.append("config = ExperimentConfig(")
    for key, value in results.config.items():
        if isinstance(value, str):
            report.append(f"    {key}=\"{value}\",")
        else:
            report.append(f"    {key}={value},")
    report.append(")")
    report.append("")
    report.append("results = run_rigorous_experiment(config)")
    report.append("```")
    report.append("")
    report.append(f"Configuration Hash: {results.config_hash}")
    report.append(f"Timestamp: {results.timestamp}")
    report.append("")
    report.append("Expected Results:")
    report.append(f"  Flat Memory: {results.flat_stats['mean']:.4f} ± {results.flat_stats['std']:.4f}")
    report.append(f"  Consolidation: {results.consolidation_stats['mean']:.4f} ± {results.consolidation_stats['std']:.4f}")
    report.append(f"  p-value: {results.hypothesis_test['p_value']:.6f}")
    report.append(f"  Effect size: {results.hypothesis_test['effect_size']:.4f}")
    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run with default configuration
    config = ExperimentConfig(
        num_seeds=10,  # 10 seeds for statistical power
        num_sessions=30,
        tasks_per_session=15
    )

    results = run_rigorous_experiment(config)

    # Generate reproducibility report
    report = generate_reproducibility_report(results)
    print("\n" + report)

    # Save report
    report_path = Path("/home/marc/research-papers/memory-consolidation/experiments/results/reproducibility_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
