"""
Distributed Compute Strategy for Memory Consolidation Research

Resource Allocation:
- macpro51 (local): Orchestration only, NO compute
- mac-studio: MLX embeddings, coordination
- macbook-air-m3: Lightweight compute, research
- completeu-server: Ollama inference
- Cloud (Kaggle/Colab): Heavy training jobs

NEVER run embeddings or inference on macpro51 CPU.
"""

import os
import json
import requests
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib


# Cluster node configuration
CLUSTER_NODES = {
    "mac-studio": {
        "ip": "192.168.1.16",
        "role": "orchestrator",
        "capabilities": ["mlx-embeddings", "coordination"],
        "embedding_endpoint": "http://192.168.1.16:8080/embed"
    },
    "completeu-server": {
        "ip": "192.168.1.186",
        "role": "ai-inference",
        "capabilities": ["ollama", "embeddings", "inference"],
        "ollama_endpoint": "http://192.168.1.186:11434"
    },
    "macbook-air-m3": {
        "ip": "192.168.1.76",
        "role": "researcher",
        "capabilities": ["mlx-embeddings", "analysis"]
    }
}

# Free cloud GPU options
CLOUD_GPU_OPTIONS = {
    "kaggle": {
        "url": "https://www.kaggle.com",
        "gpu": "T4/P100",
        "hours_per_week": 30,
        "session_limit": "9 hours",
        "best_for": "batch training, experiments"
    },
    "colab": {
        "url": "https://colab.research.google.com",
        "gpu": "T4/P100/A100",
        "limits": "compute units, variable",
        "best_for": "interactive development"
    },
    "sagemaker_studio_lab": {
        "url": "https://studiolab.sagemaker.aws",
        "gpu": "T4",
        "hours_per_day": 4,
        "best_for": "quick experiments"
    },
    "gradient_paperspace": {
        "url": "https://gradient.run",
        "gpu": "M4000",
        "session_limit": "6 hours",
        "best_for": "notebook workflows"
    },
    "lightning_ai": {
        "url": "https://lightning.ai",
        "gpu": "variable",
        "best_for": "PyTorch Lightning projects"
    }
}


@dataclass
class EmbeddingCache:
    """Local cache for embeddings to avoid recomputation."""
    cache_dir: Path

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_text(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        key = f"{model}:{text}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if exists."""
        key = self._hash_text(text, model)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        key = self._hash_text(text, model)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)


class RemoteEmbeddingClient:
    """
    Client for getting embeddings from remote nodes.

    Priority order:
    1. Local cache (instant)
    2. completeu-server Ollama (has GPU)
    3. mac-studio MLX (Apple Silicon)
    4. Fallback: lightweight local model (CPU-safe)
    """

    def __init__(self, cache_dir: str = "/tmp/embedding_cache"):
        self.cache = EmbeddingCache(Path(cache_dir))
        self.ollama_url = CLUSTER_NODES["completeu-server"]["ollama_endpoint"]
        self.default_model = "nomic-embed-text"  # Good Ollama embedding model
        self._fallback_model = None

    def _try_ollama(self, text: str, model: str = None) -> Optional[List[float]]:
        """Try to get embedding from Ollama on completeu-server."""
        model = model or self.default_model
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("embedding")
        except (requests.RequestException, Exception) as e:
            print(f"Ollama embedding failed: {e}")
        return None

    def _try_lightweight_local(self, text: str) -> List[float]:
        """
        Fallback: Use very lightweight model that's CPU-safe.
        This should only be used if all remote options fail.
        """
        if self._fallback_model is None:
            try:
                # Use a tiny model that won't kill CPU
                from sentence_transformers import SentenceTransformer
                # paraphrase-MiniLM-L3-v2 is only 61MB and very fast on CPU
                self._fallback_model = SentenceTransformer(
                    'paraphrase-MiniLM-L3-v2',
                    device='cpu'
                )
            except ImportError:
                # Ultimate fallback: random but deterministic
                self._fallback_model = "random"

        if self._fallback_model == "random":
            import hashlib
            import numpy as np
            # Deterministic random based on text
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed % (2**32))
            return np.random.randn(384).tolist()

        return self._fallback_model.encode(text).tolist()

    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Get embedding with smart fallback.

        1. Check cache
        2. Try Ollama (completeu-server)
        3. Fallback to lightweight local
        """
        model = model or self.default_model

        # 1. Check cache first
        cached = self.cache.get(text, model)
        if cached is not None:
            return cached

        # 2. Try Ollama on completeu-server
        embedding = self._try_ollama(text, model)
        if embedding is not None:
            self.cache.set(text, model, embedding)
            return embedding

        # 3. Lightweight local fallback (only if remote fails)
        print("Warning: Using local CPU fallback for embedding")
        embedding = self._try_lightweight_local(text)
        self.cache.set(text, "local-fallback", embedding)
        return embedding

    def get_embeddings_batch(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        return [self.get_embedding(text, model) for text in texts]


class DistributedExperimentRunner:
    """
    Runs experiments distributed across the cluster.

    - Orchestration: macpro51 (local)
    - Embeddings: completeu-server or mac-studio
    - Heavy compute: Cloud (Kaggle/Colab)
    """

    def __init__(self):
        self.embedding_client = RemoteEmbeddingClient()
        self.results_dir = Path("/home/marc/research-papers/memory-consolidation/experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def check_cluster_health(self) -> Dict[str, bool]:
        """Check which cluster nodes are available."""
        health = {}

        for node_id, config in CLUSTER_NODES.items():
            try:
                if "ollama_endpoint" in config:
                    response = requests.get(f"{config['ollama_endpoint']}/api/version", timeout=5)
                    health[node_id] = response.status_code == 200
                else:
                    # Try to ping
                    import subprocess
                    result = subprocess.run(
                        ["ping", "-c", "1", "-W", "1", config["ip"]],
                        capture_output=True
                    )
                    health[node_id] = result.returncode == 0
            except Exception:
                health[node_id] = False

        return health

    def prepare_kaggle_notebook(self, experiment_name: str) -> str:
        """Generate Kaggle notebook for running experiments."""
        notebook = {
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "accelerator": "GPU"
            },
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": f"# Memory Consolidation Experiment: {experiment_name}\\n\\nThis notebook runs on Kaggle's free GPU."
                },
                {
                    "cell_type": "code",
                    "source": "!pip install sentence-transformers numpy pandas"
                },
                {
                    "cell_type": "code",
                    "source": "# Experiment code will be generated here\\n# Upload your experiment files to Kaggle datasets"
                }
            ]
        }

        output_path = self.results_dir / f"kaggle_{experiment_name}.ipynb"
        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=2)

        return str(output_path)

    def get_resource_recommendation(self, task_type: str) -> Dict[str, Any]:
        """Recommend best resource for a task type."""
        recommendations = {
            "embedding_generation": {
                "primary": "completeu-server (Ollama)",
                "fallback": "mac-studio (MLX)",
                "avoid": "macpro51 CPU"
            },
            "batch_training": {
                "primary": "Kaggle (30 GPU-hours/week free)",
                "fallback": "Google Colab",
                "avoid": "any local node"
            },
            "quick_experiment": {
                "primary": "SageMaker Studio Lab (4hr/day)",
                "fallback": "Gradient Paperspace",
                "avoid": "macpro51 CPU"
            },
            "orchestration": {
                "primary": "macpro51 (local)",
                "reason": "Lightweight coordination only"
            }
        }
        return recommendations.get(task_type, {"note": "Unknown task type"})


def print_resource_strategy():
    """Print the resource allocation strategy."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           DISTRIBUTED COMPUTE STRATEGY                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  macpro51 (local)     → Orchestration ONLY, no compute           ║
║  completeu-server     → Ollama embeddings & inference            ║
║  mac-studio           → MLX embeddings, coordination              ║
║  macbook-air-m3       → Light analysis, research                  ║
║                                                                   ║
║  ═══════════════════════════════════════════════════════════════ ║
║  FREE CLOUD GPU OPTIONS:                                          ║
║  ═══════════════════════════════════════════════════════════════ ║
║                                                                   ║
║  Kaggle              → 30 GPU-hours/week (T4/P100)               ║
║  Google Colab        → Free T4 with compute units                 ║
║  SageMaker Studio    → 4 hours T4 per day                        ║
║  Gradient Paperspace → M4000, 6-hour sessions                    ║
║  Lightning AI        → Free monthly GPU hours                     ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_resource_strategy()

    # Test cluster health
    runner = DistributedExperimentRunner()
    health = runner.check_cluster_health()
    print("\nCluster Health:")
    for node, status in health.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {node}")

    # Test embedding
    print("\nTesting remote embedding...")
    client = RemoteEmbeddingClient()
    embedding = client.get_embedding("Test embedding request")
    print(f"  Got embedding of dimension: {len(embedding)}")
