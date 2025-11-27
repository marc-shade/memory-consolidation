"""
Ablation Study for Memory Consolidation Features

Tests each consolidation feature individually to identify which ones
contribute most to the improvement observed in the long-horizon experiment.

Features to ablate:
1. Memory replay during consolidation
2. Concept updates (not one-and-done)
3. Failure learning (anti-procedures)
4. Causal tracking
5. Temporal decay
6. Access boost multiplier
7. Recency half-life
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from datetime import datetime
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from rigorous_experiment import ExperimentConfig, run_rigorous_experiment


# Ablation configurations - each disables ONE feature
ABLATION_CONFIGS = {
    "full": {
        "description": "All features enabled (control)",
        "overrides": {}
    },
    "no_memory_replay": {
        "description": "Disable memory replay during consolidation",
        "overrides": {"enable_memory_replay": False}
    },
    "no_concept_updates": {
        "description": "Disable concept updates (one-and-done concepts)",
        "overrides": {"enable_concept_updates": False}
    },
    "no_failure_learning": {
        "description": "Disable failure learning (no anti-procedures)",
        "overrides": {"enable_failure_learning": False}
    },
    "no_causal_tracking": {
        "description": "Disable causal relationship tracking",
        "overrides": {"enable_causal_tracking": False}
    },
    "no_temporal_decay": {
        "description": "Disable temporal decay (no recency weighting)",
        "overrides": {"enable_temporal_decay": False}
    },
    "no_access_boost": {
        "description": "Disable access boost (set multiplier to 1.0)",
        "overrides": {"access_boost_multiplier": 1.0}
    },
    "no_recency_halflife": {
        "description": "Disable recency half-life (set to very high value)",
        "overrides": {"recency_half_life_sessions": 1000}
    },
}


def create_ablation_config(ablation_name: str) -> ExperimentConfig:
    """Create experiment config with specific ablation."""
    ablation = ABLATION_CONFIGS[ablation_name]

    # Base config (same as 60-session experiment)
    config = ExperimentConfig(
        experiment_name=f'ablation_{ablation_name}',
        experiment_version='3.0.0-ablation',
        num_seeds=5,  # Reduced for faster ablation (still statistically valid)
        num_sessions=60,  # Same as long-horizon experiment
        tasks_per_session=15,
        use_llm_evaluation=True,
        use_llm_summaries=True,
        evaluation_temperature=0.1,
        summary_temperature=0.3,
        use_ollama_cloud=True,
        ollama_cloud_model='glm-4.6',
        # Default: all features enabled
        enable_memory_replay=True,
        enable_concept_updates=True,
        enable_failure_learning=True,
        enable_causal_tracking=True,
        enable_temporal_decay=True,
        replay_top_k=10,
        failure_threshold=0.3,
        access_boost_multiplier=1.2,
        recency_half_life_sessions=10
    )

    # Apply ablation overrides
    for key, value in ablation["overrides"].items():
        setattr(config, key, value)

    return config


def run_ablation(ablation_name: str) -> dict:
    """Run a single ablation experiment."""
    print(f"\n{'='*70}")
    print(f"ABLATION: {ablation_name}")
    print(f"Description: {ABLATION_CONFIGS[ablation_name]['description']}")
    print(f"{'='*70}\n")

    config = create_ablation_config(ablation_name)
    results = run_rigorous_experiment(config)

    return {
        "ablation_name": ablation_name,
        "description": ABLATION_CONFIGS[ablation_name]['description'],
        "config": asdict(config),
        "results": results
    }


def run_ablation_study(ablations: list = None, parallel: bool = False):
    """Run full ablation study.

    Args:
        ablations: List of ablation names to run (None = all)
        parallel: Run ablations in parallel (default: False for resource constraints)
    """
    if ablations is None:
        ablations = list(ABLATION_CONFIGS.keys())

    print("="*70)
    print("ABLATION STUDY: Memory Consolidation Features")
    print("="*70)
    print(f"Running {len(ablations)} ablation configurations:")
    for name in ablations:
        print(f"  - {name}: {ABLATION_CONFIGS[name]['description']}")
    print("="*70)

    all_results = {}

    if parallel:
        # Run in parallel (may hit API rate limits)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(run_ablation, name): name for name in ablations}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    all_results[name] = result
                except Exception as e:
                    print(f"ERROR in ablation {name}: {e}")
                    all_results[name] = {"error": str(e)}
    else:
        # Run sequentially
        for name in ablations:
            try:
                result = run_ablation(name)
                all_results[name] = result
            except Exception as e:
                print(f"ERROR in ablation {name}: {e}")
                all_results[name] = {"error": str(e)}

    # Analyze results
    analyze_ablation_results(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("/home/marc/research-papers/memory-consolidation/experiments/results")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"ablation_study_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return all_results


def analyze_ablation_results(results: dict):
    """Analyze and display ablation results."""
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)

    # Extract consolidation success rates
    ablation_scores = []
    for name, data in results.items():
        if "error" in data:
            print(f"  {name}: ERROR - {data['error']}")
            continue

        if "results" in data and data["results"]:
            # Extract consolidation success rate from results
            res = data["results"]
            if isinstance(res, dict) and "consolidation" in res:
                consol_mean = res["consolidation"].get("mean", 0)
                flat_mean = res["flat"].get("mean", 0)
                diff = consol_mean - flat_mean
                ablation_scores.append({
                    "name": name,
                    "description": data["description"],
                    "consolidation_mean": consol_mean,
                    "flat_mean": flat_mean,
                    "improvement": diff
                })

    # Sort by improvement (highest first)
    ablation_scores.sort(key=lambda x: x["improvement"], reverse=True)

    print("\nRanked by Consolidation Improvement:")
    print("-"*70)
    baseline_improvement = None

    for score in ablation_scores:
        if score["name"] == "full":
            baseline_improvement = score["improvement"]
            marker = " (BASELINE)"
        else:
            marker = ""

        print(f"  {score['name']:25s}: {score['improvement']:+.1%} improvement{marker}")
        print(f"    Consolidation: {score['consolidation_mean']:.1%}, Flat: {score['flat_mean']:.1%}")

    if baseline_improvement is not None:
        print("\n" + "-"*70)
        print("Feature Importance (drop in improvement when disabled):")
        print("-"*70)

        importance = []
        for score in ablation_scores:
            if score["name"] != "full":
                drop = baseline_improvement - score["improvement"]
                importance.append({
                    "feature": score["name"].replace("no_", ""),
                    "drop": drop,
                    "description": score["description"]
                })

        importance.sort(key=lambda x: x["drop"], reverse=True)

        for item in importance:
            pct = (item["drop"] / baseline_improvement * 100) if baseline_improvement != 0 else 0
            print(f"  {item['feature']:25s}: {item['drop']:+.2%} ({pct:+.0f}% of baseline)")

    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--ablations", nargs="+", choices=list(ABLATION_CONFIGS.keys()),
                        help="Specific ablations to run (default: all)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run ablations in parallel")

    args = parser.parse_args()

    run_ablation_study(ablations=args.ablations, parallel=args.parallel)
