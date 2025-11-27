"""
Metrics Collection for Memory Consolidation Experiments

Collects and analyzes all metrics defined in the research plan:
- Task Success Rate
- Retrieval Precision@k
- Memory Size / Storage Efficiency
- Retrieval Latency
- Cross-Session Coherence
- Learning Curve metrics
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    session_id: int
    agent_type: str
    success: bool
    success_score: float  # 0.0 to 1.0
    execution_time_ms: float
    retrieval_results: Dict[str, Any]
    memory_stats: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation."""
    query: str
    retrieved_ids: List[str]
    relevant_ids: List[str]  # Ground truth
    retrieval_time_ms: float
    precision_at_k: Dict[int, float]  # k -> precision
    recall_at_k: Dict[int, float]
    relevance_scores: List[float]


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""
    session_id: int
    agent_type: str
    tasks_completed: int
    tasks_successful: int
    avg_success_score: float
    total_retrieval_time_ms: float
    avg_retrieval_precision: float
    memory_size_bytes: int
    coherence_score: float  # 0-1, consistency with prior sessions


@dataclass
class ExperimentMetrics:
    """Complete metrics for an experiment run."""
    experiment_id: str
    agent_type: str
    start_time: datetime
    end_time: Optional[datetime]
    sessions: List[SessionMetrics]
    task_results: List[TaskResult]

    # Aggregate metrics
    overall_success_rate: float = 0.0
    avg_retrieval_precision: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    total_memory_size: int = 0
    learning_curve_slope: float = 0.0
    cross_session_coherence: float = 0.0


class MetricsCollector:
    """
    Collects and computes all experiment metrics.

    Primary metrics from research plan:
    1. Task Success Rate: % tasks completed correctly
    2. Retrieval Precision@k: Relevance of top-k retrieved memories
    3. Memory Size: Total storage used
    4. Retrieval Latency: Time to retrieve relevant context
    5. Cross-Session Coherence: Consistency across sessions
    6. Learning Curve: Improvement rate over sessions
    """

    def __init__(self, experiment_id: str, agent_type: str):
        self.experiment_id = experiment_id
        self.agent_type = agent_type
        self.start_time = datetime.now()

        self.task_results: List[TaskResult] = []
        self.retrieval_metrics: List[RetrievalMetrics] = []
        self.session_metrics: List[SessionMetrics] = []
        self._current_session: Optional[int] = None
        self._session_tasks: Dict[int, List[TaskResult]] = defaultdict(list)

    def record_task_result(self, result: TaskResult) -> None:
        """Record a task execution result."""
        self.task_results.append(result)
        self._session_tasks[result.session_id].append(result)

    def record_retrieval(
        self,
        query: str,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        retrieval_time_ms: float,
        relevance_scores: List[float] = None
    ) -> RetrievalMetrics:
        """Record and compute retrieval metrics."""

        # Compute precision@k for k = 1, 3, 5, 10
        precision_at_k = {}
        recall_at_k = {}

        for k in [1, 3, 5, 10]:
            top_k = set(retrieved_ids[:k])
            relevant_set = set(relevant_ids)

            if k <= len(retrieved_ids):
                hits = len(top_k & relevant_set)
                precision_at_k[k] = hits / k
                recall_at_k[k] = hits / len(relevant_set) if relevant_set else 0.0
            else:
                precision_at_k[k] = 0.0
                recall_at_k[k] = 0.0

        metrics = RetrievalMetrics(
            query=query,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            retrieval_time_ms=retrieval_time_ms,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            relevance_scores=relevance_scores or []
        )

        self.retrieval_metrics.append(metrics)
        return metrics

    def compute_session_metrics(self, session_id: int) -> SessionMetrics:
        """Compute aggregated metrics for a session."""
        session_tasks = self._session_tasks[session_id]

        if not session_tasks:
            return SessionMetrics(
                session_id=session_id,
                agent_type=self.agent_type,
                tasks_completed=0,
                tasks_successful=0,
                avg_success_score=0.0,
                total_retrieval_time_ms=0.0,
                avg_retrieval_precision=0.0,
                memory_size_bytes=0,
                coherence_score=0.0
            )

        tasks_successful = sum(1 for t in session_tasks if t.success)
        avg_success = np.mean([t.success_score for t in session_tasks])

        # Retrieval metrics for this session
        session_retrievals = [r for r in self.retrieval_metrics
                             if any(t.task_id in r.query for t in session_tasks)]
        total_retrieval_time = sum(r.retrieval_time_ms for r in session_retrievals)
        avg_precision = np.mean([r.precision_at_k.get(5, 0) for r in session_retrievals]) if session_retrievals else 0

        # Memory size from last task
        memory_size = session_tasks[-1].memory_stats.get("total_size_bytes", 0) if session_tasks else 0

        # Coherence score: consistency with prior sessions
        coherence = self._compute_coherence(session_id)

        metrics = SessionMetrics(
            session_id=session_id,
            agent_type=self.agent_type,
            tasks_completed=len(session_tasks),
            tasks_successful=tasks_successful,
            avg_success_score=float(avg_success),
            total_retrieval_time_ms=total_retrieval_time,
            avg_retrieval_precision=float(avg_precision),
            memory_size_bytes=memory_size,
            coherence_score=coherence
        )

        self.session_metrics.append(metrics)
        return metrics

    def _compute_coherence(self, session_id: int) -> float:
        """
        Compute cross-session coherence.

        Measures how consistently the agent references prior context.
        Higher score = better use of cross-session information.
        """
        if session_id <= 1:
            return 1.0  # First session is fully coherent with itself

        current_tasks = self._session_tasks[session_id]
        prior_tasks = []
        for sid in range(1, session_id):
            prior_tasks.extend(self._session_tasks[sid])

        if not current_tasks or not prior_tasks:
            return 0.0

        # Coherence: % of tasks that successfully retrieved relevant prior context
        coherent_tasks = 0
        for task in current_tasks:
            # Check if task used cross-session context
            retrieval_info = task.retrieval_results
            if retrieval_info.get("cross_session_retrieved", 0) > 0:
                coherent_tasks += 1

        return coherent_tasks / len(current_tasks)

    def compute_learning_curve(self) -> Tuple[float, List[float]]:
        """
        Compute learning curve slope.

        Returns (slope, per_session_success_rates)
        Positive slope = agent is improving over sessions.
        """
        if len(self.session_metrics) < 2:
            return 0.0, []

        success_rates = [s.avg_success_score for s in sorted(self.session_metrics, key=lambda x: x.session_id)]

        # Linear regression for slope
        x = np.arange(len(success_rates))
        if len(x) > 1:
            slope, _ = np.polyfit(x, success_rates, 1)
        else:
            slope = 0.0

        return float(slope), success_rates

    def finalize(self) -> ExperimentMetrics:
        """Compute final experiment metrics."""
        end_time = datetime.now()

        # Overall success rate
        if self.task_results:
            overall_success = sum(1 for t in self.task_results if t.success) / len(self.task_results)
            avg_success_score = np.mean([t.success_score for t in self.task_results])
        else:
            overall_success = 0.0
            avg_success_score = 0.0

        # Retrieval metrics
        if self.retrieval_metrics:
            avg_precision = np.mean([r.precision_at_k.get(5, 0) for r in self.retrieval_metrics])
            avg_latency = np.mean([r.retrieval_time_ms for r in self.retrieval_metrics])
        else:
            avg_precision = 0.0
            avg_latency = 0.0

        # Memory size (final)
        total_memory = self.task_results[-1].memory_stats.get("total_size_bytes", 0) if self.task_results else 0

        # Learning curve
        slope, _ = self.compute_learning_curve()

        # Cross-session coherence (average)
        if self.session_metrics:
            coherence = np.mean([s.coherence_score for s in self.session_metrics])
        else:
            coherence = 0.0

        return ExperimentMetrics(
            experiment_id=self.experiment_id,
            agent_type=self.agent_type,
            start_time=self.start_time,
            end_time=end_time,
            sessions=self.session_metrics,
            task_results=self.task_results,
            overall_success_rate=float(overall_success),
            avg_retrieval_precision=float(avg_precision),
            avg_retrieval_latency_ms=float(avg_latency),
            total_memory_size=total_memory,
            learning_curve_slope=slope,
            cross_session_coherence=float(coherence)
        )

    def save_results(self, path: Path) -> None:
        """Save all metrics to JSON."""
        metrics = self.finalize()

        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if hasattr(obj, '__dict__'):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        data = serialize(metrics)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class MetricsComparator:
    """Compare metrics across different agent types."""

    def __init__(self):
        self.experiments: Dict[str, ExperimentMetrics] = {}

    def add_experiment(self, metrics: ExperimentMetrics) -> None:
        """Add experiment results for comparison."""
        self.experiments[metrics.agent_type] = metrics

    def compare(self) -> Dict[str, Any]:
        """Generate comparison summary."""
        if not self.experiments:
            return {}

        comparison = {
            "agent_types": list(self.experiments.keys()),
            "metrics": {}
        }

        # Compare each metric
        metrics_to_compare = [
            ("overall_success_rate", "higher_is_better"),
            ("avg_retrieval_precision", "higher_is_better"),
            ("avg_retrieval_latency_ms", "lower_is_better"),
            ("total_memory_size", "lower_is_better"),
            ("learning_curve_slope", "higher_is_better"),
            ("cross_session_coherence", "higher_is_better")
        ]

        for metric_name, direction in metrics_to_compare:
            values = {}
            for agent_type, exp in self.experiments.items():
                values[agent_type] = getattr(exp, metric_name, 0)

            if direction == "higher_is_better":
                best = max(values, key=values.get)
            else:
                best = min(values, key=values.get)

            comparison["metrics"][metric_name] = {
                "values": values,
                "best": best,
                "direction": direction
            }

        # Overall ranking (weighted)
        weights = {
            "overall_success_rate": 0.3,
            "avg_retrieval_precision": 0.2,
            "avg_retrieval_latency_ms": -0.1,  # Negative because lower is better
            "total_memory_size": -0.1,
            "learning_curve_slope": 0.15,
            "cross_session_coherence": 0.15
        }

        scores = {}
        for agent_type in self.experiments:
            score = 0
            for metric_name, weight in weights.items():
                values = comparison["metrics"][metric_name]["values"]
                all_values = list(values.values())
                if max(all_values) != min(all_values):
                    normalized = (values[agent_type] - min(all_values)) / (max(all_values) - min(all_values))
                else:
                    normalized = 0.5
                score += weight * normalized
            scores[agent_type] = score

        comparison["overall_scores"] = scores
        comparison["ranking"] = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return comparison

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        comparison = self.compare()
        if not comparison:
            return ""

        agents = comparison["agent_types"]
        metrics = ["overall_success_rate", "avg_retrieval_precision", "avg_retrieval_latency_ms",
                   "total_memory_size", "learning_curve_slope", "cross_session_coherence"]

        metric_labels = {
            "overall_success_rate": r"Success Rate (\%)",
            "avg_retrieval_precision": "Precision@5",
            "avg_retrieval_latency_ms": "Latency (ms)",
            "total_memory_size": "Memory (KB)",
            "learning_curve_slope": "Learning Slope",
            "cross_session_coherence": "Coherence"
        }

        # Header
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Comparison of memory agent types. Best results in \\textbf{bold}.}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{@{}l" + "c" * len(agents) + "@{}}",
            "\\toprule",
            "Metric & " + " & ".join(agents) + " \\\\",
            "\\midrule"
        ]

        # Data rows
        for metric in metrics:
            values = comparison["metrics"][metric]["values"]
            best = comparison["metrics"][metric]["best"]

            row = metric_labels.get(metric, metric)
            for agent in agents:
                val = values[agent]
                # Format based on metric type
                if "rate" in metric or "precision" in metric or "coherence" in metric:
                    formatted = f"{val:.1%}"
                elif "latency" in metric:
                    formatted = f"{val:.1f}"
                elif "memory" in metric:
                    formatted = f"{val/1024:.1f}"
                else:
                    formatted = f"{val:.3f}"

                if agent == best:
                    formatted = f"\\textbf{{{formatted}}}"
                row += f" & {formatted}"

            lines.append(row + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return "\n".join(lines)
