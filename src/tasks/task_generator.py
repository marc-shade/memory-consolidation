"""
Multi-Session Task Generator

Creates coding tasks with cross-session dependencies for testing
memory consolidation benefits.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random
import json
from enum import Enum
from pathlib import Path


class TaskType(Enum):
    BUG_FIX = "bug_fix"
    FEATURE_ADD = "feature_add"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TEST_WRITE = "test_write"
    DEBUG = "debug"


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


@dataclass
class Task:
    """A single coding task."""
    id: str
    session_id: int
    task_type: TaskType
    difficulty: Difficulty
    title: str
    description: str
    context_required: List[str]  # IDs of prior tasks that provide context
    expected_output: str
    evaluation_criteria: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A coding session containing multiple tasks."""
    id: int
    tasks: List[Task]
    session_context: str  # Context established in this session
    duration_minutes: int = 60


@dataclass
class TaskDataset:
    """Complete dataset of sessions and tasks."""
    name: str
    sessions: List[Session]
    total_tasks: int
    dependency_density: float  # 0-1, how interconnected tasks are
    created_at: datetime = field(default_factory=datetime.now)


class MultiSessionTaskGenerator:
    """
    Generates multi-session coding tasks with dependencies.

    Task types designed to test memory:
    - Bug fixes that recur with variations
    - Features that build on each other
    - Refactoring that requires remembering prior structure
    - Documentation requiring cross-session knowledge
    """

    # Code snippets for generating realistic tasks
    CODE_PATTERNS = {
        "api_endpoint": """
@app.route('/api/{name}', methods=['{method}'])
def {name}_endpoint(request):
    data = request.json
    result = process_{name}(data)
    return jsonify(result)
""",
        "class_definition": """
class {name}:
    def __init__(self, {params}):
        {init_body}

    def {method}(self, {method_params}):
        {method_body}
""",
        "function": """
def {name}({params}):
    \"\"\"{docstring}\"\"\"
    {body}
    return {return_value}
""",
        "test_case": """
def test_{name}():
    # Setup
    {setup}

    # Execute
    result = {call}

    # Assert
    assert {assertion}
"""
    }

    BUG_TEMPLATES = [
        {
            "title": "Fix off-by-one error in {component}",
            "description": "The {component} has an off-by-one error in the loop at line {line}. The loop should iterate until {correct_condition} instead of {wrong_condition}.",
            "recurrence_pattern": "off_by_one"
        },
        {
            "title": "Fix null reference in {component}",
            "description": "The {component} crashes when {field} is None. Add proper null checking before accessing {field}.{attribute}.",
            "recurrence_pattern": "null_check"
        },
        {
            "title": "Fix race condition in {component}",
            "description": "The {component} has a race condition when multiple requests access {resource}. Add proper locking.",
            "recurrence_pattern": "race_condition"
        },
        {
            "title": "Fix memory leak in {component}",
            "description": "The {component} leaks memory by not closing {resource}. Ensure cleanup in finally block.",
            "recurrence_pattern": "memory_leak"
        },
        {
            "title": "Fix incorrect type conversion in {component}",
            "description": "The {component} incorrectly converts {type1} to {type2}, losing precision. Use {correct_method} instead.",
            "recurrence_pattern": "type_conversion"
        }
    ]

    FEATURE_TEMPLATES = [
        {
            "title": "Add {feature} to {component}",
            "description": "Implement {feature} functionality in {component}. Should support {requirements}.",
            "builds_on": []
        },
        {
            "title": "Extend {component} with {capability}",
            "description": "Add {capability} support to existing {component}. Must maintain backward compatibility.",
            "builds_on": ["Add {feature} to {component}"]
        },
        {
            "title": "Add caching to {component}",
            "description": "Implement caching for {component} to improve performance. Use {cache_type} with TTL of {ttl}.",
            "builds_on": []
        },
        {
            "title": "Add validation to {component}",
            "description": "Add input validation for {component}. Validate: {validations}.",
            "builds_on": []
        }
    ]

    COMPONENTS = [
        "UserService", "OrderProcessor", "PaymentHandler", "AuthManager",
        "DataLoader", "CacheManager", "MessageQueue", "TaskScheduler",
        "FileHandler", "DatabaseConnector", "APIClient", "ConfigManager"
    ]

    FEATURES = [
        "pagination", "filtering", "sorting", "rate limiting",
        "retry logic", "logging", "metrics", "webhooks"
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self._task_counter = 0

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter:04d}"

    def generate_bug_fix_task(
        self,
        session_id: int,
        difficulty: Difficulty,
        prior_tasks: List[Task] = None
    ) -> Task:
        """Generate a bug fix task."""
        template = random.choice(self.BUG_TEMPLATES)
        component = random.choice(self.COMPONENTS)

        title = template["title"].format(component=component)
        description = template["description"].format(
            component=component,
            line=random.randint(10, 100),
            correct_condition="i < len(items)",
            wrong_condition="i <= len(items)",
            field="user",
            attribute="email",
            resource="database_connection",
            type1="float",
            type2="int",
            correct_method="Decimal()"
        )

        # Find related prior tasks (same bug pattern or component)
        context_required = []
        if prior_tasks:
            for task in prior_tasks:
                if (task.metadata.get("recurrence_pattern") == template["recurrence_pattern"] or
                    task.metadata.get("component") == component):
                    context_required.append(task.id)

        return Task(
            id=self._generate_task_id(),
            session_id=session_id,
            task_type=TaskType.BUG_FIX,
            difficulty=difficulty,
            title=title,
            description=description,
            context_required=context_required[:3],  # Max 3 dependencies
            expected_output=f"Fixed {component} with proper {template['recurrence_pattern']} handling",
            evaluation_criteria={
                "bug_fixed": True,
                "no_regressions": True,
                "test_added": difficulty != Difficulty.EASY
            },
            metadata={
                "component": component,
                "recurrence_pattern": template["recurrence_pattern"],
                "bug_type": template["recurrence_pattern"]
            }
        )

    def generate_feature_task(
        self,
        session_id: int,
        difficulty: Difficulty,
        prior_tasks: List[Task] = None
    ) -> Task:
        """Generate a feature addition task."""
        template = random.choice(self.FEATURE_TEMPLATES)
        component = random.choice(self.COMPONENTS)
        feature = random.choice(self.FEATURES)

        title = template["title"].format(component=component, feature=feature, capability=feature)
        description = template["description"].format(
            component=component,
            feature=feature,
            capability=feature,
            requirements="basic " + feature,
            cache_type="LRU",
            ttl="300s",
            validations="type, range, format"
        )

        # Find building-block tasks
        context_required = []
        if prior_tasks:
            for task in prior_tasks:
                if (task.metadata.get("component") == component or
                    task.metadata.get("feature") == feature):
                    context_required.append(task.id)

        return Task(
            id=self._generate_task_id(),
            session_id=session_id,
            task_type=TaskType.FEATURE_ADD,
            difficulty=difficulty,
            title=title,
            description=description,
            context_required=context_required[:3],
            expected_output=f"Implemented {feature} in {component}",
            evaluation_criteria={
                "feature_works": True,
                "backward_compatible": True,
                "documented": difficulty == Difficulty.HARD
            },
            metadata={
                "component": component,
                "feature": feature
            }
        )

    def generate_refactor_task(
        self,
        session_id: int,
        difficulty: Difficulty,
        prior_tasks: List[Task] = None
    ) -> Task:
        """Generate a refactoring task."""
        component = random.choice(self.COMPONENTS)

        refactor_types = [
            ("Extract method from", "Extract common logic into reusable method"),
            ("Split", "Split large class into focused components"),
            ("Simplify", "Reduce complexity using design patterns"),
            ("Optimize", "Improve performance without changing behavior")
        ]

        refactor_type, refactor_desc = random.choice(refactor_types)

        # Refactoring requires knowing prior structure
        context_required = []
        if prior_tasks:
            for task in prior_tasks:
                if task.metadata.get("component") == component:
                    context_required.append(task.id)

        return Task(
            id=self._generate_task_id(),
            session_id=session_id,
            task_type=TaskType.REFACTOR,
            difficulty=difficulty,
            title=f"{refactor_type} {component}",
            description=f"{refactor_desc} in {component}. Maintain all existing functionality.",
            context_required=context_required,  # Refactoring needs full context
            expected_output=f"Refactored {component} with improved structure",
            evaluation_criteria={
                "tests_pass": True,
                "complexity_reduced": True,
                "no_behavior_change": True
            },
            metadata={
                "component": component,
                "refactor_type": refactor_type
            }
        )

    def generate_debug_task(
        self,
        session_id: int,
        difficulty: Difficulty,
        prior_tasks: List[Task] = None
    ) -> Task:
        """Generate a debugging task (similar bugs should be recognized)."""
        component = random.choice(self.COMPONENTS)

        debug_scenarios = [
            ("intermittent failure", "Sometimes fails with timeout, need to identify trigger"),
            ("performance regression", "Response time increased 10x after recent changes"),
            ("data corruption", "Output data occasionally has incorrect values"),
            ("memory growth", "Memory usage increases over time, needs profiling")
        ]

        scenario, description = random.choice(debug_scenarios)

        # Debug tasks benefit from similar past debugging experiences
        context_required = []
        if prior_tasks:
            for task in prior_tasks:
                if task.task_type == TaskType.DEBUG or task.task_type == TaskType.BUG_FIX:
                    if task.metadata.get("component") == component:
                        context_required.append(task.id)

        return Task(
            id=self._generate_task_id(),
            session_id=session_id,
            task_type=TaskType.DEBUG,
            difficulty=difficulty,
            title=f"Debug {scenario} in {component}",
            description=f"{component}: {description}",
            context_required=context_required,
            expected_output=f"Identified and documented root cause of {scenario}",
            evaluation_criteria={
                "root_cause_found": True,
                "fix_identified": True,
                "steps_documented": True
            },
            metadata={
                "component": component,
                "debug_scenario": scenario
            }
        )

    def generate_session(
        self,
        session_id: int,
        num_tasks: int,
        prior_tasks: List[Task] = None,
        task_mix: Dict[TaskType, float] = None
    ) -> Session:
        """Generate a coding session with multiple tasks."""

        task_mix = task_mix or {
            TaskType.BUG_FIX: 0.3,
            TaskType.FEATURE_ADD: 0.3,
            TaskType.REFACTOR: 0.2,
            TaskType.DEBUG: 0.2
        }

        tasks = []
        all_prior = (prior_tasks or []) + tasks

        for _ in range(num_tasks):
            # Select task type based on mix
            task_type = random.choices(
                list(task_mix.keys()),
                weights=list(task_mix.values())
            )[0]

            difficulty = random.choice(list(Difficulty))

            if task_type == TaskType.BUG_FIX:
                task = self.generate_bug_fix_task(session_id, difficulty, all_prior)
            elif task_type == TaskType.FEATURE_ADD:
                task = self.generate_feature_task(session_id, difficulty, all_prior)
            elif task_type == TaskType.REFACTOR:
                task = self.generate_refactor_task(session_id, difficulty, all_prior)
            else:
                task = self.generate_debug_task(session_id, difficulty, all_prior)

            tasks.append(task)
            all_prior.append(task)

        # Generate session context summary
        components_touched = list(set(t.metadata.get("component", "") for t in tasks))
        session_context = f"Session {session_id}: Worked on {', '.join(components_touched[:3])}"

        return Session(
            id=session_id,
            tasks=tasks,
            session_context=session_context,
            duration_minutes=random.randint(30, 120)
        )

    def generate_dataset(
        self,
        name: str,
        num_sessions: int = 10,
        tasks_per_session: int = 5,
        dependency_density: float = 0.3
    ) -> TaskDataset:
        """Generate a complete multi-session task dataset."""

        sessions = []
        all_tasks = []

        for session_id in range(1, num_sessions + 1):
            # Prior tasks for dependency generation
            prior_tasks = all_tasks if random.random() < dependency_density else []

            session = self.generate_session(
                session_id=session_id,
                num_tasks=tasks_per_session,
                prior_tasks=prior_tasks
            )

            sessions.append(session)
            all_tasks.extend(session.tasks)

        # Calculate actual dependency density
        total_deps = sum(len(t.context_required) for t in all_tasks)
        max_deps = len(all_tasks) * 3
        actual_density = total_deps / max_deps if max_deps > 0 else 0

        return TaskDataset(
            name=name,
            sessions=sessions,
            total_tasks=len(all_tasks),
            dependency_density=actual_density
        )

    def save_dataset(self, dataset: TaskDataset, path: Path) -> None:
        """Save dataset to JSON file."""

        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, '__dict__'):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        data = serialize(dataset)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_dataset(path: Path) -> TaskDataset:
        """Load dataset from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct objects
        sessions = []
        for s in data['sessions']:
            tasks = []
            for t in s['tasks']:
                tasks.append(Task(
                    id=t['id'],
                    session_id=t['session_id'],
                    task_type=TaskType(t['task_type']),
                    difficulty=Difficulty(t['difficulty']),
                    title=t['title'],
                    description=t['description'],
                    context_required=t['context_required'],
                    expected_output=t['expected_output'],
                    evaluation_criteria=t['evaluation_criteria'],
                    metadata=t['metadata']
                ))
            sessions.append(Session(
                id=s['id'],
                tasks=tasks,
                session_context=s['session_context'],
                duration_minutes=s['duration_minutes']
            ))

        return TaskDataset(
            name=data['name'],
            sessions=sessions,
            total_tasks=data['total_tasks'],
            dependency_density=data['dependency_density'],
            created_at=datetime.fromisoformat(data['created_at'])
        )


def generate_experiment_datasets():
    """Generate datasets for all experiments."""
    generator = MultiSessionTaskGenerator(seed=42)
    base_path = Path("/home/marc/research-papers/memory-consolidation/experiments/tasks")

    # Main experiment dataset: 10 sessions, 5 tasks each
    main_dataset = generator.generate_dataset(
        name="main_experiment",
        num_sessions=10,
        tasks_per_session=5,
        dependency_density=0.4
    )
    generator.save_dataset(main_dataset, base_path / "main_experiment.json")
    print(f"Generated main experiment: {main_dataset.total_tasks} tasks across {len(main_dataset.sessions)} sessions")

    # High-dependency dataset (for testing cross-session coherence)
    high_dep_dataset = generator.generate_dataset(
        name="high_dependency",
        num_sessions=10,
        tasks_per_session=5,
        dependency_density=0.7
    )
    generator.save_dataset(high_dep_dataset, base_path / "high_dependency.json")
    print(f"Generated high dependency: {high_dep_dataset.total_tasks} tasks")

    # Noise dataset (with distractors for forgetting experiment)
    noise_dataset = generator.generate_dataset(
        name="noise_experiment",
        num_sessions=15,
        tasks_per_session=8,
        dependency_density=0.2  # Low dependency = more noise
    )
    generator.save_dataset(noise_dataset, base_path / "noise_experiment.json")
    print(f"Generated noise experiment: {noise_dataset.total_tasks} tasks")

    return main_dataset, high_dep_dataset, noise_dataset


if __name__ == "__main__":
    generate_experiment_datasets()
