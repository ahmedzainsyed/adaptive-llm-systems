"""
Project 3: Agentic Benchmark Generator for Real-World LLM Evaluation

Paper: "Dynamic Benchmarking for Large Language Models
        via Agentic Task Generation and Failure-Driven Adaptation"

Core claim:
  Static benchmarks (MMLU, BIG-Bench, HellaSwag) systematically fail
  to capture real-world failure modes because D_static ≠ D_real.
  An agentic system that generates tasks conditioned on observed failures
  closes this gap and enables proactive discovery of model weaknesses.

Key contributions:
  1. Multi-agent benchmark architecture (4 specialized agents)
  2. Failure-conditioned task synthesis with adversarial diversity
  3. Adaptive difficulty controller maintaining ~50% failure rate
  4. Real-world correlation validation: ρ=0.73 vs ρ=0.41 static

Integration:
  - Feeds tasks to base LLM (evaluation subject)
  - Uses Project 1 EvaluatorModel to score outputs + detect failures
  - Detected failures trigger Project 2 RLHF for model improvement
  - Creates the proactive, continuously evolving closed-loop system
"""

from benchmark.agents import (
    TaskGeneratorAgent,
    FailureAnalyzerAgent,
    DifficultyController,
    TaskRefinerAgent,
)
from benchmark.evolution_loop import AgenticBenchmarkSystem, Task

__all__ = [
    "TaskGeneratorAgent",
    "FailureAnalyzerAgent",
    "DifficultyController",
    "TaskRefinerAgent",
    "AgenticBenchmarkSystem",
    "Task",
]
