"""
Project 1: Self-Improving Evaluator for LLMs
via Disagreement-Driven Active Learning

Paper: "Adaptive Evaluation for Large Language Models
        via Disagreement-Driven Active Learning"

Core contributions:
  1. Multi-head evaluator: score + confidence + reasoning heads
  2. Uncertainty-aware loss: L_unc = (1 - c_θ) · (f_θ - h)²
  3. Disagreement-driven active learning acquisition function
  4. Temperature-calibrated confidence with ECE monitoring
  5. Curriculum learning schedule (easy → hard)
"""

from evaluator.model import EvaluatorModel
from evaluator.loss import UncertaintyAwareLoss
from evaluator.active_learning import ActiveLearner, ActiveLearningCycle
from evaluator.calibration import TemperatureCalibrator, compute_ece, compute_brier_score
from evaluator.disagreement import EnsembleDisagreementDetector

__all__ = [
    "EvaluatorModel",
    "UncertaintyAwareLoss",
    "ActiveLearner",
    "ActiveLearningCycle",
    "TemperatureCalibrator",
    "compute_ece",
    "compute_brier_score",
    "EnsembleDisagreementDetector",
]
