"""
evaluator/calibration.py

Confidence Calibration for the Evaluator Model.

A well-calibrated model: when it says confidence=0.8, it should be right ~80% of the time.
An uncalibrated model might say confidence=0.9 but only be right 60% of the time.

Methods:
  1. Temperature Scaling: scale logits by T, then apply sigmoid.
     T > 1 softens confidence (makes model less extreme).
     T < 1 sharpens confidence (makes model more extreme).

  2. Expected Calibration Error (ECE): primary calibration metric.
     Bins predictions by confidence, measures avg |accuracy - confidence| per bin.

  3. Brier Score: proper scoring rule, measures mean squared error of probabilities.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from loguru import logger


class TemperatureCalibrator:
    """
    Post-hoc temperature scaling calibration.

    After training the evaluator, we fit a single temperature parameter T
    on a held-out calibration set to minimize negative log-likelihood.

    This does NOT change the model predictions — only rescales the confidence.

    Usage:
        calibrator = TemperatureCalibrator(evaluator)
        calibrator.fit(cal_loader)
        calibrator.apply(evaluator)  # updates evaluator.log_temperature in place
    """

    def __init__(self, evaluator, lr: float = 0.01, max_iter: int = 50):
        self.evaluator = evaluator
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, calibration_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """
        Fit temperature T on calibration data.

        Args:
            calibration_data: List of (confidence_logits, binary_correct) tuples
                confidence_logits: raw pre-sigmoid confidence logits [B]
                binary_correct:    1 if model was correct, 0 otherwise [B]

        Returns:
            Optimal temperature T
        """
        # Collect all logits and labels
        all_logits = []
        all_labels = []
        for logits, labels in calibration_data:
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        logits_tensor = torch.cat(all_logits)
        labels_tensor = torch.cat(all_labels).float()

        # Optimize temperature to minimize NLL on calibration set
        log_T = nn.Parameter(torch.zeros(1))
        optimizer = optim.LBFGS([log_T], lr=self.lr, max_iter=self.max_iter)

        def eval_nll():
            optimizer.zero_grad()
            T = log_T.exp().clamp(0.1, 10.0)
            scaled_probs = torch.sigmoid(logits_tensor / T)
            nll = nn.functional.binary_cross_entropy(
                scaled_probs, labels_tensor, reduction="mean"
            )
            nll.backward()
            return nll

        optimizer.step(eval_nll)

        optimal_T = log_T.exp().item()
        logger.info(f"Temperature calibration: T = {optimal_T:.4f}")

        # Update evaluator's temperature in place
        with torch.no_grad():
            self.evaluator.log_temperature.copy_(log_T.detach())

        return optimal_T


def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures average absolute difference between confidence and accuracy
    across confidence bins.

    ECE = Σ_{b=1}^{B} (|B_b| / n) · |acc(B_b) - conf(B_b)|

    Args:
        confidences: [N] array of confidence scores ∈ [0,1]
        correct:     [N] binary array: 1 if prediction correct, 0 otherwise
        n_bins:      Number of confidence bins

    Returns:
        ECE score ∈ [0,1] (lower = better calibrated)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        n_in_bin = in_bin.sum()

        if n_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += (n_in_bin / n) * abs(accuracy_in_bin - confidence_in_bin)

    return float(ece)


def compute_brier_score(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Brier Score — proper scoring rule for probabilistic predictions.

    BS = (1/N) · Σ (f_θ(x,y) - h(x,y))²

    Equivalent to MSE for binary targets.
    Range: [0, 1], lower = better.

    Args:
        predictions: [N] predicted scores ∈ [0,1]
        targets:     [N] true labels ∈ {0,1} or ∈ [0,1]

    Returns:
        Brier score ∈ [0,1]
    """
    return float(np.mean((predictions - targets) ** 2))


def compute_calibration_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data for plotting.

    Returns:
        bin_centers:  confidence bin centers
        accuracies:   mean accuracy in each bin
        bin_counts:   number of samples in each bin
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin = (confidences >= lower) & (confidences < upper)
        bin_counts[i] = in_bin.sum()
        if bin_counts[i] > 0:
            accuracies[i] = correct[in_bin].mean()

    return bin_centers, accuracies, bin_counts


def log_calibration_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    step: int,
    prefix: str = "eval",
) -> dict:
    """
    Compute and log all calibration metrics at once.

    Returns dict for W&B logging.
    """
    # Binary correct: prediction within 0.1 of target
    correct = (np.abs(predictions - targets) < 0.1).astype(float)

    ece = compute_ece(confidences, correct)
    brier = compute_brier_score(predictions, targets)
    mean_conf = float(np.mean(confidences))
    mean_acc = float(np.mean(correct))

    metrics = {
        f"{prefix}/ece": ece,
        f"{prefix}/brier_score": brier,
        f"{prefix}/mean_confidence": mean_conf,
        f"{prefix}/mean_accuracy": mean_acc,
        f"{prefix}/calibration_gap": abs(mean_conf - mean_acc),
        "step": step,
    }

    logger.info(
        f"[{prefix}] ECE={ece:.4f} | Brier={brier:.4f} | "
        f"Conf={mean_conf:.3f} | Acc={mean_acc:.3f}"
    )

    return metrics
