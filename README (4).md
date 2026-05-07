"""
evaluator/loss.py

Uncertainty-Aware Training Loss for the Self-Improving Evaluator.

Core idea:
  Standard MSE gives equal weight to all samples.
  Our loss DOWN-WEIGHTS samples where the evaluator is confident it's right,
  and UP-WEIGHTS samples where the evaluator is uncertain — forcing learning
  on the hard/uncertain examples.

  L_unc(θ) = E[(1 - c_θ(x,y)) · (f_θ(x,y) - h(x,y))²]
            + λ · L_conf_reg(θ)

  where:
    c_θ(x,y)  = evaluator confidence score ∈ [0,1]
    f_θ(x,y)  = evaluator score prediction ∈ [0,1]
    h(x,y)    = human ground-truth score ∈ [0,1]
    λ         = confidence regularization coefficient

  Confidence Regularization:
    L_conf_reg = -E[log(c_θ(x,y))]
    Prevents the model from always outputting c=0 (which would zero the loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class UncertaintyAwareLoss(nn.Module):
    """
    Uncertainty-weighted MSE loss with confidence regularization.

    Args:
        lambda_conf (float): Weight of confidence regularization term.
            Higher values force the model to be more confident on easy examples.
        eps (float): Numerical stability for log(confidence).
    """

    def __init__(self, lambda_conf: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.lambda_conf = lambda_conf
        self.eps = eps

    def forward(
        self,
        score: torch.Tensor,
        confidence: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            score:      [B, 1] predicted quality scores from evaluator
            confidence: [B, 1] calibrated confidence scores from evaluator
            target:     [B, 1] human ground-truth labels ∈ [0, 1]
            reduction:  "mean" | "sum" | "none"

        Returns:
            loss:        scalar total loss
            components:  dict with individual loss terms for logging
        """
        # ── Term 1: Uncertainty-weighted MSE ──────────────────────────
        # (1 - c) weights the squared error:
        #   high confidence (c→1): weight→0, small gradient (model is sure)
        #   low confidence (c→0):  weight→1, full gradient (model is uncertain)
        uncertainty_weight = 1.0 - confidence.detach()  # detach to avoid double gradient
        squared_error = (score - target) ** 2
        unc_mse = (uncertainty_weight * squared_error)

        if reduction == "mean":
            unc_mse = unc_mse.mean()
        elif reduction == "sum":
            unc_mse = unc_mse.sum()

        # ── Term 2: Confidence Regularization ─────────────────────────
        # Penalize always-zero confidence (which would trivially zero the loss).
        # Encourages the model to be confident on examples it should know.
        conf_reg = -torch.log(confidence + self.eps).mean()

        # ── Total Loss ─────────────────────────────────────────────────
        total_loss = unc_mse + self.lambda_conf * conf_reg

        components = {
            "loss_unc_mse": unc_mse.item(),
            "loss_conf_reg": conf_reg.item(),
            "loss_total": total_loss.item(),
            "mean_confidence": confidence.mean().item(),
            "mean_uncertainty": uncertainty_weight.mean().item(),
            "mean_squared_error": squared_error.mean().item(),
        }

        return total_loss, components


class CurriculumLoss(nn.Module):
    """
    Curriculum Learning wrapper for UncertaintyAwareLoss.

    At epoch t, only trains on samples with difficulty ≤ γ(t):
        γ(t) = γ_min + (γ_max - γ_min) · min(t / T_warmup, 1.0)

    Difficulty of each sample:
        difficulty(x,y) = α·uncertainty + β·disagreement + γ·novelty

    This starts training on clear-cut evaluations and gradually
    introduces harder, more ambiguous examples.
    """

    def __init__(
        self,
        base_loss: UncertaintyAwareLoss,
        gamma_min: float = 0.2,
        gamma_max: float = 1.0,
        warmup_epochs: int = 2,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.warmup_epochs = warmup_epochs

    def get_difficulty_threshold(self, current_epoch: int, total_epochs: int) -> float:
        """
        Compute current difficulty threshold γ(t).

        γ starts at γ_min and linearly increases to γ_max over warmup_epochs.
        After warmup, all samples are included.
        """
        progress = min(current_epoch / max(self.warmup_epochs, 1), 1.0)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * progress
        return gamma

    def forward(
        self,
        score: torch.Tensor,
        confidence: torch.Tensor,
        target: torch.Tensor,
        difficulty: torch.Tensor,
        current_epoch: int,
        total_epochs: int,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            score:          [B, 1] predicted scores
            confidence:     [B, 1] confidence scores
            target:         [B, 1] human labels
            difficulty:     [B]    per-sample difficulty in [0, 1]
            current_epoch:  current training epoch
            total_epochs:   total number of epochs

        Returns:
            loss:       curriculum-filtered total loss
            components: dict of loss components + curriculum stats
        """
        gamma = self.get_difficulty_threshold(current_epoch, total_epochs)

        # Create curriculum mask: include only samples with difficulty ≤ gamma
        curriculum_mask = (difficulty <= gamma)  # [B]
        n_included = curriculum_mask.sum().item()

        if n_included == 0:
            # Edge case: no samples pass filter — relax to all samples
            curriculum_mask = torch.ones_like(difficulty, dtype=torch.bool)
            n_included = len(difficulty)

        # Apply curriculum mask
        score_curr = score[curriculum_mask]
        conf_curr = confidence[curriculum_mask]
        target_curr = target[curriculum_mask]

        loss, base_components = self.base_loss(score_curr, conf_curr, target_curr)

        components = {
            **base_components,
            "curriculum_gamma": gamma,
            "curriculum_n_included": n_included,
            "curriculum_fraction": n_included / len(difficulty),
        }

        return loss, components


def compute_difficulty_score(
    uncertainty: float,
    disagreement: float,
    novelty: float,
    alpha: float = 0.4,
    beta: float = 0.35,
    gamma: float = 0.25,
) -> float:
    """
    Composite difficulty score used for both curriculum learning and active learning.

    S(x,y) = α·uncertainty + β·disagreement + γ·novelty

    Args:
        uncertainty:  1 - c_θ(x,y)    (low confidence = high uncertainty)
        disagreement: |f_θ - f_ensemble| (evaluator vs. ensemble mismatch)
        novelty:      1 - max_z cos(h(x,y), h(z))  (distance to known samples)
        alpha, beta, gamma: component weights (must sum to 1.0)

    Returns:
        difficulty score ∈ [0, 1]
    """
    assert abs(alpha + beta + gamma - 1.0) < 1e-4, "Weights must sum to 1.0"
    score = alpha * uncertainty + beta * disagreement + gamma * novelty
    return float(max(0.0, min(1.0, score)))  # clamp to [0, 1]
