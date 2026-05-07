"""
evaluator/disagreement.py

Ensemble Disagreement Detection for the Self-Improving Evaluator.

Disagreement detection serves two purposes:
  1. Training signal: samples where evaluator disagrees with human (or ensemble)
     are most informative for active learning queries.
  2. Routing: samples with high disagreement at inference time are flagged
     and sent to the human annotation queue (Kafka topic).

Three types of disagreement we track:
  A. Evaluator vs Human:   |f_θ(x,y) - h(x,y)| > δ
  B. Evaluator vs Ensemble: |f_θ - avg(f_θ_1, ..., f_θ_k)| > δ
  C. Confidence-based:      c_θ(x,y) < τ  (below confidence threshold)
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class DisagreementRecord:
    """Result of disagreement analysis on a single sample."""
    sample_id: str
    score: float
    confidence: float
    ensemble_scores: List[float]
    ensemble_mean: float
    ensemble_std: float
    disagreement_magnitude: float      # |f_θ - f_ensemble|
    human_score: Optional[float]       # None if not yet labeled
    evaluator_vs_human: Optional[float]  # |f_θ - h| if human_score available
    flag_low_confidence: bool
    flag_high_disagreement: bool
    flag_entropy: bool
    should_query_human: bool           # final routing decision


class EnsembleDisagreementDetector:
    """
    Detects high-disagreement samples using an ensemble of evaluator checkpoints.

    The ensemble is formed from the K most recent evaluator checkpoints saved
    during active learning cycles. When a new sample arrives, we run all K
    checkpoints and measure how much they disagree with the current evaluator.

    High disagreement → high uncertainty → valuable for active learning.
    """

    def __init__(
        self,
        disagreement_threshold: float = 0.20,     # δ: |f - f_ens| > δ triggers flag
        confidence_threshold: float = 0.75,        # τ: c < τ triggers flag
        entropy_threshold: float = 0.60,           # H > threshold triggers flag
        n_checkpoints: int = 3,                    # K: ensemble size
    ):
        self.disagreement_threshold = disagreement_threshold
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.n_checkpoints = n_checkpoints
        self.ensemble_checkpoints: List[torch.nn.Module] = []

    def register_checkpoint(self, model: torch.nn.Module) -> None:
        """Add a model checkpoint to the ensemble (keep only last K)."""
        self.ensemble_checkpoints.append(model)
        if len(self.ensemble_checkpoints) > self.n_checkpoints:
            self.ensemble_checkpoints.pop(0)
        logger.info(f"Ensemble size: {len(self.ensemble_checkpoints)}/{self.n_checkpoints}")

    @torch.no_grad()
    def get_ensemble_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
        """
        Run all ensemble checkpoints on a sample.

        Returns:
            mean:  mean ensemble score
            std:   standard deviation of scores (spread = disagreement)
            entropy: approximate score entropy (high = uncertain)
        """
        if not self.ensemble_checkpoints:
            return 0.5, 0.0, 0.0  # no ensemble yet

        scores = []
        for ckpt in self.ensemble_checkpoints:
            ckpt.eval()
            device = next(ckpt.parameters()).device
            score, _, _ = ckpt(
                input_ids.to(device),
                attention_mask.to(device) if attention_mask is not None else None,
            )
            scores.append(score.item())

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        # Approximate entropy: treat scores as Bernoulli probabilities
        p = np.clip(np.array(scores), 1e-8, 1 - 1e-8)
        entropy = float(-np.mean(p * np.log(p) + (1 - p) * np.log(1 - p)))

        return mean_score, std_score, entropy

    @torch.no_grad()
    def analyze(
        self,
        evaluator: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sample_id: str = "unknown",
        human_score: Optional[float] = None,
    ) -> DisagreementRecord:
        """
        Full disagreement analysis for one sample.

        Returns a DisagreementRecord with all flags set.
        """
        # Current evaluator prediction
        evaluator.eval()
        device = next(evaluator.parameters()).device
        score, confidence, _ = evaluator(
            input_ids.to(device),
            attention_mask.to(device) if attention_mask is not None else None,
        )
        score_val = score.item()
        conf_val = confidence.item()

        # Ensemble prediction
        ens_mean, ens_std, ens_entropy = self.get_ensemble_scores(input_ids, attention_mask)
        disagreement_mag = abs(score_val - ens_mean)

        # Optional: compare with human label if available
        ev_vs_human = abs(score_val - human_score) if human_score is not None else None

        # Set flags
        flag_low_conf = conf_val < self.confidence_threshold
        flag_high_dis = disagreement_mag > self.disagreement_threshold
        flag_entropy = ens_entropy > self.entropy_threshold

        # Route to human if ANY flag is raised
        should_query = flag_low_conf or flag_high_dis or flag_entropy

        return DisagreementRecord(
            sample_id=sample_id,
            score=score_val,
            confidence=conf_val,
            ensemble_scores=[],            # omit for memory efficiency
            ensemble_mean=ens_mean,
            ensemble_std=ens_std,
            disagreement_magnitude=disagreement_mag,
            human_score=human_score,
            evaluator_vs_human=ev_vs_human,
            flag_low_confidence=flag_low_conf,
            flag_high_disagreement=flag_high_dis,
            flag_entropy=flag_entropy,
            should_query_human=should_query,
        )

    def batch_analyze(
        self,
        evaluator: torch.nn.Module,
        batch_ids: torch.Tensor,
        batch_masks: Optional[torch.Tensor] = None,
        sample_ids: Optional[List[str]] = None,
        human_scores: Optional[List[float]] = None,
    ) -> List[DisagreementRecord]:
        """
        Analyze a batch of samples for disagreement.
        Returns list of DisagreementRecord, one per sample.
        """
        records = []
        n = len(batch_ids)

        for i in range(n):
            ids_i = batch_ids[i].unsqueeze(0)
            mask_i = batch_masks[i].unsqueeze(0) if batch_masks is not None else None
            sid = sample_ids[i] if sample_ids else str(i)
            hscore = human_scores[i] if human_scores else None

            record = self.analyze(evaluator, ids_i, mask_i, sid, hscore)
            records.append(record)

        return records

    def get_routing_stats(self, records: List[DisagreementRecord]) -> dict:
        """
        Summarize routing decisions across a batch.
        Returns dict for W&B logging.
        """
        n = len(records)
        if n == 0:
            return {}

        n_query = sum(r.should_query_human for r in records)
        n_low_conf = sum(r.flag_low_confidence for r in records)
        n_high_dis = sum(r.flag_high_disagreement for r in records)
        n_entropy = sum(r.flag_entropy for r in records)
        mean_dis = float(np.mean([r.disagreement_magnitude for r in records]))
        mean_conf = float(np.mean([r.confidence for r in records]))

        return {
            "routing/fraction_to_human": n_query / n,
            "routing/fraction_low_confidence": n_low_conf / n,
            "routing/fraction_high_disagreement": n_high_dis / n,
            "routing/fraction_high_entropy": n_entropy / n,
            "routing/mean_disagreement": mean_dis,
            "routing/mean_confidence": mean_conf,
            "routing/n_routed_to_human": n_query,
            "routing/n_total": n,
        }
