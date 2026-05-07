"""
evaluator/active_learning.py

Active Learning Strategy for the Self-Improving Evaluator.

Key insight: Most samples are easy for the evaluator. 
Querying humans on ALL samples is expensive and mostly redundant.
We select only the top-K most informative samples per cycle.

Acquisition function:
    S(x,y) = α·Uncertainty(x,y) + β·Disagreement(x,y) + γ·Novelty(x,y)

where:
    Uncertainty(x,y)   = 1 - c_θ(x,y)                    low evaluator confidence
    Disagreement(x,y)  = |f_θ - f_ensemble|               evaluator vs. ensemble of checkpoints
    Novelty(x,y)       = 1 - max_{z∈labeled} cos(h, h_z)  embedding distance to labeled corpus

Sample selection: top-K by S → human annotation → add to training buffer → retrain.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from loguru import logger
import wandb


@dataclass
class SampleRecord:
    """A single sample in the pool with all associated metadata."""
    prompt: str
    response: str
    input_ids: Optional[torch.Tensor] = None
    embedding: Optional[np.ndarray] = None
    score_pred: float = 0.0
    confidence: float = 1.0
    ensemble_score: float = 0.0
    acquisition_score: float = 0.0
    human_label: Optional[float] = None
    difficulty: float = 0.0
    is_labeled: bool = False


class EmbeddingIndex:
    """
    Fast approximate nearest-neighbor index for novelty computation.
    Wraps FAISS for efficient cosine similarity over labeled corpus embeddings.
    """

    def __init__(self, dim: int = 768):
        try:
            import faiss
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after L2 norm)
            self.dim = dim
            self.n_vectors = 0
        except ImportError:
            logger.warning("FAISS not found. Using numpy fallback for novelty (slower).")
            self.index = None
            self.embeddings_np: List[np.ndarray] = []
            self.dim = dim
            self.n_vectors = 0

    def add(self, embeddings: np.ndarray) -> None:
        """Add normalized embeddings to the index."""
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        if self.index is not None:
            self.index.add(embeddings.astype(np.float32))
        else:
            self.embeddings_np.extend(embeddings)
        self.n_vectors += len(embeddings)

    def max_similarity(self, query: np.ndarray) -> float:
        """Return max cosine similarity of query to any indexed embedding."""
        if self.n_vectors == 0:
            return 0.0
        query = query / (np.linalg.norm(query) + 1e-8)
        if self.index is not None:
            D, _ = self.index.search(query.reshape(1, -1).astype(np.float32), k=1)
            return float(D[0, 0])
        else:
            sims = [float(np.dot(query, e)) for e in self.embeddings_np]
            return max(sims)

    def novelty(self, query: np.ndarray) -> float:
        """Novelty = 1 - max_cosine_similarity. Range [0, 1]."""
        return 1.0 - self.max_similarity(query)


class ActiveLearner:
    """
    Selects the most informative unlabeled samples for human annotation.

    Algorithm per cycle:
      1. Score each unlabeled sample with acquisition function S
      2. Select top-K samples
      3. Send to human annotator (or oracle in experiments)
      4. Add labeled samples to training buffer
      5. Trigger evaluator retraining
    """

    def __init__(
        self,
        alpha: float = 0.40,   # weight for uncertainty
        beta: float = 0.35,    # weight for disagreement
        gamma: float = 0.25,   # weight for novelty
        embedding_dim: int = 768,
        disagreement_threshold: float = 0.20,
        confidence_threshold: float = 0.75,
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-4, "Weights must sum to 1.0"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.disagreement_threshold = disagreement_threshold
        self.confidence_threshold = confidence_threshold
        self.embedding_index = EmbeddingIndex(dim=embedding_dim)
        self._cycle = 0

    def compute_acquisition_score(
        self,
        sample: SampleRecord,
    ) -> float:
        """
        Composite acquisition score:
            S = α·uncertainty + β·disagreement + γ·novelty

        Higher score = more valuable to annotate.
        """
        uncertainty = 1.0 - sample.confidence
        disagreement = abs(sample.score_pred - sample.ensemble_score)
        novelty = (
            self.embedding_index.novelty(sample.embedding)
            if sample.embedding is not None
            else 0.5
        )
        score = (
            self.alpha * uncertainty
            + self.beta * disagreement
            + self.gamma * novelty
        )
        return float(np.clip(score, 0.0, 1.0))

    def score_pool(
        self,
        pool: List[SampleRecord],
    ) -> List[SampleRecord]:
        """Score all unlabeled samples in the pool by acquisition function."""
        for sample in pool:
            if not sample.is_labeled:
                sample.acquisition_score = self.compute_acquisition_score(sample)
        return pool

    def select_top_k(
        self,
        pool: List[SampleRecord],
        k: int,
    ) -> Tuple[List[SampleRecord], List[SampleRecord]]:
        """
        Select top-K unlabeled samples by acquisition score.

        Returns:
            selected:   List of k samples to annotate
            remaining:  Remaining pool (unlabeled, not selected)
        """
        unlabeled = [s for s in pool if not s.is_labeled]
        unlabeled.sort(key=lambda s: s.acquisition_score, reverse=True)

        selected = unlabeled[:k]
        remaining = unlabeled[k:]

        logger.info(
            f"[Active Learning] Selected {len(selected)} samples. "
            f"Pool remaining: {len(remaining)}. "
            f"Top score: {selected[0].acquisition_score:.4f}"
        )
        return selected, remaining

    def flag_disagreement(self, sample: SampleRecord) -> bool:
        """
        Binary flag: True if sample should be routed to human.
        Used in inference-time disagreement detection.

        Criteria:
          1. Low confidence (c < threshold)
          2. High disagreement with ensemble (|f - f_ens| > delta)
        """
        low_confidence = sample.confidence < self.confidence_threshold
        high_disagreement = abs(sample.score_pred - sample.ensemble_score) > self.disagreement_threshold
        return low_confidence or high_disagreement

    def add_labeled_to_index(self, samples: List[SampleRecord]) -> None:
        """After annotation, add embeddings to the novelty index."""
        embeddings = [s.embedding for s in samples if s.embedding is not None]
        if embeddings:
            self.embedding_index.add(np.stack(embeddings))

    def run_cycle(
        self,
        pool: List[SampleRecord],
        labeled_buffer: List[SampleRecord],
        human_annotator: Callable[[SampleRecord], float],
        cycle_budget: int = 500,
        cycle_index: int = 0,
    ) -> Tuple[List[SampleRecord], dict]:
        """
        Execute one active learning cycle.

        Args:
            pool:             Unlabeled sample pool
            labeled_buffer:   Previously labeled samples
            human_annotator:  Callable: sample → human_label (float)
            cycle_budget:     How many samples to annotate this cycle
            cycle_index:      Current cycle number (for logging)

        Returns:
            updated_labeled_buffer: labeled_buffer + new annotations
            metrics:                dict of cycle metrics for W&B logging
        """
        self._cycle = cycle_index
        logger.info(f"=== Active Learning Cycle {cycle_index} ===")

        # Step 1: Score pool
        pool = self.score_pool(pool)

        # Step 2: Select top-K
        selected, _ = self.select_top_k(pool, k=cycle_budget)

        # Step 3: Human annotation
        n_annotated = 0
        for sample in selected:
            label = human_annotator(sample)
            sample.human_label = label
            sample.is_labeled = True
            labeled_buffer.append(sample)
            n_annotated += 1

        # Step 4: Update novelty index
        self.add_labeled_to_index(selected)

        # Compute cycle metrics
        scores = [s.acquisition_score for s in selected]
        confidences = [s.confidence for s in selected]
        uncertainties = [1.0 - c for c in confidences]

        metrics = {
            "al_cycle": cycle_index,
            "al_n_annotated": n_annotated,
            "al_total_labeled": len(labeled_buffer),
            "al_pool_size": len(pool),
            "al_mean_acquisition": float(np.mean(scores)),
            "al_mean_uncertainty": float(np.mean(uncertainties)),
            "al_mean_confidence": float(np.mean(confidences)),
        }

        logger.info(f"Cycle {cycle_index} done. Total labeled: {len(labeled_buffer)}")
        return labeled_buffer, metrics


class ActiveLearningCycle:
    """
    High-level orchestrator for repeated active learning cycles.
    Wraps ActiveLearner + EvaluatorModel training in a loop.
    """

    def __init__(
        self,
        evaluator,
        learner: ActiveLearner,
        trainer,
        n_cycles: int = 10,
        cycle_budget: int = 500,
    ):
        self.evaluator = evaluator
        self.learner = learner
        self.trainer = trainer
        self.n_cycles = n_cycles
        self.cycle_budget = cycle_budget
        self.labeled_buffer: List[SampleRecord] = []
        self.all_metrics: List[dict] = []

    def run(
        self,
        pool: List[SampleRecord],
        human_annotator: Callable[[SampleRecord], float],
        seed_labeled: Optional[List[SampleRecord]] = None,
    ) -> dict:
        """
        Run n_cycles of active learning.

        Args:
            pool:             Full unlabeled pool
            human_annotator:  Oracle labeling function
            seed_labeled:     Optional initial labeled set (small seed)

        Returns:
            Summary metrics across all cycles.
        """
        if seed_labeled:
            self.labeled_buffer = list(seed_labeled)
            self.learner.add_labeled_to_index(seed_labeled)

        for cycle_i in range(self.n_cycles):
            # Score pool with current evaluator
            pool = self._score_pool_with_evaluator(pool)

            # Run AL cycle
            self.labeled_buffer, metrics = self.learner.run_cycle(
                pool=pool,
                labeled_buffer=self.labeled_buffer,
                human_annotator=human_annotator,
                cycle_budget=self.cycle_budget,
                cycle_index=cycle_i,
            )

            # Retrain evaluator on updated labeled buffer
            train_metrics = self.trainer.train(self.labeled_buffer)
            metrics.update(train_metrics)
            self.all_metrics.append(metrics)

            # Log to W&B
            wandb.log(metrics)
            logger.info(f"Cycle {cycle_i} complete. Metrics: {metrics}")

        return {
            "total_labeled": len(self.labeled_buffer),
            "n_cycles": self.n_cycles,
            "final_metrics": self.all_metrics[-1] if self.all_metrics else {},
        }

    def _score_pool_with_evaluator(
        self, pool: List[SampleRecord]
    ) -> List[SampleRecord]:
        """
        Run evaluator inference on all unlabeled pool samples.
        Updates sample.score_pred, sample.confidence, sample.ensemble_score.
        """
        self.evaluator.eval()
        device = next(self.evaluator.parameters()).device

        for sample in pool:
            if sample.is_labeled or sample.input_ids is None:
                continue
            with torch.no_grad():
                ids = sample.input_ids.unsqueeze(0).to(device)
                score, confidence, _ = self.evaluator(ids)
                sample.score_pred = score.item()
                sample.confidence = confidence.item()

        return pool
