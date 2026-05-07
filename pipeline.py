"""
evaluator/train.py

Training entry point for Project 1: Self-Improving Evaluator.

Runs the full active learning training loop:
  1. Initialize evaluator model
  2. Score unlabeled pool (acquisition function)
  3. Select top-K samples → human annotation
  4. Retrain evaluator on labeled buffer
  5. Repeat for N cycles

Usage:
    python evaluator/train.py --config configs/evaluator_config.yaml
    python evaluator/train.py --config configs/evaluator_config.yaml --wandb-project my-project
"""

import os
import argparse
import yaml
import torch
import wandb
import mlflow
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from evaluator.model import EvaluatorModel
from evaluator.loss import UncertaintyAwareLoss, CurriculumLoss
from evaluator.active_learning import ActiveLearner, ActiveLearningCycle, SampleRecord
from evaluator.calibration import TemperatureCalibrator, log_calibration_metrics, compute_ece
from evaluator.disagreement import EnsembleDisagreementDetector


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class EvaluationDataset(Dataset):
    """Simple dataset for (prompt, response, score) triples."""

    def __init__(self, records: list, tokenizer, max_length: int = 2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = f"Prompt: {r['prompt']}\n\nResponse: {r['response']}"
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": torch.tensor([r["score"]], dtype=torch.float32),
            "difficulty": torch.tensor(r.get("difficulty", 0.5), dtype=torch.float32),
        }


class EvaluatorTrainer:
    """Manages training of the EvaluatorModel with curriculum learning."""

    def __init__(self, model, tokenizer, config: dict, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.base_loss = UncertaintyAwareLoss(
            lambda_conf=config["training"]["lambda_conf"]
        )
        self.criterion = CurriculumLoss(
            base_loss=self.base_loss,
            gamma_min=0.2,
            gamma_max=1.0,
            warmup_epochs=config["training"]["curriculum_warmup_epochs"],
        )

        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.global_step = 0
        self.current_epoch = 0

    def train_epoch(self, dataloader: DataLoader, total_epochs: int) -> dict:
        """Train for one epoch. Returns dict of aggregated metrics."""
        self.model.train()
        self.model.to(self.device)

        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target"].to(self.device)
            difficulty = batch["difficulty"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            score, confidence, _ = self.model(input_ids, attention_mask)

            # Curriculum-filtered uncertainty-aware loss
            loss, components = self.criterion(
                score, confidence, targets, difficulty,
                current_epoch=self.current_epoch,
                total_epochs=total_epochs,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if self.global_step % 50 == 0:
                wandb.log({**components, "step": self.global_step})

        self.current_epoch += 1
        return {"epoch_loss": total_loss / max(n_batches, 1)}

    def train(self, labeled_buffer: list, n_epochs: int = 3) -> dict:
        """Train on the current labeled buffer for n_epochs."""
        dataset = EvaluationDataset(
            [{"prompt": s.prompt, "response": s.response,
              "score": s.human_label, "difficulty": s.difficulty}
             for s in labeled_buffer if s.human_label is not None],
            self.tokenizer,
            max_length=self.config["training"]["max_seq_length"],
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
        )

        metrics = {}
        for epoch in range(n_epochs):
            epoch_metrics = self.train_epoch(loader, total_epochs=n_epochs)
            metrics.update(epoch_metrics)

        return metrics


def evaluate_model(model, eval_loader, device: str) -> dict:
    """Compute Kendall Tau, Spearman rho, MSE on eval set."""
    from scipy.stats import kendalltau, spearmanr

    model.eval()
    preds, targets, confs = [], [], []

    with torch.no_grad():
        for batch in eval_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            score, conf, _ = model(ids, mask)
            preds.extend(score.squeeze(-1).cpu().numpy().tolist())
            confs.extend(conf.squeeze(-1).cpu().numpy().tolist())
            targets.extend(batch["target"].squeeze(-1).numpy().tolist())

    import numpy as np
    preds = np.array(preds)
    targets = np.array(targets)
    confs = np.array(confs)

    kt, _ = kendalltau(preds, targets)
    sr, _ = spearmanr(preds, targets)
    mse = float(np.mean((preds - targets) ** 2))
    correct = (np.abs(preds - targets) < 0.1).astype(float)
    ece = compute_ece(confs, correct)

    return {
        "eval/kendall_tau": float(kt),
        "eval/spearman_rho": float(sr),
        "eval/mse": mse,
        "eval/ece": ece,
        "eval/mean_confidence": float(np.mean(confs)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Self-Improving Evaluator (Project 1)")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluator_config.yaml")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Seeding
    torch.manual_seed(args.seed)

    # W&B setup
    wandb_project = args.wandb_project or config["logging"]["wandb_project"]
    wandb.init(
        project=wandb_project,
        entity=config["logging"]["wandb_entity"],
        config=config,
        tags=["evaluator", "active-learning", "uncertainty"],
    )

    # MLflow setup
    mlflow.set_tracking_uri(config["logging"].get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_experiment("evaluator-project1")

    # Output directory
    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Project 1: Self-Improving Evaluator ===")
    logger.info(f"Config: {config}")

    # Model
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    evaluator = EvaluatorModel(
        base_model_name=config["model"]["base_model"],
        hidden_size=config["model"]["hidden_size"],
        vocab_size=config["model"]["vocab_size"],
        freeze_backbone=config["model"]["freeze_backbone"],
        use_lora=config["model"]["use_lora"],
        lora_r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
    )

    # Trainer
    trainer = EvaluatorTrainer(evaluator, tokenizer, config, device=args.device)

    # Active learner
    al_config = config["active_learning"]
    learner = ActiveLearner(
        alpha=al_config["uncertainty_weight"],
        beta=al_config["disagreement_weight"],
        gamma=al_config["novelty_weight"],
        disagreement_threshold=al_config["disagreement_threshold"],
        confidence_threshold=al_config["confidence_threshold"],
    )

    # Disagreement detector (for ensemble)
    detector = EnsembleDisagreementDetector(
        disagreement_threshold=al_config["disagreement_threshold"],
        confidence_threshold=al_config["confidence_threshold"],
        n_checkpoints=al_config["ensemble_size"],
    )

    # AL cycle
    al_cycle = ActiveLearningCycle(
        evaluator=evaluator,
        learner=learner,
        trainer=trainer,
        n_cycles=10,
        cycle_budget=al_config["cycle_budget"],
    )

    # In real usage: load actual data from config["data"] paths
    # Here we simulate with placeholder human annotator oracle
    def human_annotator_oracle(sample: SampleRecord) -> float:
        """Simulate human annotation. Replace with real annotation API."""
        return float(torch.rand(1).item())

    # Run active learning
    with mlflow.start_run():
        mlflow.log_params(config["training"])
        mlflow.log_params(config["active_learning"])

        summary = al_cycle.run(
            pool=[],  # load real pool from config["data"]["unlabeled_pool_path"]
            human_annotator=human_annotator_oracle,
        )

        # Save final model
        ckpt_path = output_dir / "best_checkpoint.pt"
        evaluator.save(str(ckpt_path))
        mlflow.log_artifact(str(ckpt_path))
        logger.info(f"Training complete. Model saved to {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
