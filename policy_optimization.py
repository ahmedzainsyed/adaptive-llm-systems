"""
rlhf/reward_model.py

Reward Model for Data-Efficient RLHF (Project 2).

Architecture:
  - Shared backbone with EvaluatorModel (optional weight sharing)
  - Single scalar reward head: r_ϕ(x, y) ∈ ℝ
  - Trained via pairwise Bradley-Terry loss:
    L = -log σ(r(x, y_good) - r(x, y_bad))

Integration with Project 1:
  The EvaluatorModel's confidence score gates which (y_good, y_bad) pairs
  enter training. Only high-confidence pairs are accepted.
  This is the primary noise-reduction mechanism that enables 1-5% label regime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from loguru import logger
from typing import Optional, Tuple


class RewardModel(nn.Module):
    """
    Pairwise reward model trained on (prompt, chosen, rejected) triples.

    Produces a scalar reward r_ϕ(x, y) for any (prompt, response) pair.
    Higher reward → better response.

    During training: we pass (x, y_good) and (x, y_bad) through the same
    model and optimize the Bradley-Terry pairwise loss.

    During RLHF policy optimization (PPO): the reward model is frozen and
    used to score generated responses.
    """

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3-8B",
        hidden_size: int = 4096,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
    ):
        super().__init__()

        logger.info(f"Initializing RewardModel from {base_model_name}")

        config = AutoConfig.from_pretrained(base_model_name)
        self.backbone = AutoModel.from_pretrained(
            base_model_name,
            config=config,
            torch_dtype=torch.float16,
            load_in_4bit=load_in_4bit,
        )

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

        # Reward head: hidden state → scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize reward head with small weights for training stability."""
        for layer in self.reward_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar reward for a batch of (prompt + response) inputs.

        Args:
            input_ids:      [B, L] tokenized prompt + response
            attention_mask: [B, L] padding mask

        Returns:
            reward: [B] scalar reward values
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use the last token's hidden state as the reward representation
        # (For decoder-only models, last non-padding token contains full context)
        if attention_mask is not None:
            # Find last non-padding position per sample
            last_pos = attention_mask.sum(dim=1) - 1          # [B]
            hidden = outputs.last_hidden_state                  # [B, L, H]
            h = hidden[torch.arange(len(last_pos)), last_pos]  # [B, H]
        else:
            h = outputs.last_hidden_state[:, -1, :]            # [B, H] — last token

        reward = self.reward_head(h.float()).squeeze(-1)        # [B]
        return reward

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.get_reward(input_ids, attention_mask)

    def save(self, path: str) -> None:
        torch.save({"model_state_dict": self.state_dict()}, path)
        logger.info(f"RewardModel saved to {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "RewardModel":
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"RewardModel loaded from {path}")
        return model


def pairwise_reward_loss(
    reward_model: RewardModel,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: Optional[torch.Tensor] = None,
    rejected_mask: Optional[torch.Tensor] = None,
    confidence_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Bradley-Terry pairwise ranking loss.

    L = -log σ(r(x, y_chosen) - r(x, y_rejected))

    Optional: weight each pair by confidence from Project 1 evaluator.
    This is the core of our novel DPO-conf variant.

    Args:
        reward_model:       RewardModel instance
        chosen_ids:         [B, L] tokenized chosen (better) responses
        rejected_ids:       [B, L] tokenized rejected (worse) responses
        chosen_mask:        [B, L] attention mask for chosen
        rejected_mask:      [B, L] attention mask for rejected
        confidence_weights: [B] per-pair confidence from evaluator (for DPO-conf)
                            If None, standard unweighted loss is applied.

    Returns:
        loss:       scalar training loss
        metrics:    dict of per-step metrics for W&B logging
    """
    r_chosen = reward_model(chosen_ids, chosen_mask)      # [B]
    r_rejected = reward_model(rejected_ids, rejected_mask)  # [B]

    # Reward margin (higher = clearer preference)
    reward_margin = r_chosen - r_rejected                  # [B]

    # Bradley-Terry log-likelihood
    log_likelihood = F.logsigmoid(reward_margin)           # [B]

    if confidence_weights is not None:
        # DPO-conf: weight loss by evaluator confidence
        # High confidence pairs → full gradient signal
        # Low confidence pairs → attenuated gradient
        weights = confidence_weights.clamp(0.1, 1.0)       # [B]
        loss = -(weights * log_likelihood).mean()
    else:
        loss = -log_likelihood.mean()

    # Accuracy: fraction where chosen > rejected
    accuracy = (reward_margin > 0).float().mean().item()

    metrics = {
        "rm/loss": loss.item(),
        "rm/mean_chosen_reward": r_chosen.mean().item(),
        "rm/mean_rejected_reward": r_rejected.mean().item(),
        "rm/mean_margin": reward_margin.mean().item(),
        "rm/accuracy": accuracy,
    }
    if confidence_weights is not None:
        metrics["rm/mean_confidence_weight"] = confidence_weights.mean().item()

    return loss, metrics
