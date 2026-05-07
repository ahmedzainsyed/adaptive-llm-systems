"""
evaluator/model.py

Multi-Head Evaluator Model for LLM Output Quality Assessment.

Architecture:
  - Shared backbone: frozen/LoRA-tuned LLM (LLaMA-3 8B / Mistral 7B)
  - Score head:       regression ∈ [0,1] — quality score
  - Confidence head:  epistemic confidence ∈ [0,1] — how sure the model is
  - Reasoning head:   token logits — for natural language explanation generation

The confidence head output feeds into:
  (a) UncertaintyAwareLoss in training
  (b) Active learning acquisition function
  (c) Project 2 synthetic preference filter
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from loguru import logger
from typing import Optional, Tuple


class EvaluatorModel(nn.Module):
    """
    Multi-head LLM evaluator with calibrated confidence output.

    Forward returns:
        score       : Tensor [B, 1]  — quality score in [0, 1]
        confidence  : Tensor [B, 1]  — calibrated epistemic confidence in [0, 1]
        reasoning   : Tensor [B, V]  — logits over vocab (for explanation decoding)
    """

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3-8B",
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        freeze_backbone: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        initial_temperature: float = 1.5,
    ):
        super().__init__()

        logger.info(f"Loading backbone: {base_model_name}")
        config = AutoConfig.from_pretrained(base_model_name)
        self.backbone = AutoModel.from_pretrained(
            base_model_name,
            config=config,
            torch_dtype=torch.float16,
        )

        # Optional: freeze backbone — only train the heads
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen. Training heads only.")

        # Optional: apply LoRA for parameter-efficient fine-tuning
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
            self.backbone.print_trainable_parameters()

        # ── Head 1: Quality Score ──────────────────────────────────────
        # Produces scalar quality estimate ∈ [0,1]
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # ── Head 2: Epistemic Confidence ──────────────────────────────
        # Produces calibrated confidence ∈ [0,1]
        # Low confidence → sample flagged for active learning query
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # ── Head 3: Reasoning / Explanation ───────────────────────────
        # Projects hidden state → vocab logits for generating explanation tokens
        self.reasoning_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # ── Learnable Temperature (Calibration) ───────────────────────
        # T > 1 softens confidence; T < 1 sharpens it.
        # Optimized during post-training calibration on held-out data.
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        """Temperature is always positive via exp."""
        return self.log_temperature.exp().clamp(min=0.1, max=10.0)

    def get_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract CLS / last-token hidden state from backbone."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use first token (CLS-equivalent) from the final hidden layer
        hidden = outputs.last_hidden_state[:, 0, :]  # [B, H]
        return hidden

    def calibrate_confidence(self, raw_logits: torch.Tensor) -> torch.Tensor:
        """
        Temperature scaling calibration.
        raw_logits / T — dividing by large T → softer (less extreme) confidence.
        After sigmoid: well-calibrated probability estimate.
        """
        return torch.sigmoid(raw_logits / self.temperature)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      [B, L] tokenized (prompt + response) input
            attention_mask: [B, L] padding mask

        Returns:
            score:      [B, 1] quality score ∈ [0, 1]
            confidence: [B, 1] calibrated epistemic confidence ∈ [0, 1]
            reasoning:  [B, V] vocab logits for explanation generation
        """
        h = self.get_hidden_state(input_ids, attention_mask)  # [B, H]

        # Score: raw logit → sigmoid → [0, 1]
        score = torch.sigmoid(self.score_head(h))             # [B, 1]

        # Confidence: raw logit → temperature calibration → [0, 1]
        raw_conf = self.confidence_head(h)                    # [B, 1]
        confidence = self.calibrate_confidence(raw_conf)      # [B, 1]

        # Reasoning: hidden → vocab logits (for optional explanation decoding)
        reasoning = self.reasoning_head(h)                    # [B, V]

        return score, confidence, reasoning

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> dict:
        """
        Inference-time prediction with structured output.

        Returns dict with:
            score       : float quality score
            confidence  : float calibrated confidence
            label       : str "high" | "medium" | "low"
            flag_for_human: bool whether to route to human annotator
        """
        self.eval()
        with torch.no_grad():
            score, confidence, _ = self.forward(input_ids, attention_mask)

        score_val = score.item()
        conf_val = confidence.item()

        return {
            "score": score_val,
            "confidence": conf_val,
            "label": "high" if score_val > 0.7 else ("medium" if score_val > 0.4 else "low"),
            "flag_for_human": conf_val < 0.5,  # low confidence → route to human
        }

    def save(self, path: str) -> None:
        """Save model weights and temperature parameter."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "temperature": self.temperature.item(),
            },
            path,
        )
        logger.info(f"EvaluatorModel saved to {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "EvaluatorModel":
        """Load saved model."""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"EvaluatorModel loaded from {path}")
        return model
