"""
evaluator/service/evaluation_service.py

Production Evaluation Service — FastAPI microservice for the EvaluatorModel.

Exposes REST endpoints consumed by:
  - Project 2 RLHF confidence filter
  - Project 3 Benchmark failure detection
  - External callers (annotation UI, downstream services)

Endpoints:
  POST /evaluate          — score a single (prompt, response) pair
  POST /evaluate/batch    — score a batch of pairs
  GET  /health            — liveness check
  GET  /metrics           — current calibration metrics

Kafka Integration:
  Samples flagged as uncertain or disagreed are pushed to the
  Kafka topic "uncertain-samples" for the human annotation queue.

Usage:
    uvicorn evaluator.service.evaluation_service:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class EvaluationRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt / instruction")
    response: str = Field(..., description="The model response to evaluate")
    sample_id: Optional[str] = Field(None, description="Optional identifier for tracking")
    route_uncertain: bool = Field(True, description="Push uncertain samples to Kafka queue")


class EvaluationResponse(BaseModel):
    sample_id: Optional[str]
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score ∈ [0, 1]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Evaluator confidence ∈ [0, 1]")
    label: str = Field(..., description="'high' | 'medium' | 'low'")
    flag_for_human: bool = Field(..., description="True if routed to human annotation queue")
    explanation: Optional[str] = Field(None, description="Natural language explanation (if enabled)")
    latency_ms: float


class BatchEvaluationRequest(BaseModel):
    items: List[EvaluationRequest]


class BatchEvaluationResponse(BaseModel):
    results: List[EvaluationResponse]
    n_flagged: int
    mean_confidence: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    n_evaluations_served: int


# ── App state ─────────────────────────────────────────────────────────────────

_state: dict = {
    "evaluator": None,
    "tokenizer": None,
    "kafka_producer": None,
    "n_evaluations": 0,
    "device": "cpu",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialize Kafka on startup; clean up on shutdown."""
    logger.info("Starting Evaluation Service...")

    # Load evaluator model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _state["device"] = device

    model_path = os.getenv("EVALUATOR_CHECKPOINT", "outputs/evaluator/best_checkpoint.pt")
    base_model  = os.getenv("BASE_MODEL", "meta-llama/Llama-3-8B")

    try:
        from transformers import AutoTokenizer
        from evaluator.model import EvaluatorModel

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        evaluator = EvaluatorModel(base_model_name=base_model)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            evaluator.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded evaluator from {model_path}")
        else:
            logger.warning(f"No checkpoint at {model_path}. Using randomly initialized model.")

        evaluator = evaluator.to(device)
        evaluator.eval()

        _state["evaluator"] = evaluator
        _state["tokenizer"] = tokenizer

    except Exception as e:
        logger.error(f"Failed to load evaluator: {e}")
        logger.warning("Service will run in mock mode.")

    # Initialize Kafka producer (optional)
    kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    if kafka_servers:
        try:
            from kafka import KafkaProducer
            import json as _json
            _state["kafka_producer"] = KafkaProducer(
                bootstrap_servers=kafka_servers,
                value_serializer=lambda v: _json.dumps(v).encode("utf-8"),
            )
            logger.info(f"Kafka producer connected: {kafka_servers}")
        except ImportError:
            logger.warning("kafka-python not installed. Kafka routing disabled.")
        except Exception as e:
            logger.warning(f"Kafka connection failed: {e}. Routing disabled.")

    logger.info(f"Evaluation Service ready on device={device}")
    yield

    # Cleanup
    if _state.get("kafka_producer"):
        _state["kafka_producer"].close()
    logger.info("Evaluation Service shut down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Evaluation Service",
    description=(
        "Production evaluation microservice for Project 1 (Self-Improving Evaluator). "
        "Provides calibrated quality scores and epistemic confidence for (prompt, response) pairs. "
        "Uncertain samples are routed to the human annotation queue via Kafka."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _evaluate_single(prompt: str, response: str) -> tuple[float, float, str]:
    """
    Run the evaluator on one (prompt, response) pair.
    Returns (score, confidence, explanation_placeholder).
    """
    evaluator = _state.get("evaluator")
    tokenizer = _state.get("tokenizer")

    if evaluator is None or tokenizer is None:
        # Mock mode: return placeholder values
        import random
        score = random.uniform(0.3, 0.9)
        confidence = random.uniform(0.4, 0.95)
        return score, confidence, "Mock mode — evaluator not loaded."

    text = f"Prompt: {prompt}\n\nResponse: {response}"
    enc = tokenizer(
        text, return_tensors="pt",
        max_length=2048, truncation=True,
    ).to(_state["device"])

    with torch.no_grad():
        score_t, conf_t, reasoning_t = evaluator(
            enc["input_ids"], enc["attention_mask"]
        )

    return score_t.item(), conf_t.item(), ""


def _push_to_kafka(topic: str, payload: dict) -> None:
    """Push a sample to Kafka annotation queue (non-blocking)."""
    producer = _state.get("kafka_producer")
    if producer:
        try:
            producer.send(topic, payload)
        except Exception as e:
            logger.warning(f"Kafka push failed: {e}")


def _make_response(
    request: EvaluationRequest,
    score: float,
    confidence: float,
    explanation: str,
    latency_ms: float,
) -> EvaluationResponse:
    """Build EvaluationResponse and handle routing logic."""
    CONFIDENCE_THRESHOLD = 0.75

    label = "high" if score > 0.7 else ("medium" if score > 0.4 else "low")
    flag = confidence < CONFIDENCE_THRESHOLD

    if flag and request.route_uncertain:
        _push_to_kafka("uncertain-samples", {
            "sample_id": request.sample_id,
            "prompt": request.prompt[:500],      # truncate for queue
            "response": request.response[:500],
            "score": score,
            "confidence": confidence,
        })

    _state["n_evaluations"] += 1

    return EvaluationResponse(
        sample_id=request.sample_id,
        score=round(score, 4),
        confidence=round(confidence, 4),
        label=label,
        flag_for_human=flag,
        explanation=explanation or None,
        latency_ms=round(latency_ms, 2),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("evaluator") is not None,
        device=_state["device"],
        n_evaluations_served=_state["n_evaluations"],
    )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """
    Evaluate a single (prompt, response) pair.

    Returns quality score + calibrated confidence.
    If confidence < threshold, automatically routes to human annotation queue.
    """
    t0 = time.perf_counter()
    try:
        score, confidence, explanation = _evaluate_single(request.prompt, request.response)
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000
    return _make_response(request, score, confidence, explanation, latency_ms)


@app.post("/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(request: BatchEvaluationRequest):
    """
    Evaluate a batch of (prompt, response) pairs.

    Returns all scores + aggregate stats.
    """
    t0 = time.perf_counter()
    results = []
    for item in request.items:
        try:
            score, confidence, explanation = _evaluate_single(item.prompt, item.response)
            results.append(_make_response(item, score, confidence, explanation, 0.0))
        except Exception as e:
            logger.error(f"Batch item error: {e}")
            results.append(EvaluationResponse(
                sample_id=item.sample_id, score=0.0, confidence=0.0,
                label="low", flag_for_human=True,
                explanation=f"Error: {str(e)}", latency_ms=0.0,
            ))

    total_latency = (time.perf_counter() - t0) * 1000
    n_flagged = sum(r.flag_for_human for r in results)
    mean_conf = sum(r.confidence for r in results) / max(len(results), 1)

    return BatchEvaluationResponse(
        results=results,
        n_flagged=n_flagged,
        mean_confidence=round(mean_conf, 4),
        latency_ms=round(total_latency, 2),
    )


@app.get("/metrics")
async def get_metrics():
    """Return current service and calibration metrics."""
    return {
        "n_evaluations_served": _state["n_evaluations"],
        "model_loaded": _state.get("evaluator") is not None,
        "device": _state["device"],
        "kafka_connected": _state.get("kafka_producer") is not None,
        "confidence_threshold": 0.75,
    }
