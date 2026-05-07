"""
evaluator/service/kafka_producer.py

Kafka Producer for the Evaluator Feedback Queue.

Publishes flagged samples (low confidence / high disagreement) to
Kafka topics for human annotation. Two topics:

  hard-samples     : high disagreement between evaluator and ensemble
  uncertain-samples: low evaluator confidence (c_θ < threshold)

Downstream consumers:
  - Human Annotation UI reads from these topics
  - Training Service retrains evaluator on new human labels
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger


@dataclass
class FeedbackMessage:
    """Message published to Kafka for human annotation."""
    sample_id: str
    prompt: str
    response: str
    evaluator_score: float
    evaluator_confidence: float
    reason: str          # "low_confidence" | "high_disagreement" | "high_entropy"
    timestamp: float
    priority: int        # 1=high, 2=medium, 3=low (used for annotation queue ordering)


class FeedbackQueueProducer:
    """
    Produces messages to Kafka feedback queues.

    Topics:
        hard-samples:      Samples where evaluator disagrees with ensemble
        uncertain-samples: Samples with low evaluator confidence

    Args:
        bootstrap_servers: Kafka broker address(es)
        topic_hard:        Topic for hard/disagreement samples
        topic_uncertain:   Topic for low-confidence samples
    """

    TOPIC_HARD = "hard-samples"
    TOPIC_UNCERTAIN = "uncertain-samples"

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_hard: str = "hard-samples",
        topic_uncertain: str = "uncertain-samples",
    ):
        self.topic_hard = topic_hard
        self.topic_uncertain = topic_uncertain
        self._producer = None
        self._available = False

        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",           # wait for all replicas to acknowledge
                retries=3,
                request_timeout_ms=5000,
            )
            self._available = True
            logger.info(f"Kafka producer connected to {bootstrap_servers}")
        except Exception as e:
            logger.warning(
                f"Kafka producer unavailable ({e}). "
                f"Flagged samples will be logged only."
            )

    def publish_uncertain(
        self,
        sample_id: str,
        prompt: str,
        response: str,
        score: float,
        confidence: float,
        priority: int = 2,
    ) -> bool:
        """
        Publish a low-confidence sample to the uncertain-samples topic.

        Args:
            sample_id:  Unique identifier for tracking
            prompt:     Input prompt text
            response:   Model response text
            score:      Evaluator quality score
            confidence: Evaluator confidence (low → flag)
            priority:   Annotation priority (1=urgent, 2=normal, 3=low)

        Returns:
            True if published successfully, False otherwise
        """
        msg = FeedbackMessage(
            sample_id=sample_id,
            prompt=prompt,
            response=response,
            evaluator_score=score,
            evaluator_confidence=confidence,
            reason="low_confidence",
            timestamp=time.time(),
            priority=priority,
        )
        return self._publish(self.topic_uncertain, sample_id, asdict(msg))

    def publish_hard(
        self,
        sample_id: str,
        prompt: str,
        response: str,
        score: float,
        ensemble_score: float,
        confidence: float,
        priority: int = 1,
    ) -> bool:
        """
        Publish a high-disagreement sample to the hard-samples topic.

        Args:
            sample_id:      Unique identifier
            prompt:         Input prompt
            response:       Model response
            score:          Current evaluator score
            ensemble_score: Mean ensemble score (disagreement = |score - ensemble|)
            confidence:     Evaluator confidence
            priority:       Annotation priority (1=urgent for high disagreement)

        Returns:
            True if published successfully
        """
        msg = FeedbackMessage(
            sample_id=sample_id,
            prompt=prompt,
            response=response,
            evaluator_score=score,
            evaluator_confidence=confidence,
            reason="high_disagreement",
            timestamp=time.time(),
            priority=priority,
        )
        data = asdict(msg)
        data["ensemble_score"] = ensemble_score
        data["disagreement_magnitude"] = abs(score - ensemble_score)
        return self._publish(self.topic_hard, sample_id, data)

    def _publish(self, topic: str, key: str, value: dict) -> bool:
        """Internal publish method with fallback to logging."""
        if self._available and self._producer is not None:
            try:
                future = self._producer.send(topic, key=key, value=value)
                future.get(timeout=5)  # block until confirmed
                logger.debug(f"Published to {topic}: sample_id={key}")
                return True
            except Exception as e:
                logger.error(f"Kafka publish failed ({topic}): {e}")
                self._log_fallback(topic, value)
                return False
        else:
            self._log_fallback(topic, value)
            return False

    def _log_fallback(self, topic: str, value: dict) -> None:
        """Fallback: log the message when Kafka is unavailable."""
        logger.info(
            f"[KAFKA-FALLBACK] Topic={topic} | "
            f"sample_id={value.get('sample_id')} | "
            f"score={value.get('evaluator_score', 'N/A'):.3f} | "
            f"confidence={value.get('evaluator_confidence', 'N/A'):.3f} | "
            f"reason={value.get('reason')}"
        )

    def flush(self) -> None:
        """Flush all pending messages."""
        if self._available and self._producer is not None:
            self._producer.flush()

    def close(self) -> None:
        """Close the producer connection."""
        if self._producer is not None:
            self._producer.close()
            logger.info("Kafka producer closed.")
