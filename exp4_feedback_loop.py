.PHONY: help install train-evaluator train-reward train-policy run-benchmark run-unified test lint format clean

help:
	@echo "Scalable LLM Research System — Makefile Commands"
	@echo "=================================================="
	@echo "  install              Install all dependencies"
	@echo "  train-evaluator      Train Project 1 evaluator"
	@echo "  train-reward         Train Project 2 reward model"
	@echo "  train-policy         Run Project 2 policy optimization"
	@echo "  run-benchmark        Run Project 3 benchmark evolution"
	@echo "  run-unified          Run full closed-loop system"
	@echo "  run-ablation         Run all ablation studies"
	@echo "  run-experiments      Run all project experiments"
	@echo "  test                 Run unit tests"
	@echo "  lint                 Run ruff linter"
	@echo "  format               Run black formatter"
	@echo "  clean                Remove cache files"

install:
	pip install -e ".[dev,serving,notebooks]"

# ── Project 1: Evaluator ────────────────────────────────────────────
train-evaluator:
	python evaluator/train.py \
		--config configs/evaluator_config.yaml \
		--wandb-project scalable-llm-evaluator

exp1-evaluator:
	python evaluator/experiments/exp1_static_vs_adaptive.py

exp2-label-efficiency:
	python evaluator/experiments/exp2_label_efficiency.py

exp3-ablation:
	python evaluator/experiments/exp3_ablation.py

exp4-dist-shift:
	python evaluator/experiments/exp4_distribution_shift.py

exp5-calibration:
	python evaluator/experiments/exp5_calibration.py

exp6-cost:
	python evaluator/experiments/exp6_cost_reduction.py

# ── Project 2: RLHF ────────────────────────────────────────────────
train-reward:
	python rlhf/train_reward.py \
		--config configs/rlhf_config.yaml \
		--wandb-project scalable-llm-rlhf

train-policy:
	python rlhf/train_policy.py \
		--config configs/rlhf_config.yaml \
		--algorithm dpo

train-policy-ppo:
	python rlhf/train_policy.py --algorithm ppo --config configs/rlhf_config.yaml

train-policy-grpo:
	python rlhf/train_policy.py --algorithm grpo --config configs/rlhf_config.yaml

compare-policies:
	python rlhf/experiments/exp6_policy_comparison.py

# ── Project 3: Benchmark ───────────────────────────────────────────
run-benchmark:
	python benchmark/evolution_loop.py \
		--config configs/benchmark_config.yaml \
		--n-cycles 10

exp1-benchmark:
	python benchmark/experiments/exp1_static_vs_adaptive.py

exp2-failure-discovery:
	python benchmark/experiments/exp2_failure_discovery.py

# ── Unified System ─────────────────────────────────────────────────
run-unified:
	python unified/pipeline.py \
		--config configs/evaluator_config.yaml \
		--rlhf-config configs/rlhf_config.yaml \
		--benchmark-config configs/benchmark_config.yaml \
		--n-iterations 10

run-end-to-end:
	python unified/end_to_end_experiment.py

run-experiments:
	$(MAKE) exp1-evaluator
	$(MAKE) exp2-label-efficiency
	$(MAKE) exp3-ablation
	$(MAKE) exp1-benchmark
	$(MAKE) exp1-evaluator
	$(MAKE) compare-policies

# ── Dev tools ──────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	ruff check evaluator/ rlhf/ benchmark/ unified/

format:
	black evaluator/ rlhf/ benchmark/ unified/ --line-length 100

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .ruff_cache .mypy_cache
