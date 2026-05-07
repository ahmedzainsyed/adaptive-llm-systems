# Unified Integration Layer

This module connects all three projects into one closed-loop pipeline.

## Files

- `pipeline.py` — `UnifiedLLMResearchSystem`: runs all 3 modules together per iteration
- `end_to_end_experiment.py` — Headline result comparing all 4 conditions

## Run

```bash
python unified/pipeline.py \
  --config configs/evaluator_config.yaml \
  --rlhf-config configs/rlhf_config.yaml \
  --benchmark-config configs/benchmark_config.yaml \
  --n-iterations 10

python unified/end_to_end_experiment.py
```
