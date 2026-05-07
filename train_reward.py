# rlhf_config.yaml — Project 2: Data-Efficient RLHF

model:
  base_model: "meta-llama/Llama-3-8B"
  hidden_size: 4096
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  load_in_4bit: true

reward_model:
  learning_rate: 1.0e-5
  batch_size: 8
  gradient_accumulation_steps: 4
  max_steps: 3000
  warmup_steps: 150
  weight_decay: 0.01
  max_seq_length: 1024

confidence_filter:
  # Project 1 evaluator integration
  evaluator_checkpoint: "outputs/evaluator/best_checkpoint"
  confidence_threshold: 0.75                   # tau: accept synthetic pref if both conf > tau
  min_agreement: 0.60                          # kappa: fraction of judges that must agree
  k_judges: 5                                  # number of LLM judges to consult
  judge_model: "gpt-4o-mini"                   # judge model (can use open-source alternative)

synthetic_preferences:
  num_responses_per_prompt: 4                  # k candidates generated per prompt
  pair_sampling_strategy: "diversity_uncertainty"  # vs "random"
  diversity_temperature: 1.0
  max_new_tokens: 512

# Human data
human_label_budget: 0.05                       # 5% of total data budget for human labels
seed_human_dataset: "data/rlhf/human_prefs_seed.jsonl"

policy_optimization:
  # Default algorithm (compare all three in experiments)
  algorithm: "dpo"                             # "ppo" | "dpo" | "grpo" | "dpo_conf"

  dpo:
    beta: 0.1                                  # KL regularization
    learning_rate: 5.0e-7
    batch_size: 4
    gradient_accumulation_steps: 8
    max_steps: 2000
    label_smoothing: 0.0

  dpo_conf:                                    # Our novel variant
    beta: 0.1
    confidence_weighting: true                 # weight loss by evaluator confidence
    min_weight: 0.1
    learning_rate: 5.0e-7
    batch_size: 4
    gradient_accumulation_steps: 8
    max_steps: 2000

  ppo:
    learning_rate: 1.0e-6
    batch_size: 4
    ppo_epochs: 4
    clip_range: 0.2
    kl_coeff: 0.1
    vf_coeff: 0.1
    max_steps: 1000

  grpo:
    learning_rate: 1.0e-6
    group_size: 8                              # G: group comparisons per prompt
    beta: 0.01
    max_steps: 1000

active_learning_loop:
  n_rounds: 10
  synthetic_candidates_per_round: 10000
  human_labels_per_round: 200                  # minimal labeling budget

data:
  prompt_pool_path: "data/rlhf/prompt_pool.jsonl"
  synthetic_output_path: "data/rlhf/synthetic_prefs.jsonl"
  final_dataset_path: "data/rlhf/final_dataset.jsonl"

logging:
  wandb_project: "scalable-llm-rlhf"
  wandb_entity: "ahmedzainsyed"
  log_every_n_steps: 25
  eval_every_n_steps: 100
  output_dir: "outputs/rlhf"
