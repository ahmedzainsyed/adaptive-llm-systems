# evaluator_config.yaml — Project 1: Self-Improving Evaluator

model:
  base_model: "meta-llama/Llama-3-8B"       # backbone LLM
  hidden_size: 4096
  vocab_size: 32000
  freeze_backbone: true                        # freeze backbone, train heads only
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  load_in_4bit: true                           # QLoRA

training:
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_steps: 200
  max_steps: 5000
  batch_size: 16
  gradient_accumulation_steps: 4
  max_seq_length: 2048
  fp16: true
  seed: 42

  # Curriculum learning
  curriculum_warmup_epochs: 2
  difficulty_alpha: 0.4                        # weight for uncertainty in difficulty score
  difficulty_beta: 0.35                        # weight for disagreement
  difficulty_gamma: 0.25                       # weight for novelty

  # Uncertainty-aware loss
  lambda_conf: 0.1                             # confidence regularization coefficient

  # Temperature calibration
  initial_temperature: 1.5
  calibrate_every_n_steps: 500

active_learning:
  cycle_budget: 500                            # human labels per cycle
  disagreement_threshold: 0.20                 # delta for disagreement criterion
  confidence_threshold: 0.75                   # samples below this → query human
  uncertainty_weight: 0.40                     # alpha in S = alpha*unc + beta*dis + gamma*nov
  disagreement_weight: 0.35                    # beta
  novelty_weight: 0.25                         # gamma
  ensemble_size: 3                             # number of evaluator checkpoints for ensemble
  retrain_every_n_cycles: 1

data:
  train_path: "data/evaluator/train.jsonl"
  val_path: "data/evaluator/val.jsonl"
  test_path: "data/evaluator/test.jsonl"
  unlabeled_pool_path: "data/evaluator/unlabeled_pool.jsonl"
  # Dataset mix weights
  instruction_following_weight: 0.30
  reasoning_weight: 0.30
  safety_weight: 0.20
  real_world_weight: 0.20

logging:
  wandb_project: "scalable-llm-evaluator"
  wandb_entity: "ahmedzainsyed"
  log_every_n_steps: 50
  eval_every_n_steps: 200
  save_every_n_steps: 500
  mlflow_tracking_uri: "http://localhost:5000"
  output_dir: "outputs/evaluator"

service:
  host: "0.0.0.0"
  port: 8001
  kafka_bootstrap_servers: "localhost:9092"
  kafka_topic_hard_samples: "hard-samples"
  kafka_topic_uncertain_samples: "uncertain-samples"
  redis_url: "redis://localhost:6379"
