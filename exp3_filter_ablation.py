# benchmark_config.yaml — Project 3: Agentic Benchmark Generator

generator:
  # Task Generator Agent
  task_model: "gpt-4o-mini"                   # agent backbone (use Llama for open-source)
  n_seed_tasks: 200                            # initial tasks before evolution begins
  n_tasks_per_cycle: 500                       # tasks generated per evolution step
  task_categories:
    - reasoning
    - customer_support
    - safety
    - instruction_following
    - multi_step_decision
    - domain_specific
  category_weights: [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]

difficulty_controller:
  initial_difficulty: 0.3
  min_difficulty: 0.1
  max_difficulty: 1.0
  target_failure_rate: 0.50                    # maintain ~50% failure rate for max info
  increase_threshold: 0.80                     # if model success > 80%, increase difficulty
  decrease_threshold: 0.40                     # if failure rate > 60%, decrease difficulty
  difficulty_step: 0.05                        # how much to adjust per cycle
  window_size: 100                             # tasks to compute rolling failure rate over

failure_analyzer:
  # Failure taxonomy
  failure_types:
    - hallucination
    - logical_error
    - safety_violation
    - ambiguity_mishandling
    - instruction_deviation
    - multi_hop_reasoning_gap
    - factual_error
    - context_misunderstanding
  # Clustering
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  cluster_algorithm: "kmeans"
  n_clusters: 20                               # max clusters to track
  min_cluster_size: 5                          # minimum failures to form a cluster

task_refiner:
  # Adversarial diversity strategies
  strategies:
    - semantic_perturbation
    - compositional_complexity
    - distractor_insertion
    - counterfactual_reformulation
    - paraphrase_difficulty_increase
  tasks_per_failure_cluster: 10               # new tasks generated per cluster centroid
  top_k_clusters: 5                           # focus on top-5 failure modes per cycle

evaluation:
  # Connects to Project 1 evaluator service
  evaluator_endpoint: "http://localhost:8001/evaluate"
  failure_threshold: 0.40                      # score below this = failure
  batch_size: 32

evolution:
  n_cycles: 20                                 # total evolution cycles to run
  save_task_pool_every: 5                      # save task pool every N cycles
  max_task_pool_size: 50000

data:
  task_pool_path: "data/benchmark/task_pool.jsonl"
  failure_clusters_path: "data/benchmark/failure_clusters.json"
  output_dir: "outputs/benchmark"

logging:
  wandb_project: "scalable-llm-benchmark"
  wandb_entity: "ahmedzainsyed"
  log_every_n_cycles: 1
