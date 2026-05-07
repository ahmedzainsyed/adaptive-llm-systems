# Core deep learning
torch>=2.0.0
transformers>=4.38.0
datasets>=2.17.0
accelerate>=0.27.0
peft>=0.8.0

# Training frameworks
trl>=0.7.11              # PPO / DPO / GRPO support
deepspeed>=0.13.0        # distributed training
bitsandbytes>=0.42.0     # quantization (QLoRA)

# Reward modeling & alignment
scipy>=1.12.0
scikit-learn>=1.4.0
numpy>=1.26.0

# Experiment tracking
wandb>=0.16.0
mlflow>=2.10.0

# Serving & queuing
fastapi>=0.109.0
uvicorn>=0.27.0
kafka-python>=2.0.2
redis>=5.0.1
pydantic>=2.5.0
httpx>=0.26.0

# Data engineering
pandas>=2.2.0
pyarrow>=15.0.0
jsonlines>=4.0.0

# Calibration & evaluation metrics
netcal>=1.3.5            # Expected Calibration Error
evaluate>=0.4.1          # HuggingFace evaluate
nltk>=3.8.1
rouge-score>=0.1.2

# Clustering & similarity (Project 3)
faiss-cpu>=1.7.4         # fast similarity search
sentence-transformers>=2.5.0

# Visualization & notebooks
matplotlib>=3.8.2
seaborn>=0.13.2
plotly>=5.18.0
jupyter>=1.0.0
ipywidgets>=8.1.1

# LLM as Judge (Project 2 & 3)
openai>=1.12.0           # GPT-4 judge API
anthropic>=0.18.0        # Claude judge API

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0.1
rich>=13.7.0
typer>=0.9.0
tqdm>=4.66.2
loguru>=0.7.2
