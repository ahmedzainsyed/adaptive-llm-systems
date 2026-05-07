from setuptools import setup, find_packages

setup(
    name="scalable-llm-system",
    version="1.0.0",
    author="Ahmed Zain Syed",
    author_email="ahmedzainsyed@github.com",
    description=(
        "A unified LLM research system: adaptive evaluation, "
        "data-efficient RLHF, and agentic benchmarking."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ahmedzainsyed/scalable-llm-system",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "datasets>=2.17.0",
        "accelerate>=0.27.0",
        "peft>=0.8.0",
        "trl>=0.7.11",
        "wandb>=0.16.0",
        "mlflow>=2.10.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0.1",
        "rich>=13.7.0",
        "loguru>=0.7.2",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.5.0",
    ],
    extras_require={
        "serving": ["fastapi>=0.109.0", "uvicorn>=0.27.0", "kafka-python>=2.0.2"],
        "dev": ["pytest>=8.0.0", "black>=24.2.0", "ruff>=0.2.2", "mypy>=1.8.0"],
        "notebooks": ["jupyter>=1.0.0", "matplotlib>=3.8.2", "plotly>=5.18.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
