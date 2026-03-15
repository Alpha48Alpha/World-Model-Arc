from setuptools import setup, find_packages

setup(
    name="world-model-arc",
    version="0.1.0",
    description=(
        "Production-grade AI research lab: modular PyTorch RL system "
        "with rich simulated worlds, policy-gradient + DQN agents, "
        "experiment configs, checkpointing, metrics logging, evaluation, "
        "and visualization."
    ),
    packages=find_packages(where=".", include=["src*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "gymnasium>=0.29.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
    entry_points={
        "console_scripts": [
            "wma-train=scripts.train:main",
            "wma-evaluate=scripts.evaluate:main",
        ],
    },
)
