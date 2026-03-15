"""Package setup for World-Model-Arc AI Research Lab."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="world-model-arc",
    version="0.1.0",
    author="Alpha48Alpha",
    description=(
        "Production-grade AI research lab: modular PyTorch RL system "
        "with rich simulated worlds, DQN + policy-gradient agents, "
        "experiment configs, checkpointing, metrics logging, and visualization."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "flake8>=6.0",
            "black>=23.0",
            "isort>=5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "wma-train=world_model_arc.scripts.train:main",
            "wma-evaluate=world_model_arc.scripts.evaluate:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
