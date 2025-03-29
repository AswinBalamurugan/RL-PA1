import os
from src.experiments.run_experiments import run_complete_experiment
from src.utils.analysis import analyze_results
from src.utils.logging import setup_wandb
from src.experiments.minigrid_5x5.train_minigrid import *
from src.experiments.minigrid_5x5.eval_minigrid import *
import gymnasium as gym
import numpy as np
from minigrid.wrappers import FullyObsWrapper
from collections import defaultdict
import time


def create_requirements_file():
    """
    Create requirements.txt file for reproducibility
    """
    requirements = [
        "numpy>=1.19.0",
        "gymnasium>=0.28.1",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "torch>=1.10.0",
        "wandb>=0.12.0",
        "minigrid>=2.1.0",  # For the bonus task
    ]

    with open("requirements.txt", "w") as f:
        for req in requirements:
            f.write(f"{req}\n")

    print("Created requirements.txt")


def main():
    # Create requirements.txt
    create_requirements_file()

    # Setup wandb
    wandb_enabled = setup_wandb()

    # Create results directories
    os.makedirs("results/Analysis", exist_ok=True)

    # # Run experiments for CartPole-v1
    # print("=== Starting CartPole-v1 Experiments ===")
    # cartpole_results = run_complete_experiment("CartPole-v1", max_steps=500)
    # cartpole_stats = analyze_results(cartpole_results, "CartPole-v1")

    # # Run experiments for MountainCar-v0
    # print("\n=== Starting MountainCar-v0 Experiments ===")
    # mountaincar_results = run_complete_experiment("MountainCar-v0", max_steps=1000)
    # mountaincar_stats = analyze_results(mountaincar_results, "MountainCar-v0")

    print("\n=== Experiments Complete ===")
    print("Results saved in the results/ directory.")

    if wandb_enabled:
        print("Experiment logs available in Weights & Biases.")


if __name__ == "__main__":
    main()
