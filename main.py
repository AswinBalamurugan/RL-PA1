import os
from src.experiments.run_experiments import run_complete_experiment
from src.utils.analysis import analyze_results
from src.utils.logging import setup_wandb


def main():
    # Setup wandb
    wandb_enabled = setup_wandb()

    # Create results directories
    os.makedirs("results/Analysis", exist_ok=True)

    # Run experiments for CartPole-v1
    print("=== Starting CartPole-v1 Experiments ===")
    cartpole_results = run_complete_experiment("CartPole-v1", max_steps=1000)
    cartpole_stats = analyze_results(cartpole_results, "CartPole-v1")

    # Run experiments for MountainCar-v0
    print("\n=== Starting MountainCar-v0 Experiments ===")
    mountaincar_results = run_complete_experiment("MountainCar-v0", max_steps=1000)
    mountaincar_stats = analyze_results(mountaincar_results, "MountainCar-v0")

    print("\n=== Experiments Complete ===")
    print("Results saved in the results/ directory.")

    if wandb_enabled:
        print("Experiment logs available in Weights & Biases.")


if __name__ == "__main__":
    main()
