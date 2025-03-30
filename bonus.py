import os
import numpy as np
import pandas as pd
import random
import time
from src.experiments.minigrid_5x5.train_minigrid import *
from src.experiments.minigrid_5x5.eval_minigrid import *
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from src.utils.visualization import plot_results, compare_algorithms
from collections import defaultdict
from src.utils.logging import log_to_wandb

SEEDS = [42, 123, 456, 789, 1024]


def bonus():
    """Run MiniGrid experiment with options to train or evaluate both Q-Learning and SARSA agents"""
    print("\n=== Starting MiniGrid Experiments ===")

    # Ask user to choose an option
    print("Choose an option:")
    print("1. Train both agents (5 seeds)")
    print("2. Evaluate both agents")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == "1":
        # Train both agents for 5 seeds
        print("\n=== Training Q-Learning and SARSA Agents (5 seeds) ===")

        # Initialize lists to store results across seeds
        all_q_rewards = []
        all_sarsa_rewards = []

        for seed in SEEDS:
            np.random.seed(seed)
            random.seed(seed)

            # Initialize agents with hyperparameters
            q_learning_agent = QLearningAgent(
                action_space=3,
                learning_rate=0.1,
                discount_factor=0.99,
                exploration_rate=1.0,
                exploration_decay=0.995,
                min_exploration_rate=0.01,
            )
            sarsa_agent = SARSAAgent(
                action_space=3,
                learning_rate=0.1,
                discount_factor=0.99,
                epsilon=1.0,
                epsilon_decay=0.995,
                min_epsilon=0.01,
            )

            # Train Q-Learning agent
            training_env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0")
            training_env = FullyObsWrapper(training_env)
            q_rewards, _ = q_learning_agent.train(
                training_env, episodes=10000, max_steps=100
            )
            training_env.close()
            if seed == SEEDS[-1]:  # Only save Q-table for the last seed
                q_learning_agent.save_q_table(PATH_Q_TABLE_QLEARNING)

            # Log Q-Learning results to WandB
            log_to_wandb(
                "Q-Learning",
                "MiniGrid-Dynamic-Obstacles-5x5-v0",
                {
                    "learning_rate": 0.1,
                    "discount_factor": 0.99,
                    "exploration_rate": 1.0,
                    "exploration_decay": 0.995,
                    "min_exploration_rate": 0.01,
                },
                q_rewards,
                seed,
            )

            # Train SARSA agent
            training_env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0")
            training_env = FullyObsWrapper(training_env)
            sarsa_rewards, _ = sarsa_agent.train(
                training_env, episodes=10000, max_steps=100
            )
            training_env.close()
            if seed == SEEDS[-1]:  # Only save Q-table for the last seed
                sarsa_agent.save_q_table(PATH_Q_TABLE_SARSA)

            # Log SARSA results to WandB
            log_to_wandb(
                "SARSA",
                "MiniGrid-Dynamic-Obstacles-5x5-v0",
                {
                    "learning_rate": 0.1,
                    "discount_factor": 0.99,
                    "epsilon": 1.0,
                    "epsilon_decay": 0.995,
                    "min_epsilon": 0.01,
                },
                sarsa_rewards,
                seed,
            )

            # Store results for this seed
            all_q_rewards.append(q_rewards)
            all_sarsa_rewards.append(sarsa_rewards)

        # Combine results across seeds
        q_results_df = combine_results(all_q_rewards, "Q-Learning")
        sarsa_results_df = combine_results(all_sarsa_rewards, "SARSA")

        # Plot individual results
        plot_results(q_results_df, "MiniGrid-Dynamic-Obstacles-5x5-v0", "Q-Learning")
        plot_results(sarsa_results_df, "MiniGrid-Dynamic-Obstacles-5x5-v0", "SARSA")

        # Compare algorithms
        compare_algorithms(
            sarsa_results_df, q_results_df, "MiniGrid-Dynamic-Obstacles-5x5-v0"
        )

    elif choice == "2":
        # Evaluate both agents
        print("\n=== Evaluating Q-Learning Agent ===")
        q_learning_agent = QLearningAgent(
            action_space=3,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.01,
        )
        if not q_learning_agent.load_q_table(PATH_Q_TABLE_QLEARNING):
            print("No Q-table found for Q-Learning. Please train the agent first.")
        else:
            visualize_agent(PATH_Q_TABLE_QLEARNING, "Q-Learning")

        print("\n=== Evaluating SARSA Agent ===")
        sarsa_agent = SARSAAgent(
            action_space=3,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01,
        )
        if not sarsa_agent.load_q_table(PATH_Q_TABLE_SARSA):
            print("No Q-table found for SARSA. Please train the agent first.")
        else:
            visualize_agent(PATH_Q_TABLE_SARSA, "SARSA")

    else:
        print("Invalid choice. Exiting MiniGrid experiment.")


def combine_results(all_rewards, agent_name):
    """Combine results from multiple seeds into a single DataFrame"""
    results = []
    for seed, rewards in enumerate(all_rewards):
        for episode, reward in enumerate(rewards):
            results.append(
                {
                    "seed": seed,
                    "episode": episode + 1,
                    "reward": reward,
                    "agent": agent_name,
                }
            )
    return pd.DataFrame(results)


def visualize_agent(q_table_filename, agent_name):
    """Evaluate the agent and visualize its performance"""
    # Action meanings for debugging
    ACTION_MEANINGS = {0: "Turn Left", 1: "Turn Right", 2: "Move Forward"}

    # Load Q-table
    q_table = np.load(q_table_filename, allow_pickle=True).item()

    # Create environment with human render mode
    env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0", render_mode="human")
    env = FullyObsWrapper(env)

    # Initialize a blank defaultdict with the loaded Q-table
    q_defaultdict = defaultdict(lambda: np.zeros(3))
    for k, v in q_table.items():
        q_defaultdict[k] = v

    episodes = 5
    max_steps = 100
    delay = 0.3  # Slow down visualization

    # Run episodes
    for episode in range(episodes):
        obs, _ = env.reset()
        state = get_state_representation(env)
        done = False
        step_count = 0
        episode_reward = 0
        moves_taken = 0

        print(f"\nEpisode {episode + 1} ({agent_name})")

        # Force render at the beginning of each episode
        env.render()
        time.sleep(1.0)  # Give time for render window to initialize

        while not done and step_count < max_steps:
            # Always try to render
            env.render()
            time.sleep(delay)

            # Choose best action from Q-table
            if state in q_defaultdict:
                q_values = q_defaultdict[state]
                action = np.argmax(q_values)
                print(f"State: {state}")
                print(f"Q-values: {q_values}")
                print(f"Chosen action: {ACTION_MEANINGS[action]}")

                if action == 2:
                    moves_taken += 1
            else:
                action = 2  # If state not in Q-table, try moving forward
                print("State not found in Q-table, moving forward")
                moves_taken += 1

            # Take action
            _, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = get_state_representation(env)
            step_count += 1

        success = episode_reward > 0
        print(f"Episode result: {'Success' if success else 'Failure'}")
        print(
            f"Finished in {step_count} steps with {moves_taken} actual movements and reward {episode_reward}"
        )

    env.close()


if __name__ == "__main__":
    bonus()
