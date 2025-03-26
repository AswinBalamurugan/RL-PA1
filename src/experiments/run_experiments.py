import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
import wandb
import os

from src.agents.base import RLAgent

from ..agents.sarsa import SARSAAgent, SARSAAgentWithShaping
from ..agents.qlearning import QLearningAgent, QLearningAgentWithShaping
from ..utils.reward_shaping import shape_reward_cartpole, shape_reward_mountaincar
from ..utils.visualization import plot_results, compare_algorithms

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    gym.utils.seeding.np_random(seed)

def run_experiment(env_name, agent_class, hyperparams, num_episodes, max_steps, seeds, bins=20, low=None, high=None):
    """Run experiments with given hyperparameters across multiple seeds."""
    all_results = []
    
    for seed in seeds:
        set_seed(seed)
        env = gym.make(env_name)
        
        # Create agent based on the specified class
        agent = agent_class(env, **hyperparams, bins=bins, low=low, high=high)
        
        # Train the agent and collect rewards
        episode_rewards = agent.train(num_episodes, max_steps)
        
        # Create DataFrame for this run
        df = pd.DataFrame({
            'episode': np.arange(1, len(episode_rewards) + 1),
            'reward': episode_rewards,
            'seed': seed,
            'agent': agent_class.__name__,
            'env': env_name
        })
        
        # Add hyperparameters to DataFrame
        for param, value in hyperparams.items():
            df[param] = value
        
        all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True)

def grid_search(env_name, agent_class, param_grid, num_episodes, max_steps, seeds, bins=20, low=None, high=None):
    """Perform grid search for hyperparameters."""
    import itertools
    
    # Generate all combinations of hyperparameters
    hyperparams_combinations = [dict(zip(param_grid.keys(), combo)) for combo in itertools.product(*param_grid.values())]
    
    results = []
    
    for hyperparams in tqdm(hyperparams_combinations, desc=f"Grid search for {agent_class.__name__}"):
        set_seed(seeds[0])  # Use the first seed for grid search
        env = gym.make(env_name)
        
        # Create agent with current hyperparameters
        agent = agent_class(env, **hyperparams, bins=bins, low=low, high=high)
        
        # Train the agent and collect rewards
        episode_rewards = agent.train(num_episodes, max_steps)
        
        # Measure performance using the last 10% of episodes
        last_rewards = episode_rewards[int(0.9 * len(episode_rewards)):]
        avg_reward = np.mean(last_rewards)
        
        results.append((hyperparams, avg_reward))
    
    return sorted(results, key=lambda x: x[1], reverse=True)

def get_environment_parameters(env_name):
    """Return bins, low, and high values based on the environment."""
    env_params = {
        'CartPole-v1': {'bins': 30, 'low': -4, 'high': 4},
        'MountainCar-v0': {'bins': 20, 'low': -10, 'high': 10},
        'MiniGrid-Dynamic-Obstacles-5x5-v0': {'bins': 15, 'low': 0, 'high': 10}
    }
    return env_params.get(env_name, {'bins': 20, 'low': None, 'high': None})  # Default values

def run_complete_experiment(env_name, discount_factor=0.99, max_steps=1000):
    """Run complete experiment for an environment."""
    seeds = [42, 123, 456, 789, 1024]
    
    # Get environment-specific parameters
    env_params = get_environment_parameters(env_name)
    bins = env_params['bins']
    low = env_params['low']
    high = env_params['high']
    
    # Define hyperparameter search grids
    sarsa_param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'discount_factor': [discount_factor],
        'epsilon': [0.01, 0.1, 0.2]
    }
    
    qlearning_param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'discount_factor': [discount_factor],
        'temperature': [0.1, 1.0, 5.0]
    }
    
    # Number of episodes for grid search and final training
    grid_search_episodes = 2500
    final_training_episodes = 15000
    
    print(f"Starting grid search for SARSA on {env_name}...")
    sarsa_results = grid_search(env_name, SARSAAgent, sarsa_param_grid, grid_search_episodes, max_steps, seeds, bins=bins, low=low, high=high)
    print("Top 3 SARSA hyperparameters:")
    for i, (params, reward) in enumerate(sarsa_results[:3]):
        print(f"{i+1}. {params} -> Reward: {reward:.2f}")
    
    print(f"\nStarting grid search for Q-Learning on {env_name}...")
    qlearning_results = grid_search(env_name, QLearningAgent, qlearning_param_grid, grid_search_episodes, max_steps, seeds, bins=bins, low=low, high=high)
    print("Top 3 Q-Learning hyperparameters:")
    for i, (params, reward) in enumerate(qlearning_results[:3]):
        print(f"{i+1}. {params} -> Reward: {reward:.2f}")
    
    # Choose reward shaping function based on environment
    reward_shaping_fn = None
    if env_name == 'CartPole-v1':
        reward_shaping_fn = shape_reward_cartpole
    elif env_name == 'MountainCar-v0':
        reward_shaping_fn = shape_reward_mountaincar
    
    # Train with best hyperparameters across all seeds
    print(f"\nTraining SARSA with best hyperparameters on {env_name}...")
    best_sarsa_params = sarsa_results[0][0]
    sarsa_df = run_experiment(env_name, SARSAAgent, best_sarsa_params, final_training_episodes, max_steps, seeds, bins=bins, low=low, high=high)
    
    print(f"Training SARSA with reward shaping on {env_name}...")
    sarsa_shaped_params = {**best_sarsa_params, 'reward_shaping_fn': reward_shaping_fn}
    sarsa_shaped_df = run_experiment(env_name, SARSAAgentWithShaping, sarsa_shaped_params, final_training_episodes, max_steps, seeds, bins=bins, low=low, high=high)
    
    print(f"Training Q-Learning with best hyperparameters on {env_name}...")
    best_qlearning_params = qlearning_results[0][0]
    qlearning_df = run_experiment(env_name, QLearningAgent, best_qlearning_params, final_training_episodes, max_steps, seeds, bins=bins, low=low, high=high)
    
    print(f"Training Q-Learning with reward shaping on {env_name}...")
    qlearning_shaped_params = {**best_qlearning_params, 'reward_shaping_fn': reward_shaping_fn}
    qlearning_shaped_df = run_experiment(env_name, QLearningAgentWithShaping, qlearning_shaped_params, final_training_episodes, max_steps, seeds, bins=bins, low=low, high=high)
    
    # Plot results
    print("Plotting results...")
    plot_results(sarsa_df, env_name, "SARSA")
    plot_results(sarsa_shaped_df, env_name, "SARSA with Reward Shaping")
    plot_results(qlearning_df, env_name, "Q-Learning")
    plot_results(qlearning_shaped_df, env_name, "Q-Learning with Reward Shaping")
    
    # Compare algorithms
    compare_algorithms(sarsa_df, qlearning_df, env_name)
    compare_algorithms(sarsa_shaped_df, qlearning_shaped_df, env_name)
    
    # Return all dataframes and best parameters
    return {
        'sarsa_df': sarsa_df,
        'sarsa_shaped_df': sarsa_shaped_df,
        'qlearning_df': qlearning_df,
        'qlearning_shaped_df': qlearning_shaped_df,
        'best_sarsa_params': best_sarsa_params,
        'best_qlearning_params': best_qlearning_params
    } 