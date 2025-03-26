import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
import wandb
import os

from ..agents.sarsa import SARSAAgent, SARSAAgentWithShaping
from ..agents.qlearning import QLearningAgent, QLearningAgentWithShaping
from ..utils.reward_shaping import shape_reward_cartpole, shape_reward_mountaincar
from ..utils.visualization import plot_results, compare_algorithms

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    gym.utils.seeding.np_random(seed)

def run_experiment(env_name, agent_class, hyperparams, num_episodes, max_steps, seeds):
    """
    Run experiments with given hyperparameters across multiple seeds
    """
    all_results = []
    
    for seed in seeds:
        set_seed(seed)
        env = gym.make(env_name)
        
        # Create agent based on agent_class
        if agent_class == SARSAAgent:
            agent = SARSAAgent(
                env, 
                learning_rate=hyperparams['learning_rate'],
                discount_factor=hyperparams['discount_factor'],
                epsilon=hyperparams['epsilon']
            )
        elif agent_class == QLearningAgent:
            agent = QLearningAgent(
                env, 
                learning_rate=hyperparams['learning_rate'],
                discount_factor=hyperparams['discount_factor'],
                temperature=hyperparams['temperature']
            )
        elif agent_class == SARSAAgentWithShaping:
            agent = SARSAAgentWithShaping(
                env, 
                learning_rate=hyperparams['learning_rate'],
                discount_factor=hyperparams['discount_factor'],
                epsilon=hyperparams['epsilon'],
                reward_shaping_fn=hyperparams.get('reward_shaping_fn')
            )
        elif agent_class == QLearningAgentWithShaping:
            agent = QLearningAgentWithShaping(
                env, 
                learning_rate=hyperparams['learning_rate'],
                discount_factor=hyperparams['discount_factor'],
                temperature=hyperparams['temperature'],
                reward_shaping_fn=hyperparams.get('reward_shaping_fn')
            )
        
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
            if param != 'reward_shaping_fn':  # Skip function objects
                df[param] = value
        
        all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True)

def grid_search(env_name, agent_class, param_grid, num_episodes, max_steps, seeds):
    """
    Perform grid search for hyperparameters
    """
    import itertools
    
    keys = param_grid.keys()
    values = param_grid.values()
    hyperparams_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    results = []
    
    for hyperparams in tqdm(hyperparams_combinations, desc=f"Grid search for {agent_class.__name__}"):
        # Run with first seed to save time during grid search
        set_seed(seeds[0])
        env = gym.make(env_name)
        
        if agent_class == SARSAAgent:
            agent = SARSAAgent(
                env, 
                learning_rate=hyperparams['learning_rate'],
                discount_factor=hyperparams['discount_factor'],
                epsilon=hyperparams['epsilon']
            )
        else:  # QLearningAgent
            agent = QLearningAgent(
                env, 
                learning_rate=hyperparams['learning_rate'],
                discount_factor=hyperparams['discount_factor'],
                temperature=hyperparams['temperature']
            )
        
        episode_rewards = agent.train(num_episodes, max_steps)
        
        # Use last 10% episodes to measure performance
        last_rewards = episode_rewards[int(0.9 * len(episode_rewards)):]
        avg_reward = np.mean(last_rewards)
        
        results.append((hyperparams, avg_reward))
    
    # Sort by average reward in descending order
    return sorted(results, key=lambda x: x[1], reverse=True)

def run_complete_experiment(env_name, discount_factor=0.99, max_steps=1000):
    """
    Run complete experiment for an environment
    """
    seeds = [42, 123, 456, 789, 1024]
    
    # Define hyperparameter search grid for SARSA
    sarsa_param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'discount_factor': [discount_factor],  # Fixed to 0.99 as recommended
        'epsilon': [0.01, 0.1, 0.2]
    }
    
    # Define hyperparameter search grid for Q-Learning
    qlearning_param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'discount_factor': [discount_factor],  # Fixed to 0.99 as recommended
        'temperature': [0.1, 1.0, 5.0]
    }
    
    # Number of episodes for grid search (fewer to save time)
    grid_search_episodes = 500
    
    # Number of episodes for final training
    final_training_episodes = 5000
    
    print(f"Starting grid search for SARSA on {env_name}...")
    sarsa_results = grid_search(env_name, SARSAAgent, sarsa_param_grid, 
                              grid_search_episodes, max_steps, seeds)
    print("Top 3 SARSA hyperparameters:")
    for i, (params, reward) in enumerate(sarsa_results[:3]):
        print(f"{i+1}. {params} -> Reward: {reward:.2f}")
    
    print(f"\nStarting grid search for Q-Learning on {env_name}...")
    qlearning_results = grid_search(env_name, QLearningAgent, qlearning_param_grid, 
                                  grid_search_episodes, max_steps, seeds)
    print("Top 3 Q-Learning hyperparameters:")
    for i, (params, reward) in enumerate(qlearning_results[:3]):
        print(f"{i+1}. {params} -> Reward: {reward:.2f}")
    
    # Choose reward shaping function
    reward_shaping_fn = None
    if env_name == 'CartPole-v1':
        reward_shaping_fn = shape_reward_cartpole
    elif env_name == 'MountainCar-v0':
        reward_shaping_fn = shape_reward_mountaincar
    
    # Train with best hyperparameters across all seeds
    print(f"\nTraining SARSA with best hyperparameters on {env_name}...")
    best_sarsa_params = sarsa_results[0][0]
    sarsa_df = run_experiment(env_name, SARSAAgent, best_sarsa_params, 
                            final_training_episodes, max_steps, seeds)
    
    print(f"Training SARSA with reward shaping on {env_name}...")
    sarsa_shaped_params = {**best_sarsa_params, 'reward_shaping_fn': reward_shaping_fn}
    sarsa_shaped_df = run_experiment(env_name, SARSAAgentWithShaping, sarsa_shaped_params, 
                                   final_training_episodes, max_steps, seeds)
    
    print(f"Training Q-Learning with best hyperparameters on {env_name}...")
    best_qlearning_params = qlearning_results[0][0]
    qlearning_df = run_experiment(env_name, QLearningAgent, best_qlearning_params, 
                                final_training_episodes, max_steps, seeds)
    
    print(f"Training Q-Learning with reward shaping on {env_name}...")
    qlearning_shaped_params = {**best_qlearning_params, 'reward_shaping_fn': reward_shaping_fn}
    qlearning_shaped_df = run_experiment(env_name, QLearningAgentWithShaping, qlearning_shaped_params, 
                                       final_training_episodes, max_steps, seeds)
    
    # Plot results
    print("Plotting results...")
    
    # Plot SARSA results
    plot_results(sarsa_df, env_name, "SARSA")
    
    # Plot SARSA with shaping results
    plot_results(sarsa_shaped_df, env_name, "SARSA with Reward Shaping")
    
    # Plot Q-Learning results
    plot_results(qlearning_df, env_name, "Q-Learning")
    
    # Plot Q-Learning with shaping results
    plot_results(qlearning_shaped_df, env_name, "Q-Learning with Reward Shaping")
    
    # Compare SARSA vs Q-Learning
    compare_algorithms(sarsa_df, qlearning_df, env_name)
    
    # Compare SARSA with shaping vs Q-Learning with shaping
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