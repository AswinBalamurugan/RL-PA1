import numpy as np
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from ..environments.wrappers import MiniGridWrapper
from ..agents.base import RLAgent
from collections import defaultdict

class MiniGridRLAgent(RLAgent):
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, temperature=1.0):
        self.env = MiniGridWrapper(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.temperature = temperature
        
        # Use a dictionary as Q-table for sparse representation
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        
        # Set observation_space and action_space from the wrapped environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def discretize_observation(self, observation):
        # Flatten the observation to use as dictionary key
        return tuple(observation.flatten())

class MiniGridSARSAAgent(MiniGridRLAgent):
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(env, learning_rate, discount_factor, epsilon)
    
    def train(self, num_episodes, max_steps):
        episode_rewards = []
        
        for episode in tqdm(range(num_episodes), desc="MiniGrid SARSA Training"):
            state, _ = self.env.reset()
            action = self.epsilon_greedy_policy(state)
            
            episode_reward = 0
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < max_steps:
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_action = self.epsilon_greedy_policy(next_state)
                
                # SARSA update rule
                current_q = self.get_q_value(state, action)
                next_q = self.get_q_value(next_state, next_action)
                
                new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
                self.set_q_value(state, action, new_q)
                
                state = next_state
                action = next_action
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
        
        return episode_rewards

class MiniGridQLearningAgent(MiniGridRLAgent):
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, temperature=1.0):
        super().__init__(env, learning_rate, discount_factor, temperature=temperature)
    
    def train(self, num_episodes, max_steps):
        episode_rewards = []
        
        for episode in tqdm(range(num_episodes), desc="MiniGrid Q-Learning Training"):
            state, _ = self.env.reset()
            
            episode_reward = 0
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < max_steps:
                action = self.softmax_policy(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Q-Learning update rule
                current_q = self.get_q_value(state, action)
                
                # Get max Q-value for next state
                state_disc = self.discretize_observation(next_state)
                next_max_q = np.max(self.q_table[state_disc])
                
                new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
                self.set_q_value(state, action, new_q)
                
                state = next_state
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
        
        return episode_rewards

def run_minigrid_experiments():
    """Run experiments for MiniGrid environment"""
    try:
        print("\n=== Starting MiniGrid-Dynamic-Obstacles-5x5-v0 Experiments (Bonus) ===")
        
        # Create and run MiniGrid environment
        env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0')
        
        # Define seeds
        seeds = [42, 123, 456, 789, 101112]
        
        # Define hyperparameters
        minigrid_sarsa_params = {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 0.2
        }
        
        minigrid_qlearning_params = {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'temperature': 1.0
        }
        
        # Number of episodes for training
        minigrid_episodes = 300
        max_steps = 100
        
        # Train SARSA agent
        print("Training SARSA agent on MiniGrid...")
        minigrid_sarsa_rewards = []
        
        for seed in seeds:
            np.random.seed(seed)
            env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0')
            agent = MiniGridSARSAAgent(
                env,
                learning_rate=minigrid_sarsa_params['learning_rate'],
                discount_factor=minigrid_sarsa_params['discount_factor'],
                epsilon=minigrid_sarsa_params['epsilon']
            )
            rewards = agent.train(minigrid_episodes, max_steps)
            minigrid_sarsa_rewards.append(rewards)
        
        # Train Q-Learning agent
        print("Training Q-Learning agent on MiniGrid...")
        minigrid_qlearning_rewards = []
        
        for seed in seeds:
            np.random.seed(seed)
            env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0')
            agent = MiniGridQLearningAgent(
                env,
                learning_rate=minigrid_qlearning_params['learning_rate'],
                discount_factor=minigrid_qlearning_params['discount_factor'],
                temperature=minigrid_qlearning_params['temperature']
            )
            rewards = agent.train(minigrid_episodes, max_steps)
            minigrid_qlearning_rewards.append(rewards)
        
        # Plot results
        minigrid_sarsa_df = pd.DataFrame({
            'episode': range(1, minigrid_episodes + 1),
            'reward': np.mean(minigrid_sarsa_rewards, axis=0),
            'std': np.std(minigrid_sarsa_rewards, axis=0)
        })
        
        minigrid_qlearning_df = pd.DataFrame({
            'episode': range(1, minigrid_episodes + 1),
            'reward': np.mean(minigrid_qlearning_rewards, axis=0),
            'std': np.std(minigrid_qlearning_rewards, axis=0)
        })
        
        # Plot comparison
        plt.figure(figsize=(12, 7))
        plt.plot(minigrid_sarsa_df['episode'], minigrid_sarsa_df['reward'], 
                label='SARSA Mean', color='blue')
        plt.fill_between(
            minigrid_sarsa_df['episode'],
            minigrid_sarsa_df['reward'] - minigrid_sarsa_df['std'],
            minigrid_sarsa_df['reward'] + minigrid_sarsa_df['std'],
            alpha=0.2,
            color='blue'
        )
        
        plt.plot(minigrid_qlearning_df['episode'], minigrid_qlearning_df['reward'], 
                label='Q-Learning Mean', color='red')
        plt.fill_between(
            minigrid_qlearning_df['episode'],
            minigrid_qlearning_df['reward'] - minigrid_qlearning_df['std'],
            minigrid_qlearning_df['reward'] + minigrid_qlearning_df['std'],
            alpha=0.2,
            color='red'
        )
        
        plt.title('SARSA vs Q-Learning in MiniGrid-Dynamic-Obstacles-5x5-v0')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        
        # Create results directory if it doesn't exist
        os.makedirs("results/MiniGrid", exist_ok=True)
        plt.savefig("results/MiniGrid/minigrid_comparison.png")
        
        return {
            'sarsa_df': minigrid_sarsa_df,
            'qlearning_df': minigrid_qlearning_df,
            'sarsa_params': minigrid_sarsa_params,
            'qlearning_params': minigrid_qlearning_params
        }
        
    except Exception as e:
        print(f"Could not run MiniGrid experiments: {e}")
        return None 