import numpy as np
import gymnasium as gym
from tqdm import tqdm

class RLAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, temperature=1.0, bins=20, low_val=-4, high_val=4):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.temperature = temperature
        self.bins = bins  
        self.low = low_val
        self.high = high_val
        
        # Initialize Q-table
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # If the observation space is discrete
        if isinstance(self.observation_space, gym.spaces.Discrete):
            self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
        else:
            # For continuous observation spaces, we'll discretize
            self.discrete_obs_size = [self.bins] * self.observation_space.shape[0]
            self.q_table = np.zeros((*self.discrete_obs_size, self.action_space.n))
    
    def discretize_observation(self, observation):
        if isinstance(self.observation_space, gym.spaces.Discrete):
            return observation
        else:
            # Scale observation to be between 0 and bins-1
            scaled_obs = []
            for i, o in enumerate(observation):
                low = self.observation_space.low[i]/2
                high = self.observation_space.high[i]/2
                
                # Handle edge cases for infinity
                if low == -np.inf:
                    low = self.low 
                if high == np.inf:
                    high = self.high  
                
                scaled_val = int(self.bins * (o - low) / (high - low))
                scaled_val = min(self.bins - 1, max(0, scaled_val))
                scaled_obs.append(scaled_val)
            
            return tuple(scaled_obs)
    
    def get_q_value(self, state, action):
        state_disc = self.discretize_observation(state)
        if isinstance(self.observation_space, gym.spaces.Discrete):
            return self.q_table[state_disc, action]
        else:
            return self.q_table[(*state_disc, action)]
    
    def set_q_value(self, state, action, value):
        state_disc = self.discretize_observation(state)
        if isinstance(self.observation_space, gym.spaces.Discrete):
            self.q_table[state_disc, action] = value
        else:
            self.q_table[(*state_disc, action)] = value
    
    def epsilon_greedy_policy(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            state_disc = self.discretize_observation(state)
            if isinstance(self.observation_space, gym.spaces.Discrete):
                return np.argmax(self.q_table[state_disc, :])
            else:
                return np.argmax(self.q_table[(*state_disc, )])
    
    def softmax_policy(self, state):
        state_disc = self.discretize_observation(state)
        
        if isinstance(self.observation_space, gym.spaces.Discrete):
            q_values = self.q_table[state_disc, :]
        else:
            q_values = self.q_table[(*state_disc, )]
        
        # Apply softmax with temperature
        # Normalize q_values to avoid overflow/underflow
        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values / self.temperature)
        probabilities = exp_q / (np.sum(exp_q) + 1e-10)  # Add small epsilon to prevent division by zero
        
        # Choose action based on probabilities
        return np.random.choice(self.action_space.n, p=probabilities)
    
    def train(self, num_episodes, max_steps):
        # To be implemented by subclasses
        pass
    
    def test(self, num_episodes):
        total_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Use epsilon_greedy_policy if epsilon exists, otherwise use softmax_policy
                action = self.epsilon_greedy_policy(state) if hasattr(self, 'epsilon') else self.softmax_policy(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards) 