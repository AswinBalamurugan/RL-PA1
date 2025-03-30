from tqdm import tqdm
from .base import RLAgent
import numpy as np
import gymnasium as gym
import os
from collections import defaultdict


class QLearningAgent(RLAgent):
    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        temperature=1.0,
        bins=20,
        low=None,
        high=None,
        scale_observations=False,
    ):
        super().__init__(
            env,
            learning_rate,
            discount_factor,
            temperature=temperature,
            bins=bins,
            low_val=low,
            high_val=high,
            scale_observations=scale_observations,
        )
        self.initial_temperature = temperature  # Store initial temperature for decay
        self.action_space = (
            env.action_space.n
        )  # Store the number of actions as an integer

    def save_q_table(self, path):
        """Save Q-table to file at the specified path including filename."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load_q_table(self, path):
        """Load Q-table from file."""
        if os.path.exists(path):
            self.q_table = np.load(path)
            return True
        return False

    def train(self, num_episodes, max_steps):
        """Train the Q-learning agent."""
        episode_rewards = []
        temperature_decay_rate = 2 / num_episodes  # Calculate decay rate dynamically

        for episode in tqdm(range(num_episodes), desc="Q-Learning Training"):
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
                if isinstance(self.observation_space, gym.spaces.Discrete):
                    next_max_q = np.max(self.q_table[state_disc, :])
                else:
                    next_max_q = np.max(self.q_table[(*state_disc,)])

                new_q = current_q + self.learning_rate * (
                    reward + self.discount_factor * next_max_q - current_q
                )
                self.set_q_value(state, action, new_q)

                state = next_state
                episode_reward += reward
                step += 1

            # Decay temperature
            self.temperature = max(
                self.initial_temperature - temperature_decay_rate * episode, 1e-10
            )

            episode_rewards.append(episode_reward)

        return episode_rewards


class QLearningAgentWithShaping(QLearningAgent):
    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        temperature=1.0,
        reward_shaping_fn=None,
        bins=20,
        low=None,
        high=None,
        scale_observations=False,
    ):
        super().__init__(
            env,
            learning_rate,
            discount_factor,
            temperature,
            bins=bins,
            low=low,
            high=high,
            scale_observations=scale_observations,
        )
        self.reward_shaping_fn = reward_shaping_fn

    def train(self, num_episodes, max_steps):
        """Train the Q-learning agent with reward shaping."""
        episode_rewards = []
        temperature_decay_rate = 2 / num_episodes  # Calculate decay rate dynamically

        for episode in tqdm(
            range(num_episodes), desc="Q-Learning with Shaping Training"
        ):
            state, _ = self.env.reset()

            episode_reward = 0
            done = False
            truncated = False
            step = 0

            while not (done or truncated) and step < max_steps:
                action = self.softmax_policy(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Apply reward shaping if specified
                if self.reward_shaping_fn:
                    shaped_reward = self.reward_shaping_fn(next_state, reward, done)
                else:
                    shaped_reward = reward

                # Q-Learning update rule
                current_q = self.get_q_value(state, action)

                # Get max Q-value for next state
                state_disc = self.discretize_observation(next_state)
                if isinstance(self.observation_space, gym.spaces.Discrete):
                    next_max_q = np.max(self.q_table[state_disc, :])
                else:
                    next_max_q = np.max(self.q_table[(*state_disc,)])

                new_q = current_q + self.learning_rate * (
                    shaped_reward + self.discount_factor * next_max_q - current_q
                )
                self.set_q_value(state, action, new_q)

                state = next_state
                episode_reward += reward  # Track original reward for reporting
                step += 1

            # Decay temperature
            self.temperature = max(
                self.initial_temperature - temperature_decay_rate * episode, 1e-10
            )

            episode_rewards.append(episode_reward)

        return episode_rewards
