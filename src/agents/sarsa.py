from tqdm import tqdm
from .base import RLAgent
import os
import numpy as np
from collections import defaultdict
import gym


class SARSAAgent(RLAgent):
    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
        bins=20,
        low=None,
        high=None,
        scale_observations=False,
    ):
        super().__init__(
            env,
            learning_rate,
            discount_factor,
            epsilon,
            bins=bins,
            low_val=low,
            high_val=high,
            scale_observations=scale_observations,
        )
        self.initial_epsilon = epsilon  # Store initial epsilon for decay
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
        episode_rewards = []
        epsilon_decay_rate = 2 / num_episodes  # Calculate decay rate dynamically

        for episode in tqdm(range(num_episodes), desc="SARSA Training"):
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

                new_q = current_q + self.learning_rate * (
                    reward + self.discount_factor * next_q - current_q
                )
                self.set_q_value(state, action, new_q)

                state = next_state
                action = next_action
                episode_reward += reward
                step += 1

            # Decay epsilon
            self.epsilon = max(
                self.initial_epsilon - epsilon_decay_rate * episode, 1e-10
            )

            episode_rewards.append(episode_reward)

        return episode_rewards


class SARSAAgentWithShaping(SARSAAgent):
    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
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
            epsilon,
            bins=bins,
            low=low,
            high=high,
            scale_observations=scale_observations,
        )
        self.reward_shaping_fn = reward_shaping_fn

    def train(self, num_episodes, max_steps):
        episode_rewards = []
        epsilon_decay_rate = 2 / num_episodes  # Calculate decay rate dynamically

        for episode in tqdm(range(num_episodes), desc="SARSA with Shaping Training"):
            state, _ = self.env.reset()
            action = self.epsilon_greedy_policy(state)

            episode_reward = 0
            done = False
            truncated = False
            step = 0

            while not (done or truncated) and step < max_steps:
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_action = self.epsilon_greedy_policy(next_state)

                # Apply reward shaping if specified
                if self.reward_shaping_fn:
                    shaped_reward = self.reward_shaping_fn(next_state, reward, done)
                else:
                    shaped_reward = reward

                # SARSA update rule
                current_q = self.get_q_value(state, action)
                next_q = self.get_q_value(next_state, next_action)

                new_q = current_q + self.learning_rate * (
                    shaped_reward + self.discount_factor * next_q - current_q
                )
                self.set_q_value(state, action, new_q)

                state = next_state
                action = next_action
                episode_reward += reward  # Track original reward for reporting
                step += 1

            # Decay epsilon
            self.epsilon = max(
                self.initial_epsilon - epsilon_decay_rate * episode, 1e-10
            )

            episode_rewards.append(episode_reward)

        return episode_rewards
