import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from minigrid.wrappers import FullyObsWrapper
import time
import os
from tqdm import tqdm

PATH_Q_TABLE_SARSA = os.path.join(
    "src", "experiments", "minigrid_5x5", "sarsa_q_table.npy"
)
PATH_Q_TABLE_QLEARNING = os.path.join(
    "src", "experiments", "minigrid_5x5", "qlearning_q_table.npy"
)


class QLearningAgent:
    def __init__(
        self,
        action_space=3,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
    ):
        # action_space: 0=turn_left, 1=turn_right, 2=move_forward
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        # Add a small bias toward moving forward
        self.action_bias = np.array([0.0, 0.0, 0.1])

    def get_state_representation(self, env):
        """Convert the environment observation into a hashable state tuple"""
        grid, agent_pos = env.unwrapped.grid.encode(), env.unwrapped.agent_pos
        agent_dir = env.unwrapped.agent_dir  # 0: right, 1: down, 2: left, 3: up
        goal_pos = None
        obstacles = []

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                obj_type = grid[x, y, 0]
                if obj_type == 2:  # Goal
                    goal_pos = (x, y)
                elif obj_type == 6:  # Dynamic Obstacle
                    obstacles.append((x, y))

        # Sort obstacles for consistent state representation
        obstacles.sort()

        # Calculate relative positions to agent for better generalization
        rel_goal = (
            (goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1])
            if goal_pos
            else (0, 0)
        )

        # Convert obstacles to relative positions
        rel_obstacles = []
        for obs in obstacles:
            rel_obstacles.append((obs[0] - agent_pos[0], obs[1] - agent_pos[1]))

        # Ensure fixed number of obstacles in state representation (pad if needed)
        max_obstacles = 3
        while len(rel_obstacles) < max_obstacles:
            rel_obstacles.append((99, 99))  # Use a value outside grid as padding

        # Keep only necessary elements to match your original state representation
        state = (
            agent_pos[0],
            agent_pos[1],
            agent_dir,
            rel_goal[0],
            rel_goal[1],
            rel_obstacles[0][0],
            rel_obstacles[0][1],
            rel_obstacles[1][0],
            rel_obstacles[1][1],
        )

        return state

    def select_action(self, state):
        """Select action using epsilon-greedy policy with a bias toward moving forward"""
        if random.random() < self.exploration_rate:
            q_values = self.q_table[state] + self.action_bias
            exp_values = np.exp(q_values - np.max(q_values))
            probabilities = exp_values / np.sum(exp_values)
            return np.random.choice(self.action_space, p=probabilities)
        else:
            q_values = self.q_table[state] + self.action_bias
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, done, steps_since_last_move):
        """Update Q-value using the Q-learning update rule with enhanced rewards"""
        # Modify reward: greatly increase goal reward, penalize excessive turning
        modified_reward = reward

        # HUGE reward for reaching the goal (original reward is 1.0)
        if reward > 0:  # Goal reached
            modified_reward = 10.0  # Dramatically increase goal reward

        # Slightly penalize turns, especially if we've been turning without moving
        if action in [0, 1]:  # Turn actions
            # Increasing penalty for consecutive turns without moving
            turn_penalty = -0.01 * min(steps_since_last_move, 5)  # Cap the penalty
            modified_reward += turn_penalty

        # Standard Q-learning update with modified reward
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = modified_reward + (
            0
            if done
            else self.discount_factor * self.q_table[next_state][best_next_action]
        )
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error

    def train(self, env, episodes=5000, max_steps=100):
        """Train the agent with mechanisms to avoid turn oscillation"""
        episode_rewards = []
        episode_steps = []
        success_count = 0

        for episode in tqdm(range(episodes), desc="Training Q-Learning Agent"):
            obs, _ = env.reset()
            state = self.get_state_representation(env)
            done = False
            total_reward = 0
            step = 0

            # Track consecutive turns to detect oscillation
            steps_since_last_move = 0
            last_positions = []

            while not done and step < max_steps:
                action = self.select_action(state)
                _, reward, done, _, _ = env.step(action)
                next_state = self.get_state_representation(env)

                # Track agent position to detect if it's actually moving
                current_pos = (next_state[0], next_state[1])

                # Update steps_since_last_move counter
                if action == 2:  # Move forward
                    steps_since_last_move = 0
                    last_positions.append(current_pos)
                    # Only keep last 3 positions
                    if len(last_positions) > 3:
                        last_positions.pop(0)
                else:  # Turn action
                    steps_since_last_move += 1

                # Update Q-value with enhanced rewards
                self.update(
                    state, action, reward, next_state, done, steps_since_last_move
                )

                state = next_state
                total_reward += reward
                step += 1

                # If goal reached, count as success
                if reward > 0:
                    success_count += 1

            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay,
            )

            episode_rewards.append(total_reward)
            episode_steps.append(step)

            # # Print progress every 100 episodes
            # if episode % 100 == 0:
            #     avg_reward = sum(episode_rewards[-100:]) / min(
            #         100, len(episode_rewards)
            #     )
            #     avg_steps = sum(episode_steps[-100:]) / min(100, len(episode_steps))
            #     success_rate = success_count / min(100, episode + 1) * 100
            #     success_count = 0  # Reset counter

            #     # print(
            #     #     f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, "
            #     #     f"Success Rate: {success_rate:.1f}%, Exploration: {self.exploration_rate:.4f}"
            #     # )

            # Save Q-table periodically
            if episode % 1000 == 0:
                self.save_q_table(PATH_Q_TABLE_QLEARNING)

        # Save final Q-table
        self.save_q_table(PATH_Q_TABLE_QLEARNING)

        return episode_rewards, episode_steps

    def save_q_table(self, path=PATH_Q_TABLE_QLEARNING):
        """Save Q-table to file at the specified path including filename"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        q_dict = dict(self.q_table)
        np.save(path, q_dict)
        # print(f"Q-table saved to {path}")

    def load_q_table(self, filename=PATH_Q_TABLE_QLEARNING):
        """Load Q-table from file"""
        if os.path.exists(filename):
            loaded_q = np.load(filename, allow_pickle=True).item()
            # Convert dict to defaultdict
            self.q_table = defaultdict(lambda: np.zeros(self.action_space))
            for k, v in loaded_q.items():
                self.q_table[k] = v
            print(f"Q-table loaded from {filename}")
            return True
        return False


class SARSAAgent:
    def __init__(
        self,
        action_space=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    ):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        # Initialize action bias to encourage moving forward
        self.action_bias = np.array(
            [0.0, 0.0, 0.1]
        )  # Small bias for action 2 (move forward)

    def get_state_representation(self, env):
        """Convert the environment observation into a hashable state tuple"""
        grid, agent_pos = env.unwrapped.grid.encode(), env.unwrapped.agent_pos
        agent_dir = env.unwrapped.agent_dir
        goal_pos = None
        obstacles = []

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                obj_type = grid[x, y, 0]
                if obj_type == 2:  # Goal
                    goal_pos = (x, y)
                elif obj_type == 6:  # Dynamic Obstacle
                    obstacles.append((x, y))

        obstacles.sort()
        rel_goal = (
            (goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1])
            if goal_pos
            else (0, 0)
        )
        rel_obstacles = [
            (obs[0] - agent_pos[0], obs[1] - agent_pos[1]) for obs in obstacles
        ]

        # Pad obstacles to ensure fixed state size
        max_obstacles = 3
        while len(rel_obstacles) < max_obstacles:
            rel_obstacles.append((99, 99))

        state = (
            agent_pos[0],
            agent_pos[1],
            agent_dir,
            rel_goal[0],
            rel_goal[1],
            *rel_obstacles[0],
            *rel_obstacles[1],
        )
        return state

    def select_action(self, state):
        """Select action using epsilon-greedy policy with a bias toward moving forward"""
        if random.random() < self.epsilon:
            # During exploration, slightly bias toward moving forward
            if random.random() < 0.6:  # 60% chance to move forward during exploration
                return 2  # Move forward
            else:
                return random.choice([0, 1])  # Turn left or right
        else:
            # Use the action bias to encourage moving forward when Q-values are similar
            q_values = self.q_table[state] + self.action_bias
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action, done):
        """Update Q-value using the SARSA update rule"""
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action] if not done else 0
        td_target = reward + self.discount_factor * next_q
        td_error = td_target - current_q
        self.q_table[state][action] += self.learning_rate * td_error

    def train(self, env, episodes=10000, max_steps=100):
        """Train the agent and return episode rewards and steps"""
        episode_rewards = []
        episode_steps = []

        for episode in tqdm(range(episodes), desc="Training SARSA Agent"):
            obs, _ = env.reset()
            state = self.get_state_representation(env)
            action = self.select_action(state)
            done = False
            total_reward = 0
            step = 0

            while not done and step < max_steps:
                _, reward, done, _, _ = env.step(action)
                next_state = self.get_state_representation(env)
                next_action = self.select_action(next_state)
                self.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                total_reward += reward
                step += 1

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            episode_rewards.append(total_reward)
            episode_steps.append(step)

        # Save final Q-table
        self.save_q_table(PATH_Q_TABLE_SARSA)
        return episode_rewards, episode_steps

    def save_q_table(self, filename=PATH_Q_TABLE_SARSA):
        """Save Q-table to file"""
        np.save(filename, dict(self.q_table))

    def load_q_table(self, filename=PATH_Q_TABLE_SARSA):
        """Load Q-table from file"""
        if os.path.exists(filename):
            self.q_table = defaultdict(
                lambda: np.zeros(self.action_space),
                np.load(filename, allow_pickle=True).item(),
            )
            print(f"Q-table loaded from {filename}")
            return True
        return False
