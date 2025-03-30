import os
import numpy as np
import gymnasium as gym
from collections import defaultdict
import time
from src.agents.sarsa import SARSAAgent, SARSAAgentWithShaping
from src.agents.qlearning import QLearningAgent, QLearningAgentWithShaping

# Path to the Q-tables directory
Q_TABLES_BASE_DIR = os.path.join("src", "experiments", "q_tables")


def load_q_table_agent(env, agent_type):
    """Load the Q-table using the appropriate agent class."""
    if agent_type == "qlearningagent":
        agent = QLearningAgent(env=env)
    elif agent_type == "qlearningagentwithshaping":
        agent = QLearningAgentWithShaping(env=env)
    elif agent_type == "sarsaagent":
        agent = SARSAAgent(env=env)
    elif agent_type == "sarsaagentwithshaping":
        agent = SARSAAgentWithShaping(env=env)
    else:
        print(f"Unknown agent type: {agent_type}")
        return None

    # Load the Q-table using the agent's method
    if agent.load_q_table(
        os.path.join(Q_TABLES_BASE_DIR, env.spec.id, f"{agent_type}.npy")
    ):
        return agent.q_table
    else:
        return None


def evaluate_agent(env_name, agent_type, episodes=100):
    """Evaluate the agent over a specified number of episodes."""
    env = gym.make(env_name)
    q_table = load_q_table_agent(env, agent_type)
    if q_table is None:
        return

    total_rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Get the state from the observation
            if env_name == "CartPole-v1":
                cart_pos_idx = np.digitize(obs[0], np.linspace(-4.8, 4.8, 20)) - 1
                cart_vel_idx = np.digitize(obs[1], np.linspace(-3.0, 3.0, 20)) - 1
                pole_ang_idx = np.digitize(obs[2], np.linspace(-0.418, 0.418, 20)) - 1
                pole_vel_idx = np.digitize(obs[3], np.linspace(-2.0, 2.0, 20)) - 1
                state = (cart_pos_idx, cart_vel_idx, pole_ang_idx, pole_vel_idx)
            elif env_name == "MountainCar-v0":
                pos_idx = np.digitize(obs[0], np.linspace(-1.2, 0.6, 40)) - 1
                vel_idx = np.digitize(obs[1], np.linspace(-0.07, 0.07, 40)) - 1
                state = (pos_idx, vel_idx)

            # Choose the action with the highest Q-value
            action = np.argmax(q_table[state])

            # Take the action
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    print(f"\nEvaluating {env_name} with {agent_type} over {episodes} test episodes")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    env.close()


def render_environment(env_name, agent_type, episodes=3, max_steps=1000, delay=1e-10):
    """Render the environment in human mode using the Q-table."""
    env = gym.make(env_name, render_mode="human")
    q_table = load_q_table_agent(env, agent_type)
    if q_table is None:
        return

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode_reward = 0

        print(f"\nEpisode {episode + 1}")

        while not done and step_count < max_steps:
            # Get the state from the observation
            if env_name == "CartPole-v1":
                cart_pos_idx = np.digitize(obs[0], np.linspace(-4.8, 4.8, 20)) - 1
                cart_vel_idx = np.digitize(obs[1], np.linspace(-3.0, 3.0, 20)) - 1
                pole_ang_idx = np.digitize(obs[2], np.linspace(-0.418, 0.418, 20)) - 1
                pole_vel_idx = np.digitize(obs[3], np.linspace(-2.0, 2.0, 20)) - 1
                state = (cart_pos_idx, cart_vel_idx, pole_ang_idx, pole_vel_idx)
            elif env_name == "MountainCar-v0":
                pos_idx = np.digitize(obs[0], np.linspace(-1.2, 0.6, 40)) - 1
                vel_idx = np.digitize(obs[1], np.linspace(-0.07, 0.07, 40)) - 1
                state = (pos_idx, vel_idx)

            # Choose the action with the highest Q-value
            action = np.argmax(q_table[state])

            # Take the action
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1

            # Render the environment
            env.render()
            time.sleep(delay)

        print(f"Episode {episode + 1} completed with reward: {episode_reward}")

    env.close()


def render_all_environments():
    """Render all environments for MountainCar-v0 and CartPole-v1."""
    # environments = ["MountainCar-v0", "CartPole-v1"]
    environments = ["MountainCar-v0"]
    agent_types = [
        "qlearningagent",
        "qlearningagentwithshaping",
        "sarsaagent",
        "sarsaagentwithshaping",
    ]
    for env_name in environments:
        for agent_type in agent_types:
            print(f"\nRendering {env_name} with {agent_type}")
            render_environment(env_name, agent_type)


if __name__ == "__main__":
    render_all_environments()
