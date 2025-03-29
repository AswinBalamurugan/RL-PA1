import gymnasium as gym
import numpy as np
import time
from collections import defaultdict
from minigrid.wrappers import FullyObsWrapper


def get_state_representation(env):
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
        (goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1]) if goal_pos else (0, 0)
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


# def visualize_agent(q_table_filename="q_table_symbolic.npy"):
#     # Action meanings for debugging
#     ACTION_MEANINGS = {0: "Turn Left", 1: "Turn Right", 2: "Move Forward"}

#     # Load Q-table
#     q_table = np.load(q_table_filename, allow_pickle=True).item()

#     # Create environment with human render mode
#     env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0", render_mode="human")
#     env = FullyObsWrapper(env)

#     # Initialize a blank defaultdict with the loaded Q-table
#     q_defaultdict = defaultdict(lambda: np.zeros(3))
#     for k, v in q_table.items():
#         q_defaultdict[k] = v

#     episodes = 5
#     max_steps = 100
#     delay = 0.3  # Slow down visualization

#     # Run episodes
#     for episode in range(episodes):
#         obs, _ = env.reset()
#         state = get_state_representation(env)
#         done = False
#         step_count = 0
#         episode_reward = 0
#         moves_taken = 0

#         print(f"\nEpisode {episode + 1}")

#         # Force render at the beginning of each episode
#         env.render()
#         time.sleep(1.0)  #


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
