import numpy as np


def shape_reward_cartpole(state, reward, done):
    """
    Simplified reward shaping for CartPole-v1 focusing on:
    - Position (x): Reward for staying centered
    - Angle (theta): Reward for keeping pole upright
    - Angular velocity (theta_dot): Penalty for fast rotation
    """
    x, x_dot, theta, theta_dot = state

    # Position reward: higher when cart is closer to center
    position_reward = -0.1 * abs(x)

    # Angle reward: higher when pole is more upright
    angle_reward = np.cos(theta)  # Maximum (1.0) when perfectly upright

    # Angular velocity reward: penalize fast rotation
    angular_velocity_penalty = -0.1 * abs(theta_dot)

    # Termination penalty
    if done and reward == 0:
        return -5

    return reward + position_reward + angle_reward + angular_velocity_penalty


def shape_reward_mountaincar(state, reward, done):
    """
    Improved reward shaping for MountainCar-v0 focusing on energy maximization
    and progress toward the goal.
    """
    position, velocity = state

    # Potential-based reward: higher positions are better
    position_reward = 10 * (position + 1.2)  # Shifted to be positive

    # Energy reward: reward for having more total energy (height + velocity)
    # This is physically motivated by the task dynamics
    energy_reward = 5 * (position + 0.5 * velocity**2)

    # Progress reward: reward for moving right with high positive velocity
    # and for moving left with high negative velocity when in left valley
    if position < -0.5:
        progress_reward = (
            -5 * position * velocity
        )  # Rewards left movement in left valley
    else:
        progress_reward = 5 * velocity  # Rewards right movement elsewhere

    # Large reward for reaching the goal
    if done and position >= 0.5:
        return 100

    return reward + position_reward + energy_reward + progress_reward
