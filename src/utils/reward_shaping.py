import numpy as np

def shape_reward_cartpole(state, reward, done):
    """
    Custom reward shaping for CartPole-v1
    
    The goal is to keep the pole upright, so we can reshape the reward
    to give higher values when the pole is more vertical.
    """
    x, x_dot, theta, theta_dot = state
    
    # Additional reward for keeping the pole upright
    # theta is the angle, 0 means perfectly upright
    upright_reward = np.cos(theta)
    
    # Penalize being far from the center
    center_reward = -0.1 * abs(x)
    
    # Heavily penalize falling (done == True and reward == 0)
    if done and reward == 0:
        return -10
    
    return reward + upright_reward + center_reward

def shape_reward_mountaincar(state, reward, done):
    """
    Custom reward shaping for MountainCar-v0
    
    The original reward is -1 for each timestep. We can add reward
    based on the car's position and velocity to encourage progress.
    """
    position, velocity = state
    
    # Reward for height (position)
    height_reward = 10 * position  # Higher position = better
    
    # Reward for moving in the right direction with positive velocity
    velocity_reward = 5 * velocity if velocity > 0 else 0
    
    # Large reward for reaching the goal
    if done and position >= 0.5:
        return 100
    
    return reward + height_reward + velocity_reward 