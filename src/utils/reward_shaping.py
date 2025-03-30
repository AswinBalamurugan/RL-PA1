import numpy as np


def shape_reward_cartpole(state, reward, done):
    """
    Reward shaping for CartPole-v1 environment that encourages:
    - Keeping the cart centered (x position near 0)
    - Maintaining the pole in an upright position (theta near 0)
    - Minimizing angular velocity (smoother balancing)
    - Minimizing cart velocity (smoother movement)
    """
    x, x_dot, theta, theta_dot = state

    POSITION_WEIGHT = 0.3
    ANGLE_WEIGHT = 0.5
    CART_VELOCITY_WEIGHT = 0.1
    POLE_VELOCITY_WEIGHT = 0.1
    STABILITY_WEIGHT = 0.3

    position_reward = POSITION_WEIGHT * np.exp(-2.0 * abs(x))
    angle_reward = ANGLE_WEIGHT * (np.cos(abs(theta)) ** 2)
    cart_velocity_penalty = -CART_VELOCITY_WEIGHT * (abs(x_dot) + 0.5 * x_dot**2)
    angular_velocity_penalty = -POLE_VELOCITY_WEIGHT * (
        abs(theta_dot) + 0.5 * theta_dot**2
    )

    stability_factor = np.exp(
        -0.5 * (abs(x) + abs(theta) + 0.5 * abs(x_dot) + 0.5 * abs(theta_dot))
    )
    stability_reward = STABILITY_WEIGHT * stability_factor

    shaped_reward = (
        reward
        + position_reward
        + angle_reward
        + cart_velocity_penalty
        + angular_velocity_penalty
        + stability_reward
    )

    if done and reward == 0:
        shaped_reward -= 5.0

    return shaped_reward


def shape_reward_mountaincar(state, reward, done):
    position, velocity = state

    shaped_reward = reward
    shaped_reward += 1.0 * abs(velocity)

    height = np.sin(3 * position) * 0.45 + 0.55
    kinetic_energy = 0.5 * velocity**2
    shaped_reward += 1.0 * (height + kinetic_energy)

    if abs(position) < 0.2 and abs(velocity) < 0.03:
        shaped_reward -= 0.5

    if position > 0 and velocity > 0:
        shaped_reward += 0.3 * velocity

    if done and position >= 0.5:
        shaped_reward += 100

    return shaped_reward
