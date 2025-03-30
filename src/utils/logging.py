import wandb
import numpy as np


def setup_wandb(project_name="rl_assignment1"):
    """
    Initialize wandb for experiment tracking
    """
    try:
        wandb.login()
        return True
    except Exception as e:
        print(f"Could not initialize wandb: {e}")
        print("Continuing without experiment tracking.")
        return False


def log_to_wandb(agent_name, env_name, hyperparams, episode_rewards, seed):
    """
    Log experiment results to wandb
    """
    try:
        # Initialize run
        run = wandb.init(
            project="rl_assignment1",
            name=f"{agent_name}_{env_name}_seed{seed}",
            config=hyperparams,
            reinit=True,
        )

        # Log episodic rewards
        for i, reward in enumerate(episode_rewards):
            wandb.log({"episode": i + 1, "reward": reward})

        # Log summary metrics
        wandb.log(
            {
                "avg_reward_last_100": np.mean(episode_rewards[-100:]),
                "max_reward": np.max(episode_rewards),
                "total_episodes": len(episode_rewards),
            }
        )

        wandb.finish()
    except Exception as e:
        print(f"Error logging to wandb: {e}")
        pass
