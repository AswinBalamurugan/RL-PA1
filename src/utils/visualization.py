import os
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

plt.style.use(['ieee','no-latex'])
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 20
sns.set_style('dark')

def plot_results(results_df, env_name, agent_name):
    """
    Plot mean and variance of rewards across seeds
    """
    plt.figure(figsize=(10, 6))
    
    # Group by episode and calculate mean and std
    grouped = results_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
    
    plt.plot(grouped['episode'], grouped['mean'], label=f'{agent_name} Mean')
    plt.fill_between(
        grouped['episode'],
        grouped['mean'] - grouped['std'],
        grouped['mean'] + grouped['std'],
        alpha=0.2
    )
    
    plt.title(f'{agent_name} in {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Create results directory if it doesn't exist
    os.makedirs(f"results/{env_name}/individual", exist_ok=True)
    plot_path = f"results/{env_name}/individual/{agent_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path)
    plt.close()

def compare_algorithms(sarsa_df, qlearning_df, env_name):
    """
    Create comparative plot of SARSA vs Q-Learning
    """
    plt.figure(figsize=(12, 7))
    
    # SARSA results
    sarsa_grouped = sarsa_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
    plt.plot(sarsa_grouped['episode'], sarsa_grouped['mean'], label='SARSA Mean', color='blue')
    plt.fill_between(
        sarsa_grouped['episode'],
        sarsa_grouped['mean'] - sarsa_grouped['std'],
        sarsa_grouped['mean'] + sarsa_grouped['std'],
        alpha=0.2,
        color='blue'
    )
    
    # Q-Learning results
    qlearning_grouped = qlearning_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
    plt.plot(qlearning_grouped['episode'], qlearning_grouped['mean'], label='Q-Learning Mean', color='red')
    plt.fill_between(
        qlearning_grouped['episode'],
        qlearning_grouped['mean'] - qlearning_grouped['std'],
        qlearning_grouped['mean'] + qlearning_grouped['std'],
        alpha=0.2,
        color='red'
    )
    
    plt.title(f'SARSA vs Q-Learning in {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Create comparison directory if it doesn't exist
    os.makedirs(f"results/{env_name}", exist_ok=True)
    
    # Save comparison plot
    plot_path = f"results/{env_name}/comparison.png"
    plt.savefig(plot_path)
    plt.close() 