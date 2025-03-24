import os
import numpy as np
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

def analyze_results(results_dict, env_name):
    """
    Analyze and visualize experiment results with enhanced plotting
    """
    print(f"\n=== Analysis for {env_name} ===")
    
    # Extract dataframes
    sarsa_df = results_dict['sarsa_df']
    sarsa_shaped_df = results_dict.get('sarsa_shaped_df')
    qlearning_df = results_dict['qlearning_df']
    qlearning_shaped_df = results_dict.get('qlearning_shaped_df')
    
    # Calculate statistics
    sarsa_avg = sarsa_df.groupby('episode')['reward'].mean().mean()
    qlearning_avg = qlearning_df.groupby('episode')['reward'].mean().mean()
    
    sarsa_final = sarsa_df[sarsa_df['episode'] > sarsa_df['episode'].max() * 0.9]['reward'].mean()
    qlearning_final = qlearning_df[qlearning_df['episode'] > qlearning_df['episode'].max() * 0.9]['reward'].mean()
    
    # Print statistics
    print(f"SARSA Average Reward: {sarsa_avg:.2f}")
    print(f"Q-Learning Average Reward: {qlearning_avg:.2f}")
    print(f"SARSA Final 10% Average: {sarsa_final:.2f}")
    print(f"Q-Learning Final 10% Average: {qlearning_final:.2f}")
    
    if sarsa_shaped_df is not None and qlearning_shaped_df is not None:
        sarsa_shaped_avg = sarsa_shaped_df.groupby('episode')['reward'].mean().mean()
        qlearning_shaped_avg = qlearning_shaped_df.groupby('episode')['reward'].mean().mean()
        
        sarsa_shaped_final = sarsa_shaped_df[sarsa_shaped_df['episode'] > sarsa_shaped_df['episode'].max() * 0.9]['reward'].mean()
        qlearning_shaped_final = qlearning_shaped_df[qlearning_shaped_df['episode'] > qlearning_shaped_df['episode'].max() * 0.9]['reward'].mean()
        
        print(f"SARSA with Shaping Average Reward: {sarsa_shaped_avg:.2f}")
        print(f"Q-Learning with Shaping Average Reward: {qlearning_shaped_avg:.2f}")
        print(f"SARSA with Shaping Final 10% Average: {sarsa_shaped_final:.2f}")
        print(f"Q-Learning with Shaping Final 10% Average: {qlearning_shaped_final:.2f}")
    
    # Enhanced plotting
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)  # Larger figure size and higher DPI
    
    # Plot with enhanced styling
    # SARSA
    sarsa_grouped = sarsa_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
    ax.plot(sarsa_grouped['episode'], sarsa_grouped['mean'], 
            label='SARSA', color='#2E86C1', linewidth=2)
    ax.fill_between(
        sarsa_grouped['episode'],
        sarsa_grouped['mean'] - sarsa_grouped['std'],
        sarsa_grouped['mean'] + sarsa_grouped['std'],
        alpha=0.15, color='#2E86C1'
    )
    
    # Q-Learning
    qlearning_grouped = qlearning_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
    ax.plot(qlearning_grouped['episode'], qlearning_grouped['mean'], 
            label='Q-Learning', color='#E74C3C', linewidth=2)
    ax.fill_between(
        qlearning_grouped['episode'],
        qlearning_grouped['mean'] - qlearning_grouped['std'],
        qlearning_grouped['mean'] + qlearning_grouped['std'],
        alpha=0.15, color='#E74C3C'
    )
    
    if sarsa_shaped_df is not None and qlearning_shaped_df is not None:
        # SARSA with shaping
        sarsa_shaped_grouped = sarsa_shaped_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
        ax.plot(sarsa_shaped_grouped['episode'], sarsa_shaped_grouped['mean'],
                label='SARSA with Shaping', color='#27AE60', linewidth=2)
        ax.fill_between(
            sarsa_shaped_grouped['episode'],
            sarsa_shaped_grouped['mean'] - sarsa_shaped_grouped['std'],
            sarsa_shaped_grouped['mean'] + sarsa_shaped_grouped['std'],
            alpha=0.15, color='#27AE60'
        )
        
        # Q-Learning with shaping
        qlearning_shaped_grouped = qlearning_shaped_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
        ax.plot(qlearning_shaped_grouped['episode'], qlearning_shaped_grouped['mean'],
                label='Q-Learning with Shaping', color='#8E44AD', linewidth=2)
        ax.fill_between(
            qlearning_shaped_grouped['episode'],
            qlearning_shaped_grouped['mean'] - qlearning_shaped_grouped['std'],
            qlearning_shaped_grouped['mean'] + qlearning_shaped_grouped['std'],
            alpha=0.15, color='#8E44AD'
        )
    
    # Enhanced styling
    ax.set_title(f'Learning Curves for {env_name}', fontsize=16, pad=20)
    ax.set_xlabel('Episode', fontsize=14, labelpad=10)
    ax.set_ylabel('Average Reward', fontsize=14, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Enhanced legend
    ax.legend(
        loc='center right',
        bbox_to_anchor=(1.25, 0.5),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save with high quality
    os.makedirs(f"results/Analysis", exist_ok=True)
    plt.savefig(
        f"results/Analysis/{env_name.replace('-', '_')}_analysis.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()
    
    return {
        'sarsa_avg': sarsa_avg,
        'qlearning_avg': qlearning_avg,
        'sarsa_final': sarsa_final,
        'qlearning_final': qlearning_final
    } 