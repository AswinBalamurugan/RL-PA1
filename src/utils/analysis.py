import numpy as np
import matplotlib.pyplot as plt

def analyze_results(results_dict, env_name):
    """
    Analyze and visualize experiment results
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
    
    # Plot learning curves with confidence intervals
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # SARSA
    sarsa_grouped = sarsa_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
    ax.plot(sarsa_grouped['episode'], sarsa_grouped['mean'], label='SARSA', color='blue')
    ax.fill_between(
        sarsa_grouped['episode'],
        sarsa_grouped['mean'] - sarsa_grouped['std'],
        sarsa_grouped['mean'] + sarsa_grouped['std'],
        alpha=0.2,
        color='blue'
    )
    
    # Q-Learning
    qlearning_grouped = qlearning_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
    ax.plot(qlearning_grouped['episode'], qlearning_grouped['mean'], label='Q-Learning', color='red')
    ax.fill_between(
        qlearning_grouped['episode'],
        qlearning_grouped['mean'] - qlearning_grouped['std'],
        qlearning_grouped['mean'] + qlearning_grouped['std'],
        alpha=0.2,
        color='red'
    )
    
    if sarsa_shaped_df is not None and qlearning_shaped_df is not None:
        # SARSA with shaping
        sarsa_shaped_grouped = sarsa_shaped_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
        ax.plot(sarsa_shaped_grouped['episode'], sarsa_shaped_grouped['mean'], 
               label='SARSA with Shaping', color='green')
        ax.fill_between(
            sarsa_shaped_grouped['episode'],
            sarsa_shaped_grouped['mean'] - sarsa_shaped_grouped['std'],
            sarsa_shaped_grouped['mean'] + sarsa_shaped_grouped['std'],
            alpha=0.2,
            color='green'
        )
        
        # Q-Learning with shaping
        qlearning_shaped_grouped = qlearning_shaped_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
        ax.plot(qlearning_shaped_grouped['episode'], qlearning_shaped_grouped['mean'], 
               label='Q-Learning with Shaping', color='purple')
        ax.fill_between(
            qlearning_shaped_grouped['episode'],
            qlearning_shaped_grouped['mean'] - qlearning_shaped_grouped['std'],
            qlearning_shaped_grouped['mean'] + qlearning_shaped_grouped['std'],
            alpha=0.2,
            color='purple'
        )
    
    ax.set_title(f'Learning Curves for {env_name}', fontsize=16)
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"results/Analysis/{env_name.replace('-', '_')}_analysis.png")
    plt.show()
    
    return {
        'sarsa_avg': sarsa_avg,
        'qlearning_avg': qlearning_avg,
        'sarsa_final': sarsa_final,
        'qlearning_final': qlearning_final
    } 