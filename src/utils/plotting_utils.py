import matplotlib.pyplot as plt
import numpy as np
import os


def plot_learning_curve(
    episode_rewards,
    window_size=100,
    title="Learning Curve",
    xlabel="Episode",
    ylabel="Reward",
    save_path=None
):
    episodes = np.arange(len(episode_rewards))
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(
            episode_rewards, 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        moving_avg_episodes = episodes[window_size-1:]
    else:
        moving_avg = episode_rewards
        moving_avg_episodes = episodes
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw rewards')
    plt.plot(moving_avg_episodes, moving_avg, color='blue', linewidth=2, 
             label=f'Moving avg (window={window_size})')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_comparison(
    rewards_dict,
    window_size=100,
    title="Algorithm Comparison",
    save_path=None
):
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (name, rewards) in enumerate(rewards_dict.items()):
        episodes = np.arange(len(rewards))
        if len(rewards) >= window_size:
            moving_avg = np.convolve(
                rewards, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            moving_avg_episodes = episodes[window_size-1:]
        else:
            moving_avg = rewards
            moving_avg_episodes = episodes
        
        color = colors[idx % len(colors)]
        plt.plot(episodes, rewards, alpha=0.2, color=color)
        plt.plot(moving_avg_episodes, moving_avg, color=color, 
                linewidth=2, label=name)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_aggregated_learning_curve(
    all_rewards,
    config,
    num_episodes,
    num_runs,
    seeds,
    window_size=50,
    title_prefix="Aggregated Learning Curve",
    save_path=None
):
    smoothed_rewards = []
    for rewards in all_rewards:
        if len(rewards) >= window_size:
            moving_avg = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            smoothed_rewards.append(moving_avg)
    smoothed_array = np.array(smoothed_rewards)  # Shape: (num_runs, num_episodes - window_size + 1)
    mean_rewards = np.mean(smoothed_array, axis=0)
    std_rewards = np.std(smoothed_array, axis=0)
    episodes = np.arange(window_size, num_episodes + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, mean_rewards, color='#1f77b4', linewidth=2.5,
             label=f'Mean reward (MA window={window_size})')
    plt.fill_between(episodes,
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     alpha=0.3, color='#1f77b4', label='Std across runs')
    lr_policy_str = f"{config['lr_policy']:.0e}".replace('e-0', 'e-').replace('e-', 'e-0')
    lr_value_str = f"{config['lr_value']:.0e}".replace('e-0', 'e-').replace('e-', 'e-0')
    seeds_str = str(seeds).replace(' ', '')

    title = (f"{title_prefix}\n"
             f"α={lr_policy_str}, critic_lr={lr_value_str}, γ={config['gamma']}, "
             f"episodes={num_episodes}, runs={num_runs}, seeds={seeds_str}")

    plt.title(title, fontsize=14)
    plt.xlabel('Episode', fontsize=13)
    plt.ylabel(f'Total Episode Reward ({window_size}-ep MA)', fontsize=13)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Aggregated learning curve saved to {save_path}")
    else:
        plt.show()

    plt.close()