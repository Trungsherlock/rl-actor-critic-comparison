
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
    """
    Plot learning curve with smoothed average
    
    Args:
        episode_rewards: List of rewards per episode
        window_size: Window for moving average smoothing
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (if None, just show)
    """
    episodes = np.arange(len(episode_rewards))
    
    # Compute moving average
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
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards (lighter)
    plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw rewards')
    
    # Plot moving average (darker)
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


def plot_multiple_seeds(
    all_rewards,
    labels=None,
    window_size=100,
    title="Learning Curves - Multiple Seeds",
    save_path=None
):
    """
    Plot learning curves from multiple random seeds
    
    Args:
        all_rewards: List of lists, each containing episode rewards
        labels: List of labels for each seed
        window_size: Window for moving average
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_rewards)))
    
    for idx, rewards in enumerate(all_rewards):
        episodes = np.arange(len(rewards))
        
        # Compute moving average
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
        
        label = labels[idx] if labels else f'Seed {idx}'
        
        # Plot raw (very light)
        plt.plot(episodes, rewards, alpha=0.15, color=colors[idx])
        
        # Plot moving average
        plt.plot(moving_avg_episodes, moving_avg, color=colors[idx], 
                linewidth=2, label=label)
    
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


def plot_comparison(
    rewards_dict,
    window_size=100,
    title="Algorithm Comparison",
    save_path=None
):
    """
    Plot comparison between different algorithms
    
    Args:
        rewards_dict: Dictionary {algorithm_name: [episode_rewards]}
        window_size: Window for moving average
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (name, rewards) in enumerate(rewards_dict.items()):
        episodes = np.arange(len(rewards))
        
        # Compute moving average
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
        
        # Plot raw (light)
        plt.plot(episodes, rewards, alpha=0.2, color=color)
        
        # Plot moving average
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


def plot_training_metrics(stats, save_path=None):
    """
    Plot multiple training metrics in subplots
    
    Args:
        stats: Dictionary with 'episode_rewards', 'policy_losses', 'value_losses'
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    episodes = np.arange(len(stats['episode_rewards']))
    ax.plot(episodes, stats['episode_rewards'], alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    ax = axes[0, 1]
    ax.plot(episodes, stats['episode_lengths'], alpha=0.6, color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Lengths')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Policy Loss
    ax = axes[1, 0]
    ax.plot(episodes, stats['policy_losses'], alpha=0.6, color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Value Loss
    ax = axes[1, 1]
    ax.plot(episodes, stats['value_losses'], alpha=0.6, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Value Loss (Baseline)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_plotting():
    """Test plotting functions"""
    print("Testing plotting utilities...")
    
    # Generate fake data
    np.random.seed(42)
    num_episodes = 500
    
    # Simulate improving rewards
    base_rewards = np.linspace(10, 100, num_episodes)
    noise = np.random.randn(num_episodes) * 20
    episode_rewards = base_rewards + noise
    episode_rewards = np.clip(episode_rewards, 0, 200)
    
    # Test 1: Single learning curve
    print("\n1. Testing single learning curve...")
    plot_learning_curve(
        episode_rewards,
        title="Test Learning Curve",
        save_path="results/plots/test_learning_curve.png"
    )
    
    # Test 2: Multiple seeds
    print("\n2. Testing multiple seeds plot...")
    all_rewards = []
    for seed in range(3):
        np.random.seed(seed)
        noise = np.random.randn(num_episodes) * 20
        rewards = base_rewards + noise
        rewards = np.clip(rewards, 0, 200)
        all_rewards.append(rewards)
    
    plot_multiple_seeds(
        all_rewards,
        labels=['Seed 0', 'Seed 1', 'Seed 2'],
        title="Test Multiple Seeds",
        save_path="results/plots/test_multiple_seeds.png"
    )
    
    # Test 3: Training metrics
    print("\n3. Testing training metrics plot...")
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_rewards * 0.8,  # Fake lengths
        'policy_losses': np.abs(np.random.randn(num_episodes) * 0.5),
        'value_losses': np.abs(np.random.randn(num_episodes) * 10)
    }
    
    plot_training_metrics(
        stats,
        save_path="results/plots/test_metrics.png"
    )
    
    print("\n✓ All plotting tests complete!")
    print("Check results/plots/ for generated figures")


if __name__ == "__main__":
    test_plotting()