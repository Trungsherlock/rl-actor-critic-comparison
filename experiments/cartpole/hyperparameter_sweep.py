import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.environments.cartpole import CartPoleEnv
from src.algorithms.reinforce import REINFORCE
from src.utils.plotting_utils import plot_comparison
import numpy as np
import json
from tqdm import tqdm
import itertools


def train_with_config(config, num_episodes=500, num_seeds=5):
    """
    Train REINFORCE with specific hyperparameter configuration
    
    Args:
        config: Dictionary with hyperparameters
        num_episodes: Number of episodes to train
        num_seeds: Number of random seeds to average over
        
    Returns:
        results: Dictionary with training results
    """
    all_rewards = []
    final_performances = []
    
    for seed in range(num_seeds):
        env = CartPoleEnv()
        
        agent = REINFORCE(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            hidden_dim=config['hidden_dim'],
            lr_policy=config['lr_policy'],
            lr_value=config['lr_value'],
            gamma=config['gamma']
        )
        
        stats = agent.train(env, num_episodes=num_episodes, print_every=999999, seed=seed)
        
        rewards = stats['episode_rewards']
        all_rewards.append(rewards)
        
        # Average of last 100 episodes
        final_perf = np.mean(rewards[-100:])
        final_performances.append(final_perf)
        
        env.close()
    
    # Compute statistics
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    results = {
        'config': config,
        'mean_rewards': mean_rewards.tolist(),
        'std_rewards': std_rewards.tolist(),
        'final_performance_mean': np.mean(final_performances),
        'final_performance_std': np.std(final_performances),
        'all_final_performances': final_performances
    }
    
    return results


def learning_rate_sweep():
    """
    Sweep over different learning rates
    """
    print("=" * 70)
    print("Learning Rate Sweep")
    print("=" * 70)
    
    # Test different learning rate combinations
    lr_policy_options = [1e-4, 3e-4, 1e-3, 3e-3]
    lr_value_options = [1e-3, 3e-3, 1e-2]
    
    base_config = {
        'hidden_dim': 128,
        'gamma': 0.99
    }
    
    results = []
    
    print(f"\nTesting {len(lr_policy_options)} policy LRs × {len(lr_value_options)} value LRs")
    print(f"Policy LRs: {lr_policy_options}")
    print(f"Value LRs: {lr_value_options}")
    print("\n" + "-" * 70)
    
    # Create all combinations
    configs = []
    for lr_p, lr_v in itertools.product(lr_policy_options, lr_value_options):
        config = base_config.copy()
        config['lr_policy'] = lr_p
        config['lr_value'] = lr_v
        configs.append(config)
    
    # Train all configurations
    for config in tqdm(configs, desc="Testing learning rates"):
        result = train_with_config(config, num_episodes=500, num_seeds=5)
        results.append(result)
        
        print(f"  Policy LR: {config['lr_policy']:.0e}, "
              f"Value LR: {config['lr_value']:.0e} → "
              f"Final: {result['final_performance_mean']:.2f} ± {result['final_performance_std']:.2f}")
    
    # Find best configuration
    best_idx = np.argmax([r['final_performance_mean'] for r in results])
    best_result = results[best_idx]
    best_config = best_result['config']
    
    print("\n" + "=" * 70)
    print("Best Configuration:")
    print("=" * 70)
    print(f"Policy LR: {best_config['lr_policy']:.0e}")
    print(f"Value LR: {best_config['lr_value']:.0e}")
    print(f"Performance: {best_result['final_performance_mean']:.2f} ± {best_result['final_performance_std']:.2f}")
    
    # Save results
    os.makedirs('experiments/cartpole/results', exist_ok=True)
    with open('experiments/cartpole/results/lr_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to experiments/cartpole/results/lr_sweep.json")
    
    # Plot comparison of top 5
    top_5_indices = np.argsort([r['final_performance_mean'] for r in results])[-5:]
    rewards_dict = {}
    
    for idx in top_5_indices:
        result = results[idx]
        config = result['config']
        label = f"π:{config['lr_policy']:.0e}, v:{config['lr_value']:.0e}"
        rewards_dict[label] = result['mean_rewards']
    
    plot_comparison(
        rewards_dict,
        window_size=50,
        title="REINFORCE Learning Rate Comparison - Top 5",
        save_path="results/plots/lr_sweep_comparison.png"
    )
    
    return best_config


def network_size_sweep(best_lr_config):
    """
    Sweep over different network sizes
    """
    print("\n" + "=" * 70)
    print("Network Size Sweep")
    print("=" * 70)
    
    hidden_dim_options = [64, 128, 256]
    
    results = []
    
    print(f"\nTesting {len(hidden_dim_options)} different network sizes")
    print(f"Hidden dims: {hidden_dim_options}")
    print("\n" + "-" * 70)
    
    for hidden_dim in tqdm(hidden_dim_options, desc="Testing network sizes"):
        config = best_lr_config.copy()
        config['hidden_dim'] = hidden_dim
        
        result = train_with_config(config, num_episodes=500, num_seeds=5)
        results.append(result)
        
        print(f"  Hidden dim: {hidden_dim} → "
              f"Final: {result['final_performance_mean']:.2f} ± {result['final_performance_std']:.2f}")
    
    # Find best configuration
    best_idx = np.argmax([r['final_performance_mean'] for r in results])
    best_result = results[best_idx]
    best_config = best_result['config']
    
    print("\n" + "=" * 70)
    print("Best Configuration:")
    print("=" * 70)
    print(f"Hidden dim: {best_config['hidden_dim']}")
    print(f"Performance: {best_result['final_performance_mean']:.2f} ± {best_result['final_performance_std']:.2f}")
    
    # Save results
    with open('experiments/cartpole/results/network_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to experiments/cartpole/results/network_sweep.json")
    
    # Plot comparison
    rewards_dict = {}
    for result in results:
        config = result['config']
        label = f"Hidden dim: {config['hidden_dim']}"
        rewards_dict[label] = result['mean_rewards']
    
    plot_comparison(
        rewards_dict,
        window_size=50,
        title="REINFORCE Network Size Comparison",
        save_path="results/plots/network_sweep_comparison.png"
    )
    
    return best_config


def main():
    print("=" * 70)
    print("REINFORCE Hyperparameter Sweep on CartPole")
    print("=" * 70)
    print("\nThis will take approximately 1-2 hours to complete")
    print("Progress will be shown for each configuration\n")
    
    # Step 1: Learning rate sweep
    best_lr_config = learning_rate_sweep()
    
    # Step 2: Network size sweep (using best LRs)
    best_overall_config = network_size_sweep(best_lr_config)
    
    # Save final best configuration
    print("\n" + "=" * 70)
    print("Final Best Configuration")
    print("=" * 70)
    print(json.dumps(best_overall_config, indent=2))
    
    with open('experiments/cartpole/results/best_config.json', 'w') as f:
        json.dump(best_overall_config, f, indent=2)
    
    print("\n✓ Hyperparameter sweep complete!")
    print("Best configuration saved to experiments/cartpole/results/best_config.json")


if __name__ == "__main__":
    main()