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
        final_perf = np.mean(rewards[-100:])
        final_performances.append(final_perf)
        
        env.close()
    
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
    lr_policy_options = [1e-4, 3e-4, 1e-3, 3e-3]
    lr_value_options = [1e-3, 3e-3, 1e-2]
    base_config = {
        'hidden_dim': 128,
        'gamma': 0.99
    }
    results = []
    configs = []
    for lr_p, lr_v in itertools.product(lr_policy_options, lr_value_options):
        config = base_config.copy()
        config['lr_policy'] = lr_p
        config['lr_value'] = lr_v
        configs.append(config)
    
    for config in tqdm(configs, desc="Testing learning rates"):
        result = train_with_config(config, num_episodes=500, num_seeds=5)
        results.append(result)
        
    best_idx = np.argmax([r['final_performance_mean'] for r in results])
    best_result = results[best_idx]
    best_config = best_result['config']
    os.makedirs('experiments/cartpole/results', exist_ok=True)
    with open('experiments/cartpole/results/lr_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
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

def main():
    best_lr_config = learning_rate_sweep()
    with open('experiments/cartpole/results/best_config.json', 'w') as f:
        json.dump(best_lr_config, f, indent=2)


if __name__ == "__main__":
    main()