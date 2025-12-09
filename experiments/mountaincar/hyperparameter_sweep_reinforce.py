import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.environments.mountaincarcontinuous import MountainCarEnv
from src.algorithms.reinforce import REINFORCE
from src.utils.plotting_utils import plot_comparison
import numpy as np
import json
from tqdm import tqdm
import itertools
from multiprocessing import Pool, cpu_count

def train_single_seed(args):
    config, num_episodes, seed = args

    env = MountainCarEnv()

    agent = REINFORCE(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        hidden_dim=config['hidden_dim'],
        lr_policy=config['lr_policy'],
        lr_value=config['lr_value'],
        gamma=config['gamma'],
        continuous=True
    )

    stats = agent.train(env, num_episodes=num_episodes, print_every=999999, seed=seed)

    rewards = stats['episode_rewards']

    final_perf = np.mean(rewards[-100:])
    success_rate = np.mean([r > 90 for r in rewards[-100:]])
    env.close()

    return rewards, final_perf, success_rate


def train_with_config(config, num_episodes=1000, num_seeds=3, use_multiprocessing=True):
    all_rewards = []
    final_performances = []
    success_rates = []

    if use_multiprocessing:
        args_list = [(config, num_episodes, seed) for seed in range(num_seeds)]
        num_workers = min(cpu_count(), num_seeds)
        with Pool(processes=num_workers) as pool:
            results_list = pool.map(train_single_seed, args_list)
        for rewards, final_perf, success_rate in results_list:
            all_rewards.append(rewards)
            final_performances.append(final_perf)
            success_rates.append(success_rate)
    else:
        for seed in range(num_seeds):
            rewards, final_perf, success_rate = train_single_seed((config, num_episodes, seed))
            all_rewards.append(rewards)
            final_performances.append(final_perf)
            success_rates.append(success_rate)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    results = {
        'config': config,
        'mean_rewards': mean_rewards.tolist(),
        'std_rewards': std_rewards.tolist(),
        'final_performance_mean': np.mean(final_performances),
        'final_performance_std': np.std(final_performances),
        'success_rate_mean': np.mean(success_rates),
        'success_rate_std': np.std(success_rates),
        'all_final_performances': final_performances
    }

    return results

def learning_rate_sweep():
    lr_policy_options = [3e-4, 5e-4, 1e-3]
    lr_value_options = [1e-3, 3e-3, 5e-3]
    
    base_config = {
        'hidden_dim': 256,
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
        result = train_with_config(config)
        results.append(result)

    best_idx = np.argmax([r['final_performance_mean'] for r in results])
    best_result = results[best_idx]
    best_config = best_result['config']

    with open('experiments/mountaincar/results/reinforce/lr_sweep.json', 'w') as f:
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
        title="REINFORCE Learning Rate Comparison - MountainCar (Top 5)",
        save_path="results/plots/reinforce_mountaincar_lr_sweep_comparison.png"
    )

    return best_config

def main():
    best_lr_config = learning_rate_sweep()
    with open('experiments/mountaincar/results/reinforce/best_config.json', 'w') as f:
        json.dump(best_lr_config, f, indent=2)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
