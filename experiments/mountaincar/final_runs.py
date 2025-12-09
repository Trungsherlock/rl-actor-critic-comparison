import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.environments.mountaincarcontinuous import MountainCarEnv
from src.algorithms.reinforce import REINFORCE
from src.utils.plotting_utils import plot_aggregated_learning_curve
import numpy as np
import json
import multiprocessing as mp


def load_best_config():
    config_path = 'experiments/mountaincar/results/reinforce/best_config.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(json.dumps(config, indent=2))
        return config
    else:
        return {
            'lr_policy': 5e-4,
            'lr_value': 1e-3
        }


def run_single_experiment(config, num_episodes=1000, seed=42):
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

    stats = agent.train(
        env,
        num_episodes=num_episodes,
        print_every=500,
        seed=seed
    )
    return stats, agent, env

def analyze_results(stats, config, seed):
    rewards = stats['episode_rewards']
    results = {
        'seed': seed,
        'config': config,
        'statistics': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
        },
        'episode_rewards': [float(r) for r in rewards]
    }
    return results


def evaluate_agent(agent, env, num_episodes=100, seed=100):
    """
    Evaluate trained agent
    """
    eval_reward, eval_std = agent.evaluate(env, num_episodes=num_episodes, seed=seed)
    success_rate = (eval_reward + 200) / 200 

    return eval_reward, eval_std

def save_model(agent, config, seed):
    os.makedirs('experiments/mountaincar/models/reinforce', exist_ok=True)
    model_path = f'experiments/mountaincar/models/reinforce/final_seed{seed}.pth'
    agent.save(model_path)
    config_path = f'experiments/mountaincar/models/reinforce/final_seed{seed}_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def run_and_process_seed(args):
    config, num_episodes, seed = args
    stats, agent, env = run_single_experiment(config, num_episodes=num_episodes, seed=seed)
    results = analyze_results(stats, config, seed)
    eval_reward, eval_std = evaluate_agent(agent, env, num_episodes=100, seed=100+seed)
    save_model(agent, config, seed)
    env.close()
    return {
        'seed': seed,
        'final_train_reward': np.mean(stats['episode_rewards'][-100:]),
        'eval_reward': eval_reward,
        'eval_std': eval_std,
        'episode_rewards': results['episode_rewards']
    }

def main():
    config = load_best_config()
    num_seeds = 5
    num_episodes = 1000
    hidden_dim = 64
    gamma = 0.99
    config['hidden_dim'] = hidden_dim
    config['gamma'] = gamma
    seed_args = [(config, num_episodes, seed) for seed in range(num_seeds)]

    import time
    start_time = time.time()

    with mp.Pool(processes=num_seeds) as pool:
        all_results = pool.map(run_and_process_seed, seed_args)

    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time/60:.1f} minutes")
    final_train_rewards = [r['final_train_reward'] for r in all_results]
    eval_rewards = [r['eval_reward'] for r in all_results]
    summary = {
        'config': config,
        'num_seeds': num_seeds,
        'num_episodes': num_episodes,
        'results': all_results,
        'summary': {
            'final_train_mean': float(np.mean(final_train_rewards)),
            'final_train_std': float(np.std(final_train_rewards)),
            'eval_mean': float(np.mean(eval_rewards)),
            'eval_std': float(np.std(eval_rewards))
        }
    }

    summary_path = 'experiments/mountaincar/results/reinforce/final_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    all_episode_rewards = [r['episode_rewards'] for r in all_results]
    seeds_list = list(range(num_seeds))

    plot_aggregated_learning_curve(
        all_rewards=all_episode_rewards,
        config=config,
        num_episodes=num_episodes,
        num_runs=num_seeds,
        seeds=seeds_list,
        window_size=50,
        title_prefix="Aggregated Learning Curve Of REINFORCE On Mountain Car Environment",
        save_path="results/plots/reinforce_mountaincar_aggregated.png"
    )

if __name__ == "__main__":
    main()
