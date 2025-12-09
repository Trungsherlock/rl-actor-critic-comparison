import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.environments.cartpole import CartPoleEnv
from src.algorithms.reinforce import REINFORCE
from src.utils.plotting_utils import plot_aggregated_learning_curve
import numpy as np
import json
from tqdm import tqdm


def main():
    with open('experiments/cartpole/results/best_config.json', 'r') as f:
        best_config = json.load(f)
    
    num_seeds = 5
    num_episodes = 1000
    hidden_dim = 64
    gamma = 0.99
    best_config['hidden_dim'] = hidden_dim
    best_config['gamma'] = gamma
    all_rewards = []
    final_performances = []
    
    for seed in tqdm(range(num_seeds), desc="Final runs"):
        env = CartPoleEnv()
        
        agent = REINFORCE(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            hidden_dim=best_config['hidden_dim'],
            lr_policy=best_config['lr_policy'],
            lr_value=best_config['lr_value'],
            gamma=best_config['gamma']
        )
        
        stats = agent.train(env, num_episodes=num_episodes, print_every=999999, seed=seed)
        
        rewards = stats['episode_rewards']
        all_rewards.append(rewards)
        
        final_perf = np.mean(rewards[-100:])
        final_performances.append(final_perf)
        
        print(f"  Seed {seed}: Final performance = {final_perf:.2f}")
        
        env.close()
    
    mean_final = np.mean(final_performances)
    std_final = np.std(final_performances)
    
    results = {
        'config': best_config,
        'num_seeds': num_seeds,
        'num_episodes': num_episodes,
        'final_performance_mean': mean_final,
        'final_performance_std': std_final,
        'all_final_performances': final_performances,
        'all_rewards': [r for r in all_rewards] 
    }
    
    with open('experiments/cartpole/results/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to experiments/cartpole/results/final_results.json")
    
    plot_aggregated_learning_curve(
        all_rewards=all_rewards,
        config=best_config,
        num_episodes=num_episodes,
        num_runs=num_seeds,
        seeds=list(range(num_seeds)),
        window_size=50,
        title_prefix="Aggregated Learning Curve Of REINFORCE On Cartpole Environment",
        save_path="results/plots/reinforce_cartpole_aggregated.png"
    )
    
if __name__ == "__main__":
    main()