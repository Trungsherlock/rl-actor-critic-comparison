import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.environments.cartpole import CartPoleEnv
from src.algorithms.reinforce import REINFORCE
from src.utils.plotting_utils import plot_multiple_seeds, plot_learning_curve
import numpy as np
import json
from tqdm import tqdm


def main():
    print("=" * 70)
    print("Final REINFORCE Runs on CartPole")
    print("=" * 70)
    
    # Load best configuration
    with open('experiments/cartpole/results/best_config.json', 'r') as f:
        best_config = json.load(f)
    
    print("\nUsing best hyperparameters:")
    print(json.dumps(best_config, indent=2))
    
    # Run with 10 different seeds
    num_seeds = 10
    num_episodes = 1000
    
    print(f"\nRunning {num_seeds} seeds × {num_episodes} episodes")
    print("-" * 70)
    
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
    
    # Compute statistics
    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)
    
    mean_final = np.mean(final_performances)
    std_final = np.std(final_performances)
    
    print(f"Final performance (last 100 episodes):")
    print(f"  Mean: {mean_final:.2f}")
    print(f"  Std:  {std_final:.2f}")
    print(f"  Min:  {min(final_performances):.2f}")
    print(f"  Max:  {max(final_performances):.2f}")
    
    # Save results
    results = {
        'config': best_config,
        'num_seeds': num_seeds,
        'num_episodes': num_episodes,
        'final_performance_mean': mean_final,
        'final_performance_std': std_final,
        'all_final_performances': final_performances,
        'all_rewards': [r for r in all_rewards]  # Save all learning curves
    }
    
    with open('experiments/cartpole/results/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to experiments/cartpole/results/final_results.json")
    
    # Plot
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)
    
    labels = [f'Seed {i}' for i in range(num_seeds)]
    plot_multiple_seeds(
        all_rewards,
        labels=labels,
        window_size=50,
        title="REINFORCE on CartPole - Final Results (10 Seeds)",
        save_path="results/plots/reinforce_cartpole_final.png"
    )
    
    # Also plot mean ± std
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    episodes = np.arange(len(mean_rewards))
    plt.plot(episodes, mean_rewards, color='blue', linewidth=2, label='Mean')
    plt.fill_between(episodes, 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     alpha=0.3, color='blue', label='±1 Std')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('REINFORCE on CartPole - Mean ± Std (10 Seeds)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/reinforce_cartpole_mean_std.png', dpi=300)
    print("Figure saved to results/plots/reinforce_cartpole_mean_std.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✓ Final runs complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()