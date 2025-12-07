import json
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_lr_sweep():
    """Analyze learning rate sweep results"""
    print("=" * 70)
    print("Learning Rate Sweep Analysis")
    print("=" * 70)
    
    # Load results
    with open('experiments/cartpole/results/lr_sweep.json', 'r') as f:
        results = json.load(f)
    
    # Extract data
    policy_lrs = []
    value_lrs = []
    performances = []
    
    for result in results:
        config = result['config']
        policy_lrs.append(config['lr_policy'])
        value_lrs.append(config['lr_value'])
        performances.append(result['final_performance_mean'])
    
    # Find best
    best_idx = np.argmax(performances)
    best_config = results[best_idx]['config']
    best_perf = performances[best_idx]
    
    print(f"\nTotal configurations tested: {len(results)}")
    print(f"\nBest configuration:")
    print(f"  Policy LR: {best_config['lr_policy']:.0e}")
    print(f"  Value LR: {best_config['lr_value']:.0e}")
    print(f"  Performance: {best_perf:.2f}")
    
    print(f"\nPerformance range: {min(performances):.2f} - {max(performances):.2f}")
    print(f"Performance std: {np.std(performances):.2f}")
    
    # Create heatmap
    unique_policy_lrs = sorted(set(policy_lrs))
    unique_value_lrs = sorted(set(value_lrs))
    
    heatmap = np.zeros((len(unique_policy_lrs), len(unique_value_lrs)))
    
    for i, plr in enumerate(unique_policy_lrs):
        for j, vlr in enumerate(unique_value_lrs):
            # Find matching result
            for result in results:
                config = result['config']
                if config['lr_policy'] == plr and config['lr_value'] == vlr:
                    heatmap[i, j] = result['final_performance_mean']
                    break
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Final Performance')
    
    plt.xticks(range(len(unique_value_lrs)), 
               [f'{lr:.0e}' for lr in unique_value_lrs])
    plt.yticks(range(len(unique_policy_lrs)), 
               [f'{lr:.0e}' for lr in unique_policy_lrs])
    
    plt.xlabel('Value Learning Rate', fontsize=12)
    plt.ylabel('Policy Learning Rate', fontsize=12)
    plt.title('Hyperparameter Heatmap - Final Performance', fontsize=14)
    
    # Annotate best
    best_i = unique_policy_lrs.index(best_config['lr_policy'])
    best_j = unique_value_lrs.index(best_config['lr_value'])
    plt.plot(best_j, best_i, 'b*', markersize=20, label='Best')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/hyperparameter_heatmap.png', dpi=300)
    print("\nHeatmap saved to results/plots/hyperparameter_heatmap.png")
    plt.close()


def create_summary_table():
    """Create summary table of all results"""
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    
    # Load best config
    with open('experiments/cartpole/results/best_config.json', 'r') as f:
        best_config = json.load(f)
    
    print("\nFinal Recommended Hyperparameters:")
    print("-" * 70)
    print(f"{'Parameter':<20} {'Value':<20}")
    print("-" * 70)
    for key, value in best_config.items():
        if isinstance(value, float) and value < 0.01:
            print(f"{key:<20} {value:.0e}")
        else:
            print(f"{key:<20} {value}")
    print("-" * 70)


def main():
    if os.path.exists('experiments/cartpole/results/lr_sweep.json'):
        analyze_lr_sweep()
    else:
        print("Learning rate sweep results not found.")
        print("Run hyperparameter_sweep.py first.")
    
    if os.path.exists('experiments/cartpole/results/best_config.json'):
        create_summary_table()
    else:
        print("\nBest config not found.")


if __name__ == "__main__":
    main()