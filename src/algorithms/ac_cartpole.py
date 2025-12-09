import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch
import time
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from environments.cartpole import CartPoleEnv
from algorithms.actor_critic import DiscreteActorCritic


def moving_average(x, window=100):
    """Compute moving average for smoothing plots."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def train_cartpole(
                    num_episodes=1000,
                    seed=0,
                    actor_lr=0.0001,
                    critic_lr=0.001,
                    gamma=0.99,
                    hidden_size=128,
                    log_every=50,
                    save_path="cartpole_results.json",
                    plot_window=100,
                ):
    """
    Train discrete actor-critic on CartPole.

    Args:
        num_episodes: number of episodes to train
        seed: random seed
        actor_lr: actor learning rate
        critic_lr: critic learning rate
        gamma: discount factor
        hidden_size: hidden layer size for networks
        log_every: logging frequency
        save_path: where to save results
        plot_window: window for moving average plots
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = CartPoleEnv()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # Create agent
    agent = DiscreteActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        hidden_size=hidden_size,
    )

    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    episode_td_errors = []
    episode_actor_losses = []
    episode_critic_losses = []
    episode_successes = []  # 1 if episode length >= 475, else 0

    print(f"Training Discrete Actor-Critic on CartPole-v1")
    print(f"Episodes: {num_episodes}, Seed: {seed}")
    print(
        f"Actor LR: {actor_lr}, Critic LR: {critic_lr}, "
        f"Gamma: {gamma}, Hidden Size: {hidden_size}"
    )

    print(
        f"{'Ep':>6} | {'Reward':>8} | {'Success':>7} | "
        f"{'TDerr':>8} | {'ActLoss':>9} | {'CriLoss':>9} | "
        f"{'Steps':>5} | {'Time':>8}"
    )

    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset(seed=seed + episode)

        ep_reward = 0.0
        steps = 0
        success = False

        ep_td_errors = []
        ep_actor_losses = []
        ep_critic_losses = []

        done = False
        while not done:
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            # Training step
            td_error, actor_loss, critic_loss = agent.train_step(
                state, log_prob, entropy, reward, next_state, done
            )

            # Track metrics
            ep_td_errors.append(td_error)
            ep_actor_losses.append(actor_loss)
            ep_critic_losses.append(critic_loss)

            ep_reward += reward
            state = next_state
            steps += 1

        if steps >= 475:
            success = True

        # Store episode metrics
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        episode_td_errors.append(np.mean(ep_td_errors))
        episode_actor_losses.append(np.mean(ep_actor_losses))
        episode_critic_losses.append(np.mean(ep_critic_losses))
        episode_successes.append(1 if success else 0)

        # Logging
        if (episode + 1) % log_every == 0 or episode == 0:
            recent = min(100, episode + 1)
            avg_reward = np.mean(episode_rewards[-recent:])
            avg_success = np.mean(episode_successes[-recent:]) * 100
            avg_td_error = np.mean(episode_td_errors[-recent:])
            avg_actor_loss = np.mean(episode_actor_losses[-recent:])
            avg_critic_loss = np.mean(episode_critic_losses[-recent:])
            avg_steps = np.mean(episode_lengths[-recent:])

            elapsed = time.time() - start_time
            print(
                f"{episode+1:6d} | {avg_reward:8.2f} | {avg_success:6.1f}% | "
                f"{avg_td_error:+8.4f} | {avg_actor_loss:+9.4f} | "
                f"{avg_critic_loss:9.4f} | {avg_steps:5.0f} | {elapsed/60:>7.1f}m"
            )

    training_time = time.time() - start_time
    final_success_rate = np.mean(episode_successes[-100:]) * 100
    final_reward = np.mean(episode_rewards[-100:])
    
    
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Final average reward (last 100): {final_reward:.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Best episode length: {max(episode_lengths)}")

    env.close()

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "td_errors": episode_td_errors,
        "actor_losses": episode_actor_losses,
        "critic_losses": episode_critic_losses,
        "successes": episode_successes,
        "training_time": float(training_time),
        "final_success_rate": float(final_success_rate),
        "final_reward": float(final_reward),
        "actor_lr": float(actor_lr),
        "critic_lr": float(critic_lr),
        "gamma": float(gamma),
        "hidden_size": int(hidden_size),
        "seed": int(seed),
        "num_episodes": int(num_episodes),
    }


def save_results(results, save_path, plot_window):
    """Save JSON results and plots."""
    if not save_path.endswith(".json"):
        save_path = save_path + ".json"

    results_json = {}
    for key, value in results.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            results_json[key] = [float(x) for x in value]
        else:
            results_json[key] = (
                float(value)
                if isinstance(value, (np.floating, np.integer))
                else value
            )

    with open(save_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Results saved to: {save_path}")

    # Plotting
    base = save_path.replace(".json", "")

    def moving_stats_prefix(y, max_window):
        """Prefix running mean/std: window grows at the start up to max_window."""
        y = np.asarray(y, dtype=float)
        means, stds = [], []
        for i in range(len(y)):
            # use as many past points as available, up to max_window
            start = max(0, i - max_window + 1)
            w = y[start : i + 1]
            means.append(w.mean())
            stds.append(w.std() if len(w) > 1 else 0.0)
        return np.array(means), np.array(stds)

    def save_plot(
                    x,
                    y,
                    title,
                    ylabel,
                    filename,
                    running_avg=True,
                    hline=None,
                    show_std=False,
                ):
        plt.figure(figsize=(10, 5))

        plt.plot(x, y, alpha=0.4, linewidth=1.2, label=ylabel)

        if running_avg and len(y) >= plot_window:
            ma, std = moving_stats_prefix(y, plot_window)
            if len(ma) > 0:
                ma_x = x

                plt.plot(ma_x, ma, linewidth=2.5, label=f"Running Avg ({plot_window})")

                if show_std:
                    upper = ma + std
                    lower = ma - std
                    plt.fill_between(
                        ma_x,
                        lower,
                        upper,
                        alpha=0.15,
                        label="±1 std",
                    )

        if hline is not None:
            plt.axhline(y=hline, linestyle="--", linewidth=2, label=f"Goal ({hline})")

        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

    episodes = np.arange(1, len(results["rewards"]) + 1)

    print("\nSaving plots...")
    save_plot(
        episodes,
        results["rewards"],
        "Reward per Episode",
        "Reward",
        base + "_reward.png",
        running_avg=True,
        hline=475,  # CartPole target
        show_std=True,  # std band only on reward plot
    )
    save_plot(
        episodes,
        results["td_errors"],
        "Average TD Error per Episode",
        "TD Error",
        base + "_td_error.png",
    )
    save_plot(
        episodes,
        results["actor_losses"],
        "Actor Loss per Episode",
        "Actor Loss",
        base + "_actor_loss.png",
    )
    save_plot(
        episodes,
        results["critic_losses"],
        "Critic Loss per Episode",
        "Critic Loss",
        base + "_critic_loss.png",
    )
    save_plot(
        episodes,
        results["lengths"],
        "Episode Length",
        "Steps",
        base + "_length.png",
        running_avg=True,
        hline=475,  
    )
    save_plot(
        episodes,
        results["successes"],
        "Success Rate",
        "Success (1/0)",
        base + "_success.png",
        running_avg=True,
    )

    print("All plots saved!\n")



# Multi-run 
def moving_average_fixed(y, window):
    """
    Fixed-size moving average over episodes for 1 run.
    Returns array of length N - window + 1, aligned so that
    index i corresponds to the average over episodes [i, i+window-1].
    """
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        raise ValueError(f"Need at least {window} episodes, got {len(y)}")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="valid")


def save_aggregated_learning_curve(reward_runs,
                                    window,
                                    base_path,
                                    actor_lr,
                                    critic_lr,
                                    gamma,
                                    hidden_size,
                                    num_episodes,
                                    seeds,
                                ):
    """
    reward_runs: list of reward sequences, one per run (all same length)
    window: moving-average window, e.g., 50
    base_path: base name for saving fig and JSON
    actor_lr, critic_lr, gamma, hidden_size, num_episodes, seeds: for annotation
    """
    reward_runs = [np.asarray(r, dtype=float) for r in reward_runs]
    lengths = [len(r) for r in reward_runs]
    if len(set(lengths)) != 1:
        raise ValueError(f"All runs must have same length, got lengths: {lengths}")
    N = lengths[0]
    num_runs = len(reward_runs)

    # Compute per-run moving averages
    ma_runs = []
    for r in reward_runs:
        ma = moving_average_fixed(r, window)  
        ma_runs.append(ma)
    ma_runs = np.stack(ma_runs, axis=0)  

    # Mean and std across runs
    mean_ma = ma_runs.mean(axis=0)
    std_ma = ma_runs.std(axis=0)
    x = np.arange(window, N + 1)

    # Hyperparam subtitle
    def fmt_lr(v):
        if v is None:
            return "NA"
        return f"{v:.0e}" if v < 1e-2 else f"{v:g}"

    subtitle = (
        f"alpha={fmt_lr(actor_lr)}, critic_lr={fmt_lr(critic_lr)}, gamma={fmt_lr(gamma)}, "
        f"hidden={hidden_size}, episodes={num_episodes}, "
        f"runs={num_runs}, seeds={seeds}"
    )

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(
        x,
        mean_ma,
        linewidth=2.5,
        label=f"Mean reward (MA window={window})",
    )
    plt.fill_between(
        x,
        mean_ma - std_ma,
        mean_ma + std_ma,
        alpha=0.2,
        label="Std across runs",
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward (moving avg)")
    plt.title(f"CartPole Aggregated Learning Curve\n{subtitle}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = base_path + f"_agg_ma{window}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[multi-run] Saved aggregated curve to: {fig_path}")

    # Also save numeric data
    agg_json = {
        "window": int(window),
        "episodes": int(num_episodes),
        "num_runs": int(num_runs),
        "seeds": list(seeds),
        "actor_lr": float(actor_lr),
        "critic_lr": float(critic_lr),
        "gamma": float(gamma),
        "hidden_size": int(hidden_size),
        "x_episodes": x.tolist(),
        "mean_ma": mean_ma.tolist(),
        "std_ma": std_ma.tolist(),
    }
    json_path = base_path + f"_agg_ma{window}.json"
    with open(json_path, "w") as f:
        json.dump(agg_json, f, indent=2)
    print(f"Saved aggregated data to: {json_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Discrete Actor-Critic on CartPole-v1"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--actor_lr", type=float, default=0.0001, help="Actor learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=0.001, help="Critic learning rate"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden layer size"
    )
    parser.add_argument(
        "--log_every", type=int, default=50, help="Logging frequency"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="cartpole_results.json",
        help="Base save path for single-run results",
    )
    parser.add_argument(
        "--plot_window",
        type=int,
        default=100,
        help="Moving average window for per-run plots",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of independent runs (seeds) for aggregation",
    )
    parser.add_argument(
        "--agg_window",
        type=int,
        default=50,
        help="Window size for aggregated moving average across runs (e.g., 50)",
    )

    args = parser.parse_args()

    if args.num_runs == 1:
        results = train_cartpole(
            num_episodes=args.episodes,
            seed=args.seed,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            hidden_size=args.hidden_size,
            log_every=args.log_every,
            save_path=args.save,
            plot_window=args.plot_window,
        )

        save_results(results, args.save, args.plot_window)

    else:
        print(
            f"\nRunning {args.num_runs} runs of {args.episodes} episodes each "
            f"for aggregation (agg_window={args.agg_window})\n"
        )

        reward_runs = []
        seeds = []

        for run_idx in range(args.num_runs):
            seed_i = args.seed + run_idx  # different seed per run
            print(f"\n=== Run {run_idx+1}/{args.num_runs} with seed={seed_i} ===")

            results_i = train_cartpole(
                num_episodes=args.episodes,
                seed=seed_i,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                gamma=args.gamma,
                hidden_size=args.hidden_size,
                log_every=args.log_every,
                save_path=args.save,  # not used inside, just kept for symmetry
                plot_window=args.plot_window,
            )

            reward_runs.append(results_i["rewards"])
            seeds.append(seed_i)

        agg_base = args.save.replace(".json", "") + f"_runs{args.num_runs}"

        save_aggregated_learning_curve(
            reward_runs=reward_runs,
            window=args.agg_window,
            base_path=agg_base,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            hidden_size=args.hidden_size,
            num_episodes=args.episodes,
            seeds=seeds,
        )
        print("Done")
