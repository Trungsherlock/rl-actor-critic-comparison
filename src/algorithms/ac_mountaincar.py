import numpy as np
import torch
import time
import json
import sys
import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from environments.mountaincar import ContinuousMountainCarEnv, StateNormalizer
from algorithms.actor_critic import ContinuousActorCritic


def train_continuous_mountaincar(
                                num_episodes=1000,
                                seed=0,
                                actor_lr=0.0001,
                                critic_lr=0.0005,
                                gamma=0.99,
                                log_every=20,
                                save_path="continuous_mountaincar_results.json",
                                plot_window=100,
                            ):
    """
    Train continuous actor-critic on MountainCar.

    Args:
        num_episodes: number of episodes to train
        seed: random seed
        actor_lr: actor learning rate (alpha)
        critic_lr: critic learning rate
        gamma: discount factor
        log_every: print progress every N episodes
        save_path: where to save results JSON & plots
        plot_window: window for running average in plots
    """

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = ContinuousMountainCarEnv(max_steps=999, gamma=gamma, seed=seed)
    normalizer = StateNormalizer()
    # Create agent
    agent = ContinuousActorCritic(
        state_dim=2,
        action_dim=1,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        hidden_size=64,
    )

    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    episode_td_errors = []
    episode_actor_losses = []
    episode_critic_losses = []
    episode_max_positions = []
    episode_successes = []

    print(f"Training Continuous Actor-Critic on MountainCar")
    print(f"Episodes: {num_episodes}, Seed: {seed}")
    print(f"Actor LR (alpha): {actor_lr}, Critic LR: {critic_lr}, Gamma: {gamma}")

    print(
        f"{'Ep':>6} | {'Reward':>8} | {'Success':>7} | {'MaxPos':>8} | "
        f"{'TDerr':>8} | {'ActLoss':>9} | {'CriLoss':>9} | {'Steps':>5} | {'Time':>8}"
    )
    print("-" * 100)

    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        state = normalizer.normalize(state)
        ep_reward = 0.0
        max_position = -1.2  
        steps = 0
        success = False

        ep_td_errors = []
        ep_actor_losses = []
        ep_critic_losses = []

        while True:
            action, log_prob, entropy = agent.select_action(state)
            next_state_raw, reward, done = env.step(action)
            next_state = normalizer.normalize(next_state_raw)
            max_position = max(max_position, next_state_raw[0])

            # Check if reached goal
            if next_state_raw[0] >= 0.45:
                success = True

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

            if done:
                break

        # Store episode metrics
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        episode_max_positions.append(max_position)
        episode_td_errors.append(np.mean(ep_td_errors))
        episode_actor_losses.append(np.mean(ep_actor_losses))
        episode_critic_losses.append(np.mean(ep_critic_losses))
        episode_successes.append(1 if success else 0)

        # Logging
        if (episode + 1) % log_every == 0 or episode == 0:
            recent = min(100, episode + 1)
            avg_reward = np.mean(episode_rewards[-recent:])
            avg_success = np.mean(episode_successes[-recent:]) * 100
            avg_maxpos = np.mean(episode_max_positions[-recent:])
            avg_td_error = np.mean(episode_td_errors[-recent:])
            avg_actor_loss = np.mean(episode_actor_losses[-recent:])
            avg_critic_loss = np.mean(episode_critic_losses[-recent:])
            avg_steps = np.mean(episode_lengths[-recent:])

            elapsed = time.time() - start_time
            print(
                f"{episode+1:6d} | {avg_reward:8.2f} | {avg_success:6.1f}% | "
                f"{avg_maxpos:+8.4f} | {avg_td_error:+8.4f} | "
                f"{avg_actor_loss:+9.4f} | {avg_critic_loss:9.4f} | "
                f"{avg_steps:5.0f} | {elapsed/60:>7.1f}m"
            )

    training_time = time.time() - start_time
    final_success_rate = np.mean(episode_successes[-100:]) * 100
    final_reward = np.mean(episode_rewards[-100:])

    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Final success rate (last 100 episodes): {final_success_rate:.1f}%")
    print(f"Final average reward (last 100): {final_reward:.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Best max position: {max(episode_max_positions):.4f}")

    env.close()

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "max_positions": episode_max_positions,
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
        "seed": int(seed),
        "num_episodes": int(num_episodes),
    }


def save_results(results, save_path, plot_window):
    """Save JSON results and plots"""
    actor_lr = results.get("actor_lr")
    critic_lr = results.get("critic_lr")
    gamma = results.get("gamma")
    seed = results.get("seed")
    num_episodes = results.get("num_episodes")

    def fmt_lr(v):
        if v is None:
            return "NA"
        return f"{v:.0e}" if v < 1e-2 else f"{v:g}"

    suffix_parts = []
    if actor_lr is not None:
        suffix_parts.append(f"a{fmt_lr(actor_lr)}")
    if critic_lr is not None:
        suffix_parts.append(f"c{fmt_lr(critic_lr)}")
    if gamma is not None:
        suffix_parts.append(f"g{fmt_lr(gamma)}")
    if seed is not None:
        suffix_parts.append(f"s{seed}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    hyper_subtitle = (
        f"alpha={fmt_lr(actor_lr)}, critic_lr={fmt_lr(critic_lr)}, gamma={fmt_lr(gamma)}, "
        f"seed={seed}, episodes={num_episodes}"
    )

    # adjust JSON filename 
    root, ext = os.path.splitext(save_path)
    if not ext:
        ext = ".json"
    save_path = root + suffix + ext

    # save JSON
    results_json = {}
    for key, value in results.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            results_json[key] = [float(x) for x in value]
        else:
            results_json[key] = (
                float(value)
                if isinstance(value, (np.floating, np.integer, float, int))
                else value
            )

    with open(save_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Results saved to: {save_path}")

    base = save_path.replace(".json", "")

    # helpers for stats & plotting

    def moving_stats_prefix(y, max_window):
        """
        Prefix running mean/std:
        - at episode 1, window size = 1
        - grows until max_window
        - then uses rolling window of size max_window
        """
        y = np.asarray(y, dtype=float)
        means, stds = [], []
        for i in range(len(y)):
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
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        assert len(x) == len(y), f"x ({len(x)}) and y ({len(y)}) must match"

        plt.figure(figsize=(10, 5))

        plt.plot(x, y, alpha=0.4, linewidth=1.2, label=ylabel)

        if running_avg and len(y) >= 1:
            ma, std = moving_stats_prefix(y, plot_window)
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
            plt.axhline(
                y=hline,
                linestyle="--",
                linewidth=2,
                label=f"Goal ({hline})",
            )

        # Title + hyperparam subtitle
        plt.title(f"{title}\n{hyper_subtitle}")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

    def plot_metric(key, title, ylabel, suffix_img, running_avg=True, hline=None, show_std=False):
        y = results[key]
        x = np.arange(1, len(y) + 1)
        save_plot(
            x,
            y,
            title,
            ylabel,
            base + suffix_img,
            running_avg=running_avg,
            hline=hline,
            show_std=show_std,
        )

    print("\nSaving plots...")
    plot_metric(
        "rewards",
        "Reward per Episode",
        "Reward",
        "_reward.png",
        running_avg=True,
        show_std=True,  # std band only on reward
    )
    plot_metric(
        "max_positions",
        "Max Position per Episode",
        "Max Position",
        "_maxpos.png",
        running_avg=False,
        hline=0.45,
    )
    plot_metric(
        "td_errors",
        "Average TD Error per Episode",
        "TD Error",
        "_td_error.png",
    )
    plot_metric(
        "actor_losses",
        "Actor Loss per Episode",
        "Actor Loss",
        "_actor_loss.png",
    )
    plot_metric(
        "critic_losses",
        "Critic Loss per Episode",
        "Critic Loss",
        "_critic_loss.png",
    )
    plot_metric(
        "lengths",
        "Episode Length",
        "Steps",
        "_length.png",
        running_avg=False,
    )
    plot_metric(
        "successes",
        "Success Rate",
        "Success (1/0)",
        "_success.png",
    )
    print("All plots saved!\n")
    
def moving_average(y, window):
    """
    Fixed-size moving average over episodes for ONE run.
    Returns array of length N - window + 1, aligned so that
    index i corresponds to the average over episodes [i, i+window-1].
    """
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        raise ValueError(f"Need at least {window} episodes, got {len(y)}")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="valid")


def save_aggregated_learning_curve(
    reward_runs,
    window,
    base_path,
    actor_lr,
    critic_lr,
    gamma,
    num_episodes,
    seeds,
):
    """
    reward_runs: list of reward sequences, one per run (all same length)
    window: moving-average window, e.g., 50
    base_path: base name for saving fig and JSON
    actor_lr, critic_lr, gamma, num_episodes, seeds: for annotation
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
        ma = moving_average(r, window) 
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
        f"episodes={num_episodes}, runs={num_runs}, seeds={seeds}"
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
    plt.ylabel("Total Episode Reward (50-ep MA)")
    plt.title(f"Aggregated Learning Curve\n{subtitle}")
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
        "x_episodes": x.tolist(),
        "mean_ma": mean_ma.tolist(),
        "std_ma": std_ma.tolist(),
    }
    json_path = base_path + f"_agg_ma{window}.json"
    with open(json_path, "w") as f:
        json.dump(agg_json, f, indent=2)
    print(f"[multi-run] Saved aggregated data to: {json_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Continuous Actor-Critic on MountainCar"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--actor_lr", type=float, default=0.0001, help="Actor learning rate (alpha)"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=0.0005, help="Critic learning rate"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--log_every", type=int, default=20, help="Logging frequency"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="continuous_mountaincar_results.json",
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
        help="Window size for aggregated moving average across runs",
    )

    args = parser.parse_args()

    if args.num_runs == 1:
        results = train_continuous_mountaincar(
            num_episodes=args.episodes,
            seed=args.seed,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
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
            seed_i = args.seed + run_idx 
            print(f"\n Run {run_idx+1}/{args.num_runs} with seed={seed_i}")

            results_i = train_continuous_mountaincar(
                num_episodes=args.episodes,
                seed=seed_i,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                gamma=args.gamma,
                log_every=args.log_every,
                save_path=args.save,      
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
            num_episodes=args.episodes,
            seeds=seeds,
        )
