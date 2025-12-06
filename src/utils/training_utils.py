import numpy as np
import torch
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns (G_t) for each timestep
    
    Args:
        rewards: list of rewards [r_0, r_1, ..., r_T]
        gamma: discount factor
    
    Returns:
        returns: list of returns [G_0, G_1, ..., G_T]
    
    Example:
        rewards = [1, 1, 1]
        gamma = 0.9
        G_2 = 1
        G_1 = 1 + 0.9 * 1 = 1.9
        G_0 = 1 + 0.9 * 1.9 = 2.71
        returns = [2.71, 1.9, 1.0]
    """
    returns = []
    G = 0
    
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    
    return returns


def test_compute_returns():
    print("Testing compute_returns...")
    
    rewards = [1, 1, 1]
    gamma = 0.9
    returns = compute_returns(rewards, gamma)
    
    expected = [2.71, 1.9, 1.0]
    print(f"Rewards: {rewards}")
    print(f"Returns: {[f'{r:.2f}' for r in returns]}")
    print(f"Expected: {expected}")
    
    assert abs(returns[0] - 2.71) < 0.01
    assert abs(returns[1] - 1.9) < 0.01
    assert abs(returns[2] - 1.0) < 0.01
    
    rewards = [1, 0, 1]
    returns = compute_returns(rewards, gamma=1.0)
    print(f"\nRewards: {rewards}")
    print(f"Returns (gamma=1.0): {returns}")
    assert returns == [2, 1, 1]
    
    print("\n✓ compute_returns working correctly!")


if __name__ == "__main__":
    test_compute_returns()