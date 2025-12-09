import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineNetwork(nn.Module):
    """
    Baseline network that estimates state value V(s)
    """
    def __init__(self, state_dim, hidden_dim=128):
        super(BaselineNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Forward pass
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


def test_baseline_network():
    """Test that baseline network works correctly"""
    print("Testing BaselineNetwork...")

    state_dim = 4  # CartPole
    baseline = BaselineNetwork(state_dim, hidden_dim=64)

    # Test single state
    state = torch.randn(state_dim)
    value = baseline(state)
    print(f"Value shape: {value.shape}")
    assert value.shape == (1,)

    # Test batch of states
    batch_states = torch.randn(32, state_dim)
    batch_values = baseline(batch_states)
    print(f"Batch values shape: {batch_values.shape}")
    assert batch_values.shape == (32, 1)

    print("✓ BaselineNetwork working!\n")


if __name__ == "__main__":
    test_baseline_network()
