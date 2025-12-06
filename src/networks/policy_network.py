import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities
    Input: state
    Output: probability distribution over actions
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Forward pass
        Args:
            state: tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            action_probs: tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs
    
    def select_action(self, state):
        """
        Sample action from policy
        Args:
            state: numpy array of shape (state_dim,)
        Returns:
            action: int
            log_prob: log probability of selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob


class ValueNetwork(nn.Module):
    """
    Value network (baseline) that estimates state value V(s)
    Input: state
    Output: scalar value estimate
    """
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Forward pass
        Args:
            state: tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            value: tensor of shape (batch_size, 1) or (1,)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


def test_networks():
    """Test that networks work correctly"""
    print("Testing PolicyNetwork...")
    state_dim = 4 
    action_dim = 2
    
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim=64)
    
    state = torch.randn(state_dim)
    action_probs = policy(state)
    print(f"Action probs shape: {action_probs.shape}")
    print(f"Action probs sum: {action_probs.sum().item():.4f}")
    assert action_probs.shape == (action_dim,)
    assert abs(action_probs.sum().item() - 1.0) < 1e-5
    
    batch_states = torch.randn(32, state_dim)
    batch_probs = policy(batch_states)
    print(f"Batch probs shape: {batch_probs.shape}")
    assert batch_probs.shape == (32, action_dim)
    
    print("✓ PolicyNetwork working!\n")
    
    print("Testing ValueNetwork...")
    value_net = ValueNetwork(state_dim, hidden_dim=64)
    
    value = value_net(state)
    print(f"Value shape: {value.shape}")
    assert value.shape == (1,)
    
    batch_values = value_net(batch_states)
    print(f"Batch values shape: {batch_values.shape}")
    assert batch_values.shape == (32, 1)
    
    print("✓ ValueNetwork working!\n")
    print("✓ All network tests passed!")


if __name__ == "__main__":
    test_networks()