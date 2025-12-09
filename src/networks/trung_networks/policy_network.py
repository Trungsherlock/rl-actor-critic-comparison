import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the Policy Network
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs

    def select_action(self, state):
        """
        Sample action from policy
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob