import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ContinuousValueNetwork(nn.Module):
    """
    Value network for continuous actor-critic.
    """
    
    def __init__(self, state_dim, hidden_size=64):
        """
        Initialize value network.
        """
        super(ContinuousValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        
        self.h1 = nn.Linear(state_dim, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, 1)
        
        self.activation = nn.ELU()
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        self.h3.bias.data.fill_(0.0)
    
    def forward(self, state):
        """
        Forward pass.
        """
        x = self.activation(self.h1(state))
        x = self.activation(self.h2(x))
        value = self.h3(x)
        
        return value
    
class DiscreteValueNetwork(nn.Module):
    """
    Value network for discrete actor-critic (CartPole).
    """
    
    def __init__(self, state_dim, hidden_size=128):
        """
        Initialize value network.
        """
        super(DiscreteValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        
        self.h1 = nn.Linear(state_dim, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, 1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """
        Forward pass.
        """
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        value = self.h3(x)
        
        return value
