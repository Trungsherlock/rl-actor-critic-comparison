import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ContinuousValueNetwork(nn.Module):
    """
    Value network for continuous actor-critic.
    
    Estimates V(s) - the expected return from state s.
    """
    
    def __init__(self, state_dim, hidden_size=64):
        """
        Initialize value network.
        
        Args:
            state_dim: dimension of state (2 for MountainCar)
            hidden_size: number of hidden units
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
        
        # Initialize output bias to 0 
        self.h3.bias.data.fill_(0.0)
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: normalized state tensor
            
        Returns:
            value: estimated value of state
        """
        x = self.activation(self.h1(state))
        x = self.activation(self.h2(x))
        value = self.h3(x)
        
        return value
    
class DiscreteValueNetwork(nn.Module):
    """
    Value network for discrete actor-critic (CartPole).
    
    Estimates V(s) - the expected return from state s.
    """
    
    def __init__(self, state_dim, hidden_size=128):
        """
        Initialize value network.
        
        Args:
            state_dim: dimension of state (4 for CartPole)
            hidden_size: number of hidden units
        """
        super(DiscreteValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        
        self.h1 = nn.Linear(state_dim, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, 1)
        
        # Initialize weights
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
        
        Args:
            state: state tensor
            
        Returns:
            value: estimated value of state
        """
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        value = self.h3(x)
        
        return value
