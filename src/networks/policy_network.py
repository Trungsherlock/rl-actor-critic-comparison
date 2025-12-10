import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ContinuousPolicyNetwork(nn.Module):
    """
    Policy network for continuous actions.
    """
    
    def __init__(self, state_dim, action_dim=1, hidden_size=64):
        """
        Initialize continuous policy network.
        """
        super(ContinuousPolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        self.h1 = nn.Linear(state_dim, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        
        self.activation = nn.ELU() 
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
        x = self.activation(self.h1(state))
        x = self.activation(self.h2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample_action(self, state):
        """
        Sample an action from the policy.
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy

class DiscretePolicyNetwork(nn.Module):
    """
    Policy network for discrete actions (CartPole).
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=128):
        """
        Initialize discrete policy network.
        """
        super(DiscretePolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        self.h1 = nn.Linear(state_dim, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, action_dim)
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
        action_logits = self.h3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs
    
    def sample_action(self, state):
        """
        Sample an action from the policy.
        """
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
