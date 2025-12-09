import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    """Critic network with proper initialization for MountainCar."""
    
    def __init__(self, state_dim, hidden_size=128):
        """Initialize the Critic Network."""
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        
        self.activation = nn.ReLU()
        
        # CRITICAL: Initialize to expect long episodes with -1 rewards
        # MountainCar: typical episode = 200-1000 steps
        # Expected return = -200 to -1000
        # Start in middle: -500
        self.layer3.bias.data.fill_(-500.0)
        
        # Scale weights for stable initialization
        self.layer3.weight.data.mul_(0.01)
        
    def forward(self, state):
        """Forward pass: compute the Critic of a state."""
        hidden1 = self.layer1(state)
        hidden1_activated = self.activation(hidden1)
        hidden2 = self.layer2(hidden1_activated)
        hidden2_activated = self.activation(hidden2)
        
        Critic = self.layer3(hidden2_activated)
        
        return Critic