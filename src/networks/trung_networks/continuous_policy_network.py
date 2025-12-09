import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ContinuousPolicyNetwork(nn.Module):
    """
    Policy network for continuous action spaces.
    Outputs mean and log_std for a Gaussian distribution.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, log_std_min=-20, log_std_max=2):
        """
        Initialize the Continuous Policy Network

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (typically 1 for MountainCarContinuous)
            hidden_dim: Number of hidden units
            log_std_min: Minimum value for log standard deviation (for numerical stability)
            log_std_max: Maximum value for log standard deviation (for numerical stability)
        """
        super(ContinuousPolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Mean head
        self.mean = nn.Linear(hidden_dim, action_dim)

        # Log standard deviation head
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass to get action distribution parameters

        Args:
            state: State tensor

        Returns:
            mean: Mean of the action distribution
            std: Standard deviation of the action distribution
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return mean, std

    def select_action(self, state, deterministic=False):
        """
        Sample action from policy

        Args:
            state: Current state (numpy array)
            deterministic: If True, return mean action (for evaluation)

        Returns:
            action: Selected action (numpy array of shape (1,))
            log_prob: Log probability of selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.forward(state)

        if deterministic:
            # For evaluation, use mean action
            action = mean
            log_prob = None
        else:
            # Sample from Gaussian distribution
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions

        # Clamp action to valid range [-1, 1] for MountainCarContinuous
        action = torch.clamp(action, -1.0, 1.0)

        # Return as numpy array, not scalar
        action_np = action.detach().cpu().numpy().flatten()

        if deterministic:
            return action_np, None
        else:
            return action_np, log_prob

    def get_log_prob(self, state, action):
        """
        Get log probability of action given state

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            log_prob: Log probability of action
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        return log_prob
