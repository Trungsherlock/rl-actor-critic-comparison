import torch
import torch.optim as optim
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from networks.policy_network import ContinuousPolicyNetwork, DiscretePolicyNetwork
from networks.value_network import ContinuousValueNetwork, DiscreteValueNetwork


class ContinuousActorCritic:
    """
    Actor-Critic for continuous action spaces.
    """
    
    def __init__(self, state_dim, action_dim=1,
                 actor_lr=0.0001, critic_lr=0.0005, gamma=0.99,
                 hidden_size=64):
        """
        Initialize continuous actor-critic.
        
        Args:
            state_dim: dimension of state space
            action_dim: dimension of action space (1 for MountainCar)
            actor_lr: learning rate for policy network
            critic_lr: learning rate for value network
            gamma: discount factor
            hidden_size: number of hidden units in networks
        """
        print(f"Initializing Continuous Actor-Critic")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Actor learning rate: {actor_lr}")
        print(f"Critic learning rate: {critic_lr}")
        print(f"Gamma (discount): {gamma}")
        print(f"Hidden size: {hidden_size}")
        
        # Create networks
        self.actor = ContinuousPolicyNetwork(state_dim, action_dim, hidden_size)
        self.critic = ContinuousValueNetwork(state_dim, hidden_size)
        
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        
    def select_action(self, state):
        """
        Select an action using the current policy.
        
        Args:
            state: current state (should be normalized)
            
        Returns:
            action: continuous action value
            log_prob: log probability of action
            entropy: entropy of action distribution
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Sample action from policy
        action, log_prob, entropy = self.actor.sample_action(state_tensor)
        action_value = action.detach().numpy()[0, 0]
        
        # Clip to valid range [-1, 1]
        action_value = np.clip(action_value, -1.0, 1.0)
        
        return action_value, log_prob, entropy
    
    def train_step(self, state, action_log_prob, action_entropy, 
                    reward, next_state, done):
        """
        Perform one training step.
        
        Args:
            state: current state (normalized)
            action_log_prob: log probability of action taken
            action_entropy: entropy of action distribution
            reward: reward received
            next_state: next state (normalized)
            done: whether episode ended
            
        Returns:
            td_error: temporal difference error
            actor_loss: actor loss value
            critic_loss: critic loss value
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        current_value = self.critic(state_tensor)
        
        # Compute TD target
        with torch.no_grad():
            if done:
                next_value = torch.FloatTensor([[0.0]])
            else:
                next_value = self.critic(next_state_tensor)
            
            td_target = reward + self.gamma * next_value
        
        # Compute TD error 
        td_error = td_target - current_value.detach()
        
        critic_loss = (current_value - td_target).pow(2).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -(td_error * action_log_prob)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return td_error.item(), actor_loss.item(), critic_loss.item()
    
    def save(self, filepath):
        """Save the networks."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the networks."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")

class DiscreteActorCritic:
    """
    Actor-Critic for discrete action spaces (CartPole).
    
    Standard actor-critic algorithm with:
    - Policy network outputting action probabilities
    - Value network estimating state values
    - TD learning for both networks
    """
    
    def __init__(self, state_dim, action_dim,
                 actor_lr=0.0001, critic_lr=0.001, gamma=0.99,
                 hidden_size=128):
        """
        Initialize discrete actor-critic.
        
        Args:
            state_dim: dimension of state space (4 for CartPole)
            action_dim: number of discrete actions (2 for CartPole)
            actor_lr: learning rate for policy network
            critic_lr: learning rate for value network
            gamma: discount factor
            hidden_size: number of hidden units in networks
        """
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Actor learning rate: {actor_lr}")
        print(f"Critic learning rate: {critic_lr}")
        print(f"Gamma (discount): {gamma}")
        print(f"Hidden size: {hidden_size}")
        
        # Create networks
        self.actor = DiscretePolicyNetwork(state_dim, action_dim, hidden_size)
        self.critic = DiscreteValueNetwork(state_dim, hidden_size)
        
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        
    def select_action(self, state):
        """
        Select an action using the current policy.
        
        Args:
            state: current state (numpy array)
            
        Returns:
            action: discrete action (int)
            log_prob: log probability of action
            entropy: entropy of action distribution
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Sample action from policy
        action, log_prob, entropy = self.actor.sample_action(state_tensor)
        
        # Return action as int
        return action.item(), log_prob, entropy
    
    def train_step(self, state, action_log_prob, action_entropy,
                    reward, next_state, done):
        """
        Perform one training step.
        
        Args:
            state: current state
            action_log_prob: log probability of action taken
            action_entropy: entropy of action distribution
            reward: reward received
            next_state: next state
            done: whether episode ended
            
        Returns:
            td_error: temporal difference error (advantage)
            actor_loss: actor loss value
            critic_loss: critic loss value
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        current_value = self.critic(state_tensor)
        
        # Compute TD target
        with torch.no_grad():
            if done:
                next_value = torch.FloatTensor([[0.0]])
            else:
                next_value = self.critic(next_state_tensor)
            
            td_target = reward + self.gamma * next_value
        
        # Compute TD error 
        td_error = td_target - current_value.detach()
    
        critic_loss = (current_value - td_target).pow(2).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -(td_error * action_log_prob)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return td_error.item(), actor_loss.item(), critic_loss.item()
    
    def save(self, filepath):
        """Save the networks."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the networks."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
