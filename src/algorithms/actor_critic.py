import torch
import torch.optim as optim
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from src.networks.anh_networks.policy_network import ContinuousPolicyNetwork, DiscretePolicyNetwork
from src.networks.anh_networks.value_network import ContinuousValueNetwork, DiscreteValueNetwork

class ContinuousActorCritic:
    def __init__(self, state_dim, action_dim=1,
                 actor_lr=0.0001, critic_lr=0.0005, gamma=0.99,
                 hidden_size=64):
        """
        Initialize continuous actor-critic.
        """
        self.actor = ContinuousPolicyNetwork(state_dim, action_dim, hidden_size)
        self.critic = ContinuousValueNetwork(state_dim, hidden_size)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        
    def select_action(self, state):
        """
        Select an action using the current policy.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, entropy = self.actor.sample_action(state_tensor)
        action_value = action.detach().numpy()[0, 0]
        action_value = np.clip(action_value, -1.0, 1.0)
        
        return action_value, log_prob, entropy
    
    def train_step(self, state, action_log_prob, action_entropy, 
                    reward, next_state, done):
        """
        Perform one training step.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        current_value = self.critic(state_tensor)
        
        with torch.no_grad():
            if done:
                next_value = torch.FloatTensor([[0.0]])
            else:
                next_value = self.critic(next_state_tensor)
            
            td_target = reward + self.gamma * next_value
        
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
    """
    
    def __init__(self, state_dim, action_dim,
                 actor_lr=0.0001, critic_lr=0.001, gamma=0.99,
                 hidden_size=128):
        """
        Initialize discrete actor-critic.
        """
        self.actor = DiscretePolicyNetwork(state_dim, action_dim, hidden_size)
        self.critic = DiscreteValueNetwork(state_dim, hidden_size)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        
    def select_action(self, state):
        """
        Select an action using the current policy.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, entropy = self.actor.sample_action(state_tensor)
        return action.item(), log_prob, entropy
    
    def train_step(self, state, action_log_prob, action_entropy,
                    reward, next_state, done):
        """
        Perform one training step.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        current_value = self.critic(state_tensor)
        
        with torch.no_grad():
            if done:
                next_value = torch.FloatTensor([[0.0]])
            else:
                next_value = self.critic(next_state_tensor)
            
            td_target = reward + self.gamma * next_value
        
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
