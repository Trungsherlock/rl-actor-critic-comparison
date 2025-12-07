"""
One-Step Actor-Critic Algorithm - FIXED for MountainCarEnv

Implements the episodic Actor-Critic method from Sutton & Barto Section 13.5.
"""

import torch
import torch.optim as optim
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork


class ActorCritic:
    """One-Step Actor-Critic Agent"""
    
    def __init__(self, state_dim, action_dim, 
                 actor_lr=0.001, critic_lr=0.001, gamma=0.99):
        """Initialize the Actor-Critic agent."""
        print(f"\nInitializing Actor-Critic Agent")
        print(f"  State dimension: {state_dim}")
        print(f"  Action dimension: {action_dim}")
        print(f"  Actor learning rate: {actor_lr}")
        print(f"  Critic learning rate: {critic_lr}")
        print(f"  Gamma (discount): {gamma}")
        
        self.actor = PolicyNetwork(state_dim, action_dim)
        print(f"  Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        
        self.critic = ValueNetwork(state_dim)
        print(f"  Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        
    def select_action(self, state):
        """Select an action using the current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        
        return action.item(), log_prob
    
    def compute_td_error(self, state, reward, next_state, done):
        """Compute the TD (Temporal Difference) Error."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        current_value = self.critic(state_tensor)
        
        with torch.no_grad():
            if done:
                next_value = torch.FloatTensor([[0.0]])
            else:
                next_value = self.critic(next_state_tensor)
        
        td_target = reward + self.gamma * next_value
        td_error = td_target - current_value
        
        return td_error, current_value
    
    def update_critic(self, td_error, current_value):
        """Update the Critic (value network)."""
        critic_loss = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(self, td_error, action_log_prob):
        """Update the Actor (policy network)."""
        actor_loss = -td_error.detach() * action_log_prob
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def train_step(self, state, action_log_prob, reward, next_state, done):
        """Perform one complete training step."""
        td_error, current_value = self.compute_td_error(state, reward, next_state, done)
        critic_loss = self.update_critic(td_error, current_value)
        actor_loss = self.update_actor(td_error, action_log_prob)
        
        return td_error.item(), actor_loss, critic_loss