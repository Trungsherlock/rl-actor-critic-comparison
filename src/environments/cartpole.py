import gymnasium as gym
import numpy as np

class CartPoleEnv:
    """
    CartPole-v1 environment
    """
    
    def __init__(self, render_mode=None):
        """
        Initialize CartPole environment
        """
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.state_dim = self.env.observation_space.shape[0] 
        self.action_dim = self.env.action_space.n 
        
    def reset(self, seed=None):
        """
        Reset environment to initial state
        """
        state, info = self.env.reset(seed=seed)
        return state
    
    def step(self, action):
        """
        Take one step in the environment
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, truncated, info
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def get_state_dim(self):
        """Return state dimension"""
        return self.state_dim
    
    def get_action_dim(self):
        """Return action dimension"""
        return self.action_dim
