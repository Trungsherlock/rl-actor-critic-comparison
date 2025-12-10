import gymnasium as gym
import numpy as np

class MountainCarEnv:

    def __init__(self, render_mode=None):
        self.env = gym.make('MountainCarContinuous-v0', render_mode=render_mode)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.position_offset = 0.3 
        self.position_scale = 0.9 
        self.velocity_scale = 0.07
        
    def _normalize_state(self, state):
        position, velocity = state
        position_norm = (position + self.position_offset) / self.position_scale
        velocity_norm = velocity / self.velocity_scale

        return np.array([position_norm, velocity_norm], dtype=np.float32)

    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)
        return self._normalize_state(state)
    
    def step(self, action):
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            action = np.array([action.item()], dtype=np.float32)

        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._normalize_state(next_state), reward, done, truncated, info
    
    def close(self):
        self.env.close()
    
    def get_state_dim(self):
        return self.state_dim
    
    def get_action_dim(self):
        return self.action_dim