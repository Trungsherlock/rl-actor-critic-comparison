import gymnasium as gym
import numpy as np


class ContinuousMountainCarEnv:
    """
    Continuous MountainCar environment imported from OpenAI gym
    """
    
    def __init__(self, max_steps=1000, gamma=0.99, seed=None):
        """
        Initialize environment.
        """
        self.env = gym.make('MountainCarContinuous-v0')
        self.max_steps = max_steps
        self.gamma = gamma
        self.current_step = 0
        
        if seed is not None:
            self.seed(seed)
    
    def seed(self, seed):
        self.env.reset(seed=seed)
        np.random.seed(seed)
    
    def reset(self):
        """
        Reset environment to initial state.
        """
        state, _ = self.env.reset()
        self.current_step = 0
        return state
    
    def step(self, action):
        """
        Take a step in the environment.
        """
        action = np.clip(action, -1.0, 1.0)
        next_state, reward, terminated, truncated, info = self.env.step([action])
        self.current_step += 1
        done = terminated or truncated or (self.current_step >= self.max_steps)
        
        return next_state, reward, done
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
        

class StateNormalizer:
    """
    Normalize MountainCar states to standard range.
    """
    
    def __init__(self):
        self.pos_min = -1.2
        self.pos_max = 0.6
        self.vel_min = -0.07
        self.vel_max = 0.07
        
    def normalize(self, state):
        """
        Normalize state to [-1, 1] range.
        """
        pos, vel = state[0], state[1]
        pos_norm = 2 * (pos - self.pos_min) / (self.pos_max - self.pos_min) - 1
        vel_norm = 2 * (vel - self.vel_min) / (self.vel_max - self.vel_min) - 1
        
        return np.array([pos_norm, vel_norm], dtype=np.float32)
