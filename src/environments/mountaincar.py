import gymnasium as gym
import numpy as np


class ContinuousMountainCarEnv:
    """
    Continuous MountainCar environment imported from OpenAI gym
    
    State space:
        - position: [-1.2, 0.6]
        - velocity: [-0.07, 0.07]
    
    Action space:
        - force: [-1.0, 1.0] (continuous)
        - Negative = push left
        - Positive = push right
    
    Goal:
        - Reach position >= 0.45 (flag at top of hill)
    
    Reward:
        - +100 for reaching goal
        - Small penalty each step for using force: -action^2
    """
    
    def __init__(self, max_steps=1000, gamma=0.99, seed=None):
        """
        Initialize environment.
        
        Args:
            max_steps: maximum steps per episode
            gamma: discount factor
            seed: random seed needed to run multiple runs
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
        
        Returns:
            state: [position, velocity]
        """
        state, _ = self.env.reset()
        self.current_step = 0
        return state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: continuous action in [-1, 1]
        
        Returns:
            next_state: [position, velocity]
            reward: scalar reward
            done: whether episode ended
        """
        action = np.clip(action, -1.0, 1.0)
        
        # Take step
        next_state, reward, terminated, truncated, info = self.env.step([action])
        self.current_step += 1
        
        # Episode ends if goal reached or max steps
        done = terminated or truncated or (self.current_step >= self.max_steps)
        
        return next_state, reward, done
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
        

class StateNormalizer:
    """
    Normalize MountainCar states to standard range.
    
    This helps neural network training by keeping inputs in [-1, 1].
    """
    
    def __init__(self):
        # MountainCar state bounds
        self.pos_min = -1.2
        self.pos_max = 0.6
        self.vel_min = -0.07
        self.vel_max = 0.07
        
    def normalize(self, state):
        """
        Normalize state to [-1, 1] range.
        
        Args:
            state: [position, velocity]
            
        Returns:
            normalized_state: scaled to [-1, 1]
        """
        pos, vel = state[0], state[1]
        
        # Normalize to [-1, 1]
        pos_norm = 2 * (pos - self.pos_min) / (self.pos_max - self.pos_min) - 1
        vel_norm = 2 * (vel - self.vel_min) / (self.vel_max - self.vel_min) - 1
        
        return np.array([pos_norm, vel_norm], dtype=np.float32)
