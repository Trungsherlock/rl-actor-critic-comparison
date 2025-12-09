import gymnasium as gym
import numpy as np

class MountainCarEnv:
    """
    MountainCarContinuous-v0 environment with state normalization

    State space: Box(2,) - [position, velocity]
        - position: Range [-1.2, 0.6]
        - velocity: Range [-0.07, 0.07]

    Normalized state space: Box(2,) - [position_norm, velocity_norm]
        - position_norm: Normalized to approximately [-1, 1]
        - velocity_norm: Normalized to approximately [-1, 1]

    Action space: Box(1,) - Continuous action in range [-1.0, 1.0]
        - Negative values: Push left
        - Positive values: Push right
        - Magnitude controls force strength

    Episode terminates when:
    - Car reaches the goal position (position >= 0.5)
    - Episode length > 999 steps (default for continuous version)

    Reward: Based on position and velocity, with bonus for reaching goal
    """

    def __init__(self, render_mode=None):
        """
        Initialize MountainCarContinuous environment

        Args:
            render_mode: None (no rendering), 'human' (visual), or 'rgb_array'
        """
        self.env = gym.make('MountainCarContinuous-v0', render_mode=render_mode)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]  # For continuous: shape[0] instead of .n

        # Normalization constants for state
        # Position: [-1.2, 0.6] -> normalize to ~[-1, 1]
        self.position_offset = 0.3  # Center: (-1.2 + 0.6) / 2 = -0.3, so offset by +0.3
        self.position_scale = 0.9   # Half-range: (0.6 - (-1.2)) / 2 = 0.9

        # Velocity: [-0.07, 0.07] -> normalize to [-1, 1]
        self.velocity_scale = 0.07
        
    def _normalize_state(self, state):
        """
        Normalize state to improve neural network training

        Args:
            state: Raw state [position, velocity]

        Returns:
            normalized_state: Normalized state with values roughly in [-1, 1]
        """
        position, velocity = state

        # Normalize position: [-1.2, 0.6] -> ~[-1, 1]
        position_norm = (position + self.position_offset) / self.position_scale

        # Normalize velocity: [-0.07, 0.07] -> [-1, 1]
        velocity_norm = velocity / self.velocity_scale

        return np.array([position_norm, velocity_norm], dtype=np.float32)

    def reset(self, seed=None):
        """
        Reset environment to initial state

        Args:
            seed: Random seed for reproducibility

        Returns:
            state: Normalized initial state (numpy array of shape (2,))
        """
        state, info = self.env.reset(seed=seed)
        return self._normalize_state(state)
    
    def step(self, action):
        """
        Take one step in the environment

        Args:
            action: numpy array of shape (1,) or float in range [-1.0, 1.0]
                - Negative values: Push left
                - Positive values: Push right
                - Magnitude controls force strength

        Returns:
            next_state: Normalized numpy array of shape (2,) - [position_norm, velocity_norm]
            reward: float (reward based on position/velocity progress and goal achievement)
            done: bool (True if episode ended)
            truncated: bool (True if hit max steps)
            info: dict (additional info)
        """
        # Ensure action is in correct format for gym (array-like)
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            action = np.array([action.item()], dtype=np.float32)

        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._normalize_state(next_state), reward, done, truncated, info
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def get_state_dim(self):
        """Return state dimension"""
        return self.state_dim
    
    def get_action_dim(self):
        """Return action dimension"""
        return self.action_dim