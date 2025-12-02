import gymnasium as gym
import numpy as np

class CartPoleEnv:
    """
    CartPole-v1 environment
    
    State space: Box(4,) - [position, velocity, angle, angular_velocity]
    Action space: Discrete(2) - 0: push left, 1: push right
    
    Episode terminates when:
    - Pole angle > ±12 degrees
    - Cart position > ±2.4
    - Episode length > 500 steps
    
    Reward: +1 for every step the pole remains upright
    """
    
    def __init__(self, render_mode=None):
        """
        Initialize CartPole environment
        
        Args:
            render_mode: None (no rendering), 'human' (visual), or 'rgb_array'
        """
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.state_dim = self.env.observation_space.shape[0]  # 4
        self.action_dim = self.env.action_space.n  # 2
        
    def reset(self, seed=None):
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            state: Initial state (numpy array of shape (4,))
        """
        state, info = self.env.reset(seed=seed)
        return state
    
    def step(self, action):
        """
        Take one step in the environment
        
        Args:
            action: int (0 or 1)
            
        Returns:
            next_state: numpy array of shape (4,)
            reward: float (typically 1.0)
            done: bool (True if episode ended)
            truncated: bool (True if hit max steps)
            info: dict (additional info)
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


def test_cartpole():
    print("Testing CartPole")
    
    env = CartPoleEnv(render_mode="human")
    
    print(f"State dimension: {env.get_state_dim()}")
    print(f"Action dimension: {env.get_action_dim()}")
    print("\nRunning 3 random episodes:")
    
    for episode in range(3):
        state = env.reset(seed=episode)
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = np.random.randint(0, env.get_action_dim())
            next_state, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
        print(f"Episode {episode + 1}: {steps} steps, total reward = {total_reward:.1f}")
    env.close()
    print("CartPole wrapper working correctly!")


if __name__ == "__main__":
    test_cartpole()