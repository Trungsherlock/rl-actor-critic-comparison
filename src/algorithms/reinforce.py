import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.policy_network import PolicyNetwork, ValueNetwork
from utils.training_utils import set_seed, compute_returns


class REINFORCE:
    """
    REINFORCE with Baseline Algorithm
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr_policy=1e-3,
        lr_value=1e-3,
        gamma=0.99,
        device='cpu'
    ):
        """
        Initialize REINFORCE with Baseline
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: Number of hidden units in networks
            lr_policy: Learning rate for policy network
            lr_value: Learning rate for value network (baseline)
            gamma: Discount factor
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.baseline = ValueNetwork(state_dim, hidden_dim).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.baseline.parameters(), lr=lr_value)
        
        self.reset_episode()
        
    def reset_episode(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def select_action(self, state):
        """
        Select action using current policy
        
        Args:
            state: Current state (numpy array)
            
        Returns:
            action: Selected action (int)
            log_prob: Log probability of selected action
        """
        action, log_prob = self.policy.select_action(state)
        return action, log_prob
    
    def store_transition(self, state, action, reward, log_prob):
        """
        Store transition data for current episode
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def update(self):
        """
        Update policy and baseline using collected episode
        """
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)
        
        returns = compute_returns(self.rewards, self.gamma)
        returns = torch.FloatTensor(returns).to(self.device)
        
        baseline_values = self.baseline(states).squeeze()
        advantages = returns - baseline_values.detach()
        policy_loss = -(log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        value_loss = ((returns - baseline_values) ** 2).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train_episode(self, env):
        """
        Run one episode and update policy
        
        Args:
            env: Environment to interact with
            
        Returns:
            episode_reward: Total reward for this episode
            episode_length: Number of steps in episode
            policy_loss: Policy loss value
            value_loss: Value loss value
        """
        self.reset_episode()
        
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, log_prob = self.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            self.store_transition(state, action, reward, log_prob)
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        policy_loss, value_loss = self.update()
        return episode_reward, episode_length, policy_loss, value_loss
    
    def train(self, env, num_episodes, print_every=100, seed=None):
        """
        Train REINFORCE for multiple episodes
        
        Args:
            env: Environment to train on
            num_episodes: Number of episodes to train
            print_every: Print statistics every N episodes
            seed: Random seed for reproducibility
            
        Returns:
            stats: Dictionary containing training statistics
        """
        if seed is not None:
            set_seed(seed)
        
        episode_rewards = []
        episode_lengths = []
        policy_losses = []
        value_losses = []
        
        pbar = tqdm(range(num_episodes), desc="Training REINFORCE")
        
        for episode in pbar:
            ep_reward, ep_length, p_loss, v_loss = self.train_episode(env)
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            policy_losses.append(p_loss)
            value_losses.append(v_loss)
            
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(episode_rewards[-print_every:])
                avg_length = np.mean(episode_lengths[-print_every:])
                pbar.set_postfix({
                    'avg_reward': f'{avg_reward:.2f}',
                    'avg_length': f'{avg_length:.2f}'
                })
        
        stats = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'policy_losses': policy_losses,
            'value_losses': value_losses
        }
        
        return stats
    
    def evaluate(self, env, num_episodes=10, seed=None):
        """
        Evaluate trained policy
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate
            seed: Random seed
            
        Returns:
            avg_reward: Average reward over episodes
            std_reward: Standard deviation of rewards
        """
        if seed is not None:
            set_seed(seed)
        
        eval_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset(seed=seed + episode if seed else None)
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_probs = self.policy(state_tensor)
                    action = torch.argmax(action_probs).item()
                
                next_state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                state = next_state
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        return avg_reward, std_reward
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'baseline_state_dict': self.baseline.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.baseline.load_state_dict(checkpoint['baseline_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


def test_reinforce():
    """
    Quick test to verify REINFORCE implementation works
    """
    print("=" * 60)
    print("Testing REINFORCE Implementation")
    print("=" * 60)
    
    from environments.cartpole import CartPoleEnv
    env = CartPoleEnv()
    
    agent = REINFORCE(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        hidden_dim=64, 
        lr_policy=1e-3,
        lr_value=1e-3,
        gamma=0.99
    )
    
    print(f"State dim: {env.get_state_dim()}")
    print(f"Action dim: {env.get_action_dim()}")
    print(f"Policy network parameters: {sum(p.numel() for p in agent.policy.parameters())}")
    print(f"Baseline network parameters: {sum(p.numel() for p in agent.baseline.parameters())}")
    
    print("\nTraining for 50 episodes...")
    stats = agent.train(env, num_episodes=50, print_every=10, seed=42)
    
    print(f"\nFirst 10 episode rewards: {stats['episode_rewards'][:10]}")
    print(f"Last 10 episode rewards: {stats['episode_rewards'][-10:]}")
    print(f"Average reward (last 10): {np.mean(stats['episode_rewards'][-10:]):.2f}")
    
    print("\nEvaluating trained policy...")
    avg_reward, std_reward = agent.evaluate(env, num_episodes=10, seed=100)
    print(f"Evaluation: {avg_reward:.2f} ± {std_reward:.2f}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("✓ REINFORCE implementation working!")
    print("=" * 60)


if __name__ == "__main__":
    test_reinforce()