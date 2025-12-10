import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.networks.policy_network import DiscretePolicyNetwork, ContinuousPolicyNetwork
from src.networks.value_network import DiscreteValueNetwork, ContinuousValueNetwork
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
        device='cpu',
        continuous=False
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        self.continuous = continuous

        if continuous:
            self.policy = ContinuousPolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
            self.value = ContinuousValueNetwork(state_dim, hidden_dim).to(device)
            self.value.h3.bias.data.fill_(-500.0)
            self.value.h3.weight.data.mul_(0.01)
        else:
            self.policy = DiscretePolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
            self.value = DiscreteValueNetwork(state_dim, hidden_dim).to(device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_value)

        self.reset_episode()
        
    def reset_episode(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def select_action(self, state):
        """
        Select action using current policy
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, entropy = self.policy.sample_action(state_tensor)

        if self.continuous:
            return action.detach().cpu().numpy()[0], log_prob
        else:
            return action.item(), log_prob
    
    def store_transition(self, state, action, reward, log_prob):
        """
        Store transition data for current episode
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def update(self):
        """
        Update policy and value using collected episode
        """
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)
        
        returns = compute_returns(self.rewards, self.gamma)
        returns = torch.FloatTensor(returns).to(self.device)
        
        values = self.value(states).squeeze()
        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        value_loss = ((returns - values) ** 2).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train_episode(self, env):
        """
        Run one episode and update policy
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
                    if self.continuous:
                        mean, log_std = self.policy(state_tensor)
                        action = mean.detach().cpu().numpy()[0]
                    else:
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
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
    