"""
Soft Actor-Critic (SAC) Agent with GNN Encoder for Traffic Signal Control.

This module implements SAC with:
- GNN-based policy and value networks
- Automatic entropy tuning
- Experience replay buffer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict, Optional, List
from collections import deque
import random

from gnn_encoder import TrafficGNNEncoder


class ReplayBuffer:
    """Experience replay buffer for off-policy training."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


class GNNActor(nn.Module):
    """
    Actor network with GNN encoder.
    
    Outputs mean and log_std of action distribution.
    """
    
    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        hidden_dim: int = 256,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 64,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # GNN encoder
        self.gnn_encoder = TrafficGNNEncoder(
            obs_dim=obs_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim
        )
        
        gnn_out_dim = self.gnn_encoder.output_dim  # 4 * gnn_output_dim
        
        # MLP head
        self.fc1 = nn.Linear(gnn_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action distribution parameters."""
        # GNN encoding
        x = self.gnn_encoder(obs)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()
        
        # Squash to [0, 1] using sigmoid
        action = torch.sigmoid(x_t)
        
        # Log probability with correction for squashing
        log_prob = normal.log_prob(x_t)
        # Correction for sigmoid (different from tanh)
        log_prob -= torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for environment interaction."""
        mean, log_std = self.forward(obs)
        
        if deterministic:
            return torch.sigmoid(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.sigmoid(x_t)


class GNNCritic(nn.Module):
    """
    Critic network (Q-function) with GNN encoder.
    
    Twin Q-networks for reducing overestimation.
    """
    
    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        hidden_dim: int = 256,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 64
    ):
        super().__init__()
        
        # GNN encoder (shared architecture, separate weights)
        self.gnn_encoder1 = TrafficGNNEncoder(
            obs_dim=obs_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim
        )
        self.gnn_encoder2 = TrafficGNNEncoder(
            obs_dim=obs_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim
        )
        
        gnn_out_dim = self.gnn_encoder1.output_dim
        
        # Q1 network
        self.q1_fc1 = nn.Linear(gnn_out_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_head = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.q2_fc1 = nn.Linear(gnn_out_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values for both critics."""
        # GNN encoding
        x1 = self.gnn_encoder1(obs)
        x2 = self.gnn_encoder2(obs)
        
        # Concatenate with action
        x1 = torch.cat([x1, action], dim=-1)
        x2 = torch.cat([x2, action], dim=-1)
        
        # Q1
        q1 = F.relu(self.q1_fc1(x1))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_head(q1)
        
        # Q2
        q2 = F.relu(self.q2_fc1(x2))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_head(q2)
        
        return q1, q2


class SACGNNAgent:
    """
    Soft Actor-Critic Agent with GNN-based networks.
    
    Features:
    - GNN encoder for traffic state
    - Twin Q-networks
    - Automatic entropy tuning
    """
    
    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        hidden_dim: int = 256,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        buffer_size: int = 100000,
        batch_size: int = 256,
        device: str = "auto"
    ):
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy = auto_entropy
        
        # Networks
        self.actor = GNNActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim
        ).to(self.device)
        
        self.critic = GNNCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim
        ).to(self.device)
        
        self.critic_target = GNNCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim
        ).to(self.device)
        
        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Entropy tuning
        if auto_entropy:
            self.target_entropy = -action_dim  # Heuristic
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.training_steps = 0
        
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor.get_action(obs_tensor, deterministic)
            return action.cpu().numpy().flatten()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Perform one update step."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update entropy coefficient
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.training_steps += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "q_value": q.mean().item()
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.auto_entropy else None,
            "training_steps": self.training_steps
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.auto_entropy and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp().item()
        self.training_steps = checkpoint["training_steps"]


# Quick test
if __name__ == "__main__":
    print("Testing SAC + GNN Agent...")
    
    # Create agent
    agent = SACGNNAgent(
        obs_dim=16,
        action_dim=4,
        hidden_dim=128,
        gnn_hidden_dim=32,
        gnn_output_dim=32
    )
    print(f"Agent created on device: {agent.device}")
    
    # Test action selection
    obs = np.random.randn(16).astype(np.float32)
    action = agent.select_action(obs)
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {action.shape}, values: {action}")
    
    # Test with dummy transitions
    for i in range(300):
        next_obs = np.random.randn(16).astype(np.float32)
        reward = np.random.randn()
        done = False
        agent.store_transition(obs, action, reward, next_obs, done)
        obs = next_obs
        action = agent.select_action(obs)
    
    # Test update
    stats = agent.update()
    print(f"Update stats: {stats}")
    
    # Test save/load
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pt")
        agent.save(path)
        print(f"Model saved to {path}")
        agent.load(path)
        print("Model loaded successfully")
    
    print("All tests passed!")
