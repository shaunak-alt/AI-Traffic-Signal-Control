"""
Double DQN Agent for Traffic Signal Control.

Double DQN fixes overestimation bias in regular DQN by using:
- Online network to SELECT the best action
- Target network to EVALUATE that action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network for discrete action space."""
    
    def __init__(self, obs_dim=16, n_actions=16, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DoubleDQNAgent:
    """
    Double DQN Agent for Traffic Signal Control.
    
    Uses discretized action space:
    - 4 signals × 11 green time choices = 44 discrete actions
    - Green times: [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] seconds
    """
    
    def __init__(
        self,
        obs_dim=16,
        n_signals=4,
        green_time_options=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],  # 11 options for finer control
        hidden_dim=64,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_freq=10,
        batch_size=64,
        device="auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.n_signals = n_signals
        self.green_time_options = green_time_options
        self.n_actions = n_signals * len(green_time_options)  # 4 × 4 = 16
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.online_net = QNetwork(obs_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.steps = 0
        
    def action_to_signal_times(self, action_idx):
        """Convert discrete action index to signal green times."""
        # action_idx tells which signal to modify and which green time to use
        signal_idx = action_idx // len(self.green_time_options)
        time_idx = action_idx % len(self.green_time_options)
        
        # Default green times
        green_times = [0.5] * self.n_signals  # Normalized [0,1]
        
        # Set the selected signal's green time
        green_times[signal_idx] = time_idx / (len(self.green_time_options) - 1)
        
        return np.array(green_times, dtype=np.float32)
    
    def select_action(self, obs, deterministic=False):
        """Select action using epsilon-greedy."""
        if not deterministic and random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values = self.online_net(obs_t)
                action_idx = q_values.argmax(dim=1).item()
        
        return self.action_to_signal_times(action_idx), action_idx
    
    def store(self, state, action_idx, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def update(self):
        """Perform Double DQN update."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online network to SELECT action, target network to EVALUATE
        with torch.no_grad():
            # Online network selects best action
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # Target network evaluates that action
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {"loss": loss.item(), "epsilon": self.epsilon}
    
    def save(self, path):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps = ckpt["steps"]


if __name__ == "__main__":
    print("Testing Double DQN Agent...")
    agent = DoubleDQNAgent()
    print(f"Device: {agent.device}")
    print(f"Number of discrete actions: {agent.n_actions}")
    
    obs = np.random.randn(16).astype(np.float32)
    action, action_idx = agent.select_action(obs)
    print(f"Action (signal times): {action}")
    print(f"Action index: {action_idx}")
    
    # Fill buffer
    for _ in range(100):
        next_obs = np.random.randn(16).astype(np.float32)
        agent.store(obs, action_idx, 0.0, next_obs, False)
        obs = next_obs
        action, action_idx = agent.select_action(obs)
    
    stats = agent.update()
    print(f"Update stats: {stats}")
    print("Test passed!")
