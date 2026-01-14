"""
Train Double DQN Agent on SUMO Environment
===========================================
Trains the Double DQN agent specifically on SUMO dynamics for better performance.
"""

import os
import sys
import time
import numpy as np
import torch

from sumo_traffic_env import SUMOTrafficEnv
from double_dqn import DoubleDQNAgent


def train_dqn_sumo(num_episodes=10000, save_dir="checkpoints"):
    """Train Double DQN agent on SUMO environment."""
    
    print("=" * 60)
    print(f"TRAINING Double DQN on SUMO Environment")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")
    print("=" * 60)
    
    # Create environment
    env = SUMOTrafficEnv(render_mode=None)
    
    # Create agent
    agent = DoubleDQNAgent(
        obs_dim=16,
        n_signals=4,
        green_time_options=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_freq=10,
        batch_size=64,
        device="auto"
    )
    
    # Training tracking
    best_reward = float('-inf')
    episode_rewards = []
    episode_crossed = []
    start_time = time.time()
    
    os.makedirs(save_dir, exist_ok=True)
    
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0
        total_crossed = 0
        done = False
        
        while not done:
            # Select action (DQN returns discrete action + continuous mapping)
            action, discrete_action = agent.select_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Enhanced reward shaping for DQN
            shaped_reward = reward
            if info.get('crossed', 0) > 0:
                shaped_reward += info['crossed'] * 2.0
            if info.get('waiting', 0) > 10:
                shaped_reward -= 0.1 * info['waiting']
            
            # Store transition
            agent.store(obs, discrete_action, shaped_reward, next_obs, done)
            
            # Update agent
            if len(agent.replay_buffer) > 256:
                agent.update()
            
            episode_reward += reward
            total_crossed += info.get('crossed', 0)
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_crossed.append(info.get('total_crossed', total_crossed))
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(save_dir, "sumo_dqn_best.pt"))
        
        # Periodic save and logging
        if episode % 20 == 0:
            agent.save(os.path.join(save_dir, "sumo_dqn_latest.pt"))
            
            avg_reward = np.mean(episode_rewards[-20:])
            avg_crossed = np.mean(episode_crossed[-20:])
            elapsed = time.time() - start_time
            eta = (elapsed / episode) * (num_episodes - episode) / 60
            
            print(f"Ep {episode:5d}/{num_episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Crossed: {avg_crossed:5.1f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"ETA: {eta:.1f}min")
        
        # Checkpoint every 500 episodes
        if episode % 500 == 0:
            agent.save(os.path.join(save_dir, f"sumo_dqn_ep{episode}.pt"))
    
    # Final save
    agent.save(os.path.join(save_dir, "sumo_dqn_final.pt"))
    env.close()
    
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final avg crossed: {np.mean(episode_crossed[-100:]):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, nargs="?", default=10000,
                       help="Number of training episodes")
    args = parser.parse_args()
    
    train_dqn_sumo(num_episodes=args.episodes)
