"""
Train SAC Agent on SUMO Environment
====================================
Trains the SAC agent specifically on SUMO dynamics for better performance.
"""

import os
import sys
import time
import numpy as np
import torch

from sumo_traffic_env import SUMOTrafficEnv
from simple_sac import SimpleSACAgent


def train_sac_sumo(num_episodes=10000, save_dir="checkpoints"):
    """Train SAC agent on SUMO environment."""
    
    print("=" * 60)
    print(f"TRAINING SAC on SUMO Environment")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")
    print("=" * 60)
    
    # Create environment
    env = SUMOTrafficEnv(render_mode=None)
    
    # Create agent
    agent = SimpleSACAgent(
        obs_dim=16,
        action_dim=4,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
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
            # Select action
            action = agent.select_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store(obs, action, reward, next_obs, done)
            
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
            agent.save(os.path.join(save_dir, "sumo_sac_best.pt"))
        
        # Periodic save and logging
        if episode % 20 == 0:
            agent.save(os.path.join(save_dir, "sumo_sac_latest.pt"))
            
            avg_reward = np.mean(episode_rewards[-20:])
            avg_crossed = np.mean(episode_crossed[-20:])
            elapsed = time.time() - start_time
            eta = (elapsed / episode) * (num_episodes - episode) / 60
            
            print(f"Ep {episode:5d}/{num_episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Crossed: {avg_crossed:5.1f} | "
                  f"Best: {best_reward:7.2f} | "
                  f"ETA: {eta:.1f}min")
        
        # Checkpoint every 500 episodes
        if episode % 500 == 0:
            agent.save(os.path.join(save_dir, f"sumo_sac_ep{episode}.pt"))
    
    # Final save
    agent.save(os.path.join(save_dir, "sumo_sac_final.pt"))
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
    
    train_sac_sumo(num_episodes=args.episodes)
