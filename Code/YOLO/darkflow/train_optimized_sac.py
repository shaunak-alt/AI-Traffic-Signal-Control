"""
Train SAC agent with OPTIMIZED reward settings for 5000 episodes.
Goal: Learn efficient green time allocation (e.g. 5 cars -> ~10s).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from traffic_env import TrafficEnv
from simple_sac import SimpleSACAgent


def train_optimized(episodes=5000):
    # Initialize environment
    # Use same settings as simulation for best transfer
    env = TrafficEnv(
        render_mode=None,
        sim_steps_per_action=5,  # Frequent decisions
        max_episode_steps=300,
        min_green=10,
        max_green=60
    )
    
    # Initialize SAC Agent
    agent = SimpleSACAgent(
        obs_dim=16,
        action_dim=4,
        hidden_dim=128,  # Larger network for nuanced control
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,   # Larger batch size for stable learning
        device="auto"
    )
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZED SAC TRAINING on {agent.device}")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Reward: +1 per car, -1 per empty green sec")
    print(f"{'='*60}\n")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    best_reward = float('-inf')
    rewards_history = []
    crossed_history = []
    
    start_time = time.time()
    
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        total_crossed = 0
        
        # Run episode
        while True:
            # Select action
            action = agent.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store in buffer
            agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
            
            # Update agent
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
            
            # Update state
            obs = next_obs
            ep_reward += reward
            total_crossed = info['total_crossed']
            
            if terminated or truncated:
                break
        
        # Logging
        rewards_history.append(ep_reward)
        crossed_history.append(total_crossed)
        
        # Save best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save("checkpoints/optimized_sac_best.pt")
        
        # Frequent updates
        if ep % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_crossed = np.mean(crossed_history[-20:])
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed
            eta = (episodes - ep) / eps_per_sec if eps_per_sec > 0 else 0
            
            print(f"Ep {ep:4d}/{episodes} | Avg Reward: {avg_reward:7.2f} | Avg Crossed: {avg_crossed:5.1f} | ETA: {eta/60:.1f}min")
            
            # Save periodic checkpoint
            agent.save("checkpoints/optimized_sac_latest.pt")
            
    # Final save
    agent.save("checkpoints/optimized_sac_final.pt")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best reward: {best_reward:.2f}")
    print(f"{'='*60}")
    
    env.close()


if __name__ == "__main__":
    import sys
    episodes = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        episodes = int(sys.argv[1])
    train_optimized(episodes)
