"""
Extended training script for SAC (MLP) with 10000 episodes.
"""

import os
import time
import numpy as np
from traffic_env import TrafficEnv
from simple_sac import SimpleSACAgent


def train_extended(episodes=10000):
    env = TrafficEnv(
        render_mode=None,
        sim_steps_per_action=5,
        max_episode_steps=300,
        min_green=10,
        max_green=60
    )
    
    agent = SimpleSACAgent(
        obs_dim=16,
        action_dim=4,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        device="auto"
    )
    
    print(f"\n{'='*60}")
    print(f"SAC (MLP) EXTENDED TRAINING on {agent.device}")
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
        
        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
            
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
            
            obs = next_obs
            ep_reward += reward
            total_crossed = info['total_crossed']
            
            if terminated or truncated:
                break
        
        rewards_history.append(ep_reward)
        crossed_history.append(total_crossed)
        
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save("checkpoints/sac_extended_best.pt")
        
        if ep % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_crossed = np.mean(crossed_history[-50:])
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed
            eta = (episodes - ep) / eps_per_sec if eps_per_sec > 0 else 0
            
            print(f"Ep {ep:5d}/{episodes} | Reward: {avg_reward:7.2f} | Crossed: {avg_crossed:5.1f} | Best: {best_reward:7.2f} | ETA: {eta/60:.1f}min")
            
            agent.save("checkpoints/sac_extended_latest.pt")
    
    agent.save("checkpoints/sac_extended_final.pt")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final avg reward: {np.mean(rewards_history[-100:]):.2f}")
    print(f"{'='*60}")
    
    env.close()


if __name__ == "__main__":
    import sys
    episodes = 10000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        episodes = int(sys.argv[1])
    train_extended(episodes)
