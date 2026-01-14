"""
Train Double DQN with the same improved settings as SAC.
"""

import os
import time
import numpy as np
from traffic_env import TrafficEnv
from double_dqn import DoubleDQNAgent


def train_dqn(episodes=1000):
    env = TrafficEnv(
        render_mode=None,
        sim_steps_per_action=15,
        max_episode_steps=200,
        min_green=10,
        max_green=60
    )
    
    agent = DoubleDQNAgent(
        obs_dim=16,
        hidden_dim=128,  # Match SAC
        batch_size=128,
        epsilon_decay=0.998,  # Slower decay
        device="auto"
    )
    
    print(f"\n{'='*60}")
    print(f"TRAINING Double DQN on {agent.device}")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
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
        
        for step in range(200):
            action, action_idx = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Enhanced reward
            crossed = info.get("total_crossed", 0) - total_crossed
            waiting = info.get("waiting_vehicles", 0)
            if crossed > 0:
                reward += crossed * 3.0
            elif waiting > 0:
                reward -= 0.1 * waiting
            
            agent.store(obs, action_idx, reward, next_obs, terminated or truncated)
            
            for _ in range(4):
                agent.update()
            
            ep_reward += reward
            total_crossed = info.get("total_crossed", 0)
            obs = next_obs
            
            if terminated or truncated:
                break
        
        rewards_history.append(ep_reward)
        crossed_history.append(total_crossed)
        
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save("checkpoints/dqn_model.pt")
        
        if ep % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_crossed = np.mean(crossed_history[-20:])
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed
            eta = (episodes - ep) / eps_per_sec if eps_per_sec > 0 else 0
            print(f"Ep {ep:4d}/{episodes} | Reward: {avg_reward:7.2f} | Crossed: {avg_crossed:5.1f} | Eps: {agent.epsilon:.3f} | ETA: {eta/60:.1f}min")
    
    agent.save("checkpoints/dqn_model.pt")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best reward: {best_reward:.2f}")
    print(f"{'='*60}")
    
    env.close()


if __name__ == "__main__":
    import sys
    episodes = 1000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        episodes = int(sys.argv[1])
    train_dqn(episodes)
