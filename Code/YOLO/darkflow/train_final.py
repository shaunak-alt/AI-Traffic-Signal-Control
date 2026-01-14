"""
Final Training Script - Properly trains the model with improved reward function.

The key improvement: reward is based on EFFICIENCY (vehicles crossed per second of green time)
This teaches the agent to give short green times for few cars, long for many.
"""

import os
import time
import numpy as np
from traffic_env import TrafficEnv
from simple_sac import SimpleSACAgent


def train_final(episodes=1000):
    """Train with improved reward function."""
    
    env = TrafficEnv(
        render_mode=None,
        sim_steps_per_action=15,  # More steps per action
        max_episode_steps=200,     # Longer episodes
        min_green=10,
        max_green=60
    )
    
    agent = SimpleSACAgent(
        obs_dim=16, 
        action_dim=4, 
        hidden_dim=128,  # Larger network
        batch_size=128,   # Larger batch
        device="auto"     # Use GPU if available
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL TRAINING - SAC on {agent.device}")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Network: 128 hidden, batch=128")
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
        total_green_time = 0
        
        for step in range(200):
            action = agent.select_action(obs)
            
            # Track green time used
            green_time = action[env.next_green] * (60 - 10) + 10
            total_green_time += green_time
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Modify reward to be more informative
            crossed = info.get("total_crossed", 0) - total_crossed
            waiting = info.get("waiting_vehicles", 0)
            
            # New reward: reward crossing, penalize long green with few cars
            vehicles_in_lane = obs[env.current_green * 3:(env.current_green + 1) * 3].sum()
            efficiency_bonus = 0
            if crossed > 0:
                efficiency_bonus = crossed * 3.0  # Big reward for crossing
            elif waiting > 0:
                # Penalty proportional to unused green time potential
                efficiency_bonus = -0.1 * waiting
            
            modified_reward = reward + efficiency_bonus
            
            agent.store(obs, action, modified_reward, next_obs, terminated or truncated)
            
            # Multiple updates per step for faster learning
            for _ in range(4):
                agent.update()
            
            ep_reward += reward
            total_crossed = info.get("total_crossed", 0)
            obs = next_obs
            
            if terminated or truncated:
                break
        
        rewards_history.append(ep_reward)
        crossed_history.append(total_crossed)
        
        # Save best
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save("checkpoints/best_model.pt")
            agent.save("checkpoints/sac_model.pt")
        
        # Progress
        if ep % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_crossed = np.mean(crossed_history[-20:])
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed
            eta = (episodes - ep) / eps_per_sec if eps_per_sec > 0 else 0
            print(f"Ep {ep:4d}/{episodes} | Reward: {avg_reward:7.2f} | Crossed: {avg_crossed:5.1f} | Best: {best_reward:7.2f} | ETA: {eta/60:.1f}min")
    
    agent.save("checkpoints/final_model.pt")
    agent.save("checkpoints/sac_model.pt")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final avg reward: {np.mean(rewards_history[-50:]):.2f}")
    print(f"Final avg crossed: {np.mean(crossed_history[-50:]):.1f}")
    print(f"Model saved: checkpoints/best_model.pt")
    print(f"{'='*60}")
    
    env.close()


if __name__ == "__main__":
    import sys
    episodes = 1000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        episodes = int(sys.argv[1])
    train_final(episodes)
