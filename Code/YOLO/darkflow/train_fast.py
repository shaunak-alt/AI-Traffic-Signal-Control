"""
Fast Training Script - Optimized for quick results on CPU.

Usage:
    python train_fast.py              # Train 500 episodes (takes ~5-10 minutes)
    python train_fast.py --eval       # Evaluate trained model
"""

import os
import time
import numpy as np
from traffic_env import TrafficEnv
from simple_sac import SimpleSACAgent


def train(episodes=500, render=False):
    """Train the agent."""
    env = TrafficEnv(
        render_mode="human" if render else None,
        sim_steps_per_action=10,  # More simulation steps per action
        max_episode_steps=100,    # Short episodes for faster iteration
        min_green=10,
        max_green=60
    )
    
    agent = SimpleSACAgent(obs_dim=16, action_dim=4, hidden_dim=64, batch_size=64)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    best_reward = float('-inf')
    rewards_history = []
    
    print(f"\n{'='*50}")
    print("Fast SAC Training for Traffic Signal Control")
    print(f"{'='*50}")
    print(f"Episodes: {episodes}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        
        for step in range(100):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.store(obs, action, reward, next_obs, terminated or truncated)
            
            # Update multiple times per step for faster learning
            for _ in range(2):
                agent.update()
            
            ep_reward += reward
            obs = next_obs
            
            if terminated or truncated:
                break
        
        rewards_history.append(ep_reward)
        
        # Save best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save("checkpoints/best_model.pt")
        
        # Print progress every 10 episodes
        if ep % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            elapsed = time.time() - start_time
            print(f"Episode {ep:4d} | Avg Reward: {avg_reward:7.2f} | Best: {best_reward:7.2f} | Crossed: {info.get('total_crossed', 0):3d} | Time: {elapsed:.0f}s")
        
        # Save checkpoint
        if ep % 100 == 0:
            agent.save(f"checkpoints/checkpoint_{ep}.pt")
    
    agent.save("checkpoints/final_model.pt")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Training Complete! ({elapsed:.0f} seconds)")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Final Avg Reward: {np.mean(rewards_history[-50:]):.2f}")
    print(f"Model saved to: checkpoints/best_model.pt")
    print(f"{'='*50}")
    
    env.close()
    return agent


def evaluate(checkpoint="checkpoints/best_model.pt", episodes=10, render=True):
    """Evaluate trained model."""
    env = TrafficEnv(
        render_mode="human" if render else None,
        sim_steps_per_action=10,
        max_episode_steps=200,  # Longer episodes for evaluation
        min_green=10,
        max_green=60
    )
    
    agent = SimpleSACAgent(obs_dim=16, action_dim=4)
    
    if os.path.exists(checkpoint):
        agent.load(checkpoint)
        print(f"Loaded model from {checkpoint}")
    else:
        print("No checkpoint found, using random agent")
    
    print(f"\n{'='*50}")
    print("Evaluating Traffic Signal Controller")
    print(f"{'='*50}\n")
    
    total_rewards = []
    total_crossed = []
    
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        
        for step in range(200):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(ep_reward)
        total_crossed.append(info.get("total_crossed", 0))
        print(f"Episode {ep} | Reward: {ep_reward:7.2f} | Vehicles Crossed: {info.get('total_crossed', 0)}")
    
    print(f"\n{'='*50}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Vehicles Crossed: {np.mean(total_crossed):.1f}")
    print(f"{'='*50}")
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if "--eval" in sys.argv:
        render = "--render" in sys.argv or "--headless" not in sys.argv
        evaluate(render=render)
    else:
        episodes = 500
        for arg in sys.argv[1:]:
            if arg.isdigit():
                episodes = int(arg)
        render = "--render" in sys.argv
        train(episodes=episodes, render=render)
