"""
Compare SAC vs Double DQN for Traffic Signal Control.

Usage:
    python compare_algorithms.py              # Compare both algorithms
    python compare_algorithms.py --sac        # Train only SAC
    python compare_algorithms.py --dqn        # Train only Double DQN
"""

import os
import time
import numpy as np
import sys
from traffic_env import TrafficEnv
from simple_sac import SimpleSACAgent
from double_dqn import DoubleDQNAgent


def train_sac(episodes=300, verbose=True):
    """Train SAC agent."""
    env = TrafficEnv(render_mode=None, sim_steps_per_action=10, max_episode_steps=100)
    agent = SimpleSACAgent(obs_dim=16, action_dim=4, hidden_dim=64)
    
    rewards = []
    crossed = []
    
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        
        for step in range(100):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.store(obs, action, reward, next_obs, terminated or truncated)
            
            for _ in range(2):
                agent.update()
            
            ep_reward += reward
            obs = next_obs
            if terminated or truncated:
                break
        
        rewards.append(ep_reward)
        crossed.append(info.get("total_crossed", 0))
        
        if verbose and ep % 50 == 0:
            print(f"SAC Episode {ep:4d} | Avg Reward: {np.mean(rewards[-50:]):7.2f} | Crossed: {np.mean(crossed[-50:]):.1f}")
    
    env.close()
    agent.save("checkpoints/sac_model.pt")
    return rewards, crossed, agent


def train_dqn(episodes=300, verbose=True):
    """Train Double DQN agent."""
    env = TrafficEnv(render_mode=None, sim_steps_per_action=10, max_episode_steps=100)
    agent = DoubleDQNAgent(obs_dim=16, hidden_dim=64)
    
    rewards = []
    crossed = []
    
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        
        for step in range(100):
            action, action_idx = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.store(obs, action_idx, reward, next_obs, terminated or truncated)
            agent.update()
            
            ep_reward += reward
            obs = next_obs
            if terminated or truncated:
                break
        
        rewards.append(ep_reward)
        crossed.append(info.get("total_crossed", 0))
        
        if verbose and ep % 50 == 0:
            print(f"DQN Episode {ep:4d} | Avg Reward: {np.mean(rewards[-50:]):7.2f} | Crossed: {np.mean(crossed[-50:]):.1f} | Eps: {agent.epsilon:.3f}")
    
    env.close()
    agent.save("checkpoints/dqn_model.pt")
    return rewards, crossed, agent


def evaluate(agent, agent_name, episodes=10):
    """Evaluate an agent."""
    env = TrafficEnv(render_mode=None, sim_steps_per_action=10, max_episode_steps=200)
    
    rewards = []
    crossed = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        
        for step in range(200):
            if agent_name == "SAC":
                action = agent.select_action(obs, deterministic=True)
            else:
                action, _ = agent.select_action(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        
        rewards.append(ep_reward)
        crossed.append(info.get("total_crossed", 0))
    
    env.close()
    return np.mean(rewards), np.mean(crossed)


def main():
    os.makedirs("checkpoints", exist_ok=True)
    
    train_sac_only = "--sac" in sys.argv
    train_dqn_only = "--dqn" in sys.argv
    episodes = 300
    
    # Parse episode count
    for arg in sys.argv[1:]:
        if arg.isdigit():
            episodes = int(arg)
    
    print(f"\n{'='*60}")
    print("Algorithm Comparison: SAC vs Double DQN")
    print(f"Episodes: {episodes}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Train SAC
    if not train_dqn_only:
        print("\n--- Training SAC ---")
        start = time.time()
        sac_rewards, sac_crossed, sac_agent = train_sac(episodes)
        sac_time = time.time() - start
        sac_eval_reward, sac_eval_crossed = evaluate(sac_agent, "SAC")
        results["SAC"] = {
            "train_time": sac_time,
            "final_train_reward": np.mean(sac_rewards[-50:]),
            "eval_reward": sac_eval_reward,
            "eval_crossed": sac_eval_crossed
        }
        print(f"SAC Training Time: {sac_time:.1f}s")
    
    # Train Double DQN
    if not train_sac_only:
        print("\n--- Training Double DQN ---")
        start = time.time()
        dqn_rewards, dqn_crossed, dqn_agent = train_dqn(episodes)
        dqn_time = time.time() - start
        dqn_eval_reward, dqn_eval_crossed = evaluate(dqn_agent, "DQN")
        results["Double DQN"] = {
            "train_time": dqn_time,
            "final_train_reward": np.mean(dqn_rewards[-50:]),
            "eval_reward": dqn_eval_reward,
            "eval_crossed": dqn_eval_crossed
        }
        print(f"Double DQN Training Time: {dqn_time:.1f}s")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Algorithm':<15} {'Train Time':<12} {'Train Reward':<14} {'Eval Reward':<12} {'Crossed':<10}")
    print(f"{'-'*60}")
    
    for algo, data in results.items():
        print(f"{algo:<15} {data['train_time']:.1f}s{'':<7} {data['final_train_reward']:>7.2f}{'':<7} {data['eval_reward']:>7.2f}{'':<5} {data['eval_crossed']:.1f}")
    
    print(f"{'='*60}")
    
    # Print winner
    if len(results) == 2:
        sac_score = results["SAC"]["eval_reward"]
        dqn_score = results["Double DQN"]["eval_reward"]
        winner = "SAC" if sac_score > dqn_score else "Double DQN"
        print(f"\n🏆 WINNER: {winner} (Eval Reward: {max(sac_score, dqn_score):.2f})")
    
    print(f"\nModels saved in checkpoints/")


if __name__ == "__main__":
    main()
