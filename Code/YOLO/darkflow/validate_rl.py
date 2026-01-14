"""
Validation Script: Compare RL Agent vs Original GST Formula

This script runs both approaches on the same traffic environment and
compares key metrics to validate if RL is working better.

Metrics:
- Total vehicles crossed
- Average waiting time
- Green time efficiency (time used vs time allocated)
"""

import numpy as np
import time
from traffic_env import TrafficEnv
from simple_sac import SimpleSACAgent


def run_gst_formula(env, episodes=10):
    """Run using the original GST formula (rule-based)."""
    results = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_crossed = 0
        total_waiting = 0
        steps = 0
        
        while True:
            # GST Formula: green_time = base + (cars * time_per_car)
            # Normalize to [0,1] for action space
            action = np.zeros(4, dtype=np.float32)
            
            for i in range(4):
                # Count vehicles in this direction (3 lanes each)
                start_idx = i * 3
                vehicles = obs[start_idx] + obs[start_idx + 1] + obs[start_idx + 2]
                
                # GST: 5s base + 1.5s per vehicle, normalized to [0,1]
                # min=10, max=60 => formula_time = 5 + 1.5*vehicles
                formula_time = 5 + 1.5 * vehicles
                action[i] = np.clip((formula_time - 10) / 50, 0, 1)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_crossed = info['total_crossed']
            total_waiting += info['waiting_vehicles']
            steps += 1
            
            if terminated or truncated:
                break
        
        avg_waiting = total_waiting / max(steps, 1)
        results.append({
            'crossed': total_crossed,
            'avg_waiting': avg_waiting,
            'steps': steps
        })
    
    return results


def run_rl_agent(env, agent, episodes=10):
    """Run using trained RL agent."""
    results = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_crossed = 0
        total_waiting = 0
        steps = 0
        
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_crossed = info['total_crossed']
            total_waiting += info['waiting_vehicles']
            steps += 1
            
            if terminated or truncated:
                break
        
        avg_waiting = total_waiting / max(steps, 1)
        results.append({
            'crossed': total_crossed,
            'avg_waiting': avg_waiting,
            'steps': steps
        })
    
    return results


def main():
    print("=" * 60)
    print("VALIDATION: RL Agent vs GST Formula")
    print("=" * 60)
    
    # Create environment
    env = TrafficEnv(
        render_mode=None,
        sim_steps_per_action=5,
        max_episode_steps=300,
        min_green=10,
        max_green=60
    )
    
    # Load trained RL agent
    agent = SimpleSACAgent(obs_dim=16, action_dim=4, hidden_dim=128, device="auto")
    try:
        agent.load("checkpoints/optimized_sac_best.pt")
        print(f"Loaded RL agent from checkpoints/optimized_sac_best.pt")
    except:
        print("WARNING: No trained model found!")
        return
    
    EVAL_EPISODES = 20
    
    # Run GST Formula
    print(f"\n{'='*40}")
    print("Running GST Formula (Rule-Based)...")
    print(f"{'='*40}")
    gst_results = run_gst_formula(env, EVAL_EPISODES)
    
    # Run RL Agent
    print(f"\n{'='*40}")
    print("Running RL Agent (SAC)...")
    print(f"{'='*40}")
    rl_results = run_rl_agent(env, agent, EVAL_EPISODES)
    
    # Calculate averages
    gst_crossed = np.mean([r['crossed'] for r in gst_results])
    gst_waiting = np.mean([r['avg_waiting'] for r in gst_results])
    
    rl_crossed = np.mean([r['crossed'] for r in rl_results])
    rl_waiting = np.mean([r['avg_waiting'] for r in rl_results])
    
    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'GST Formula':<15} {'RL Agent':<15} {'Winner':<10}")
    print("-" * 60)
    
    crossed_winner = "RL ✅" if rl_crossed > gst_crossed else ("GST ✅" if gst_crossed > rl_crossed else "Tie")
    waiting_winner = "RL ✅" if rl_waiting < gst_waiting else ("GST ✅" if gst_waiting < rl_waiting else "Tie")
    
    print(f"{'Avg Vehicles Crossed':<25} {gst_crossed:<15.1f} {rl_crossed:<15.1f} {crossed_winner:<10}")
    print(f"{'Avg Waiting Vehicles':<25} {gst_waiting:<15.2f} {rl_waiting:<15.2f} {waiting_winner:<10}")
    
    improvement = ((rl_crossed - gst_crossed) / gst_crossed * 100) if gst_crossed > 0 else 0
    print(f"\n{'='*60}")
    print(f"RL improvement over GST: {improvement:+.1f}% more vehicles crossed")
    print(f"{'='*60}")
    
    env.close()


if __name__ == "__main__":
    main()
