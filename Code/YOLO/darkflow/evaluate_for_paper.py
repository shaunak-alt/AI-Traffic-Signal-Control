"""
Comprehensive Evaluation Script for Research Paper

Compares:
1. SAC (MLP) - Soft Actor-Critic with MLP
2. Double DQN - Double Deep Q-Network
3. GST Formula - Rule-based baseline

Output: Metrics suitable for academic research paper
"""

import numpy as np
import time
import os
from traffic_env import TrafficEnv
from simple_sac import SimpleSACAgent
from double_dqn import DoubleDQNAgent


def run_gst_baseline(env, episodes=20):
    """Run GST formula (rule-based baseline)."""
    results = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_crossed = 0
        total_waiting = 0
        total_steps = 0
        
        while True:
            # GST Formula: green_time = 5 + (vehicles × 1.5)
            action = np.zeros(4, dtype=np.float32)
            for i in range(4):
                start_idx = i * 3
                vehicles = obs[start_idx] + obs[start_idx + 1] + obs[start_idx + 2]
                formula_time = 5 + 1.5 * vehicles
                action[i] = np.clip((formula_time - 10) / 50, 0, 1)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_crossed = info['total_crossed']
            total_waiting += info['waiting_vehicles']
            total_steps += 1
            
            if terminated or truncated:
                break
        
        results.append({
            'crossed': total_crossed,
            'avg_waiting': total_waiting / max(total_steps, 1),
            'throughput': total_crossed / (total_steps * env.sim_steps_per_action) if total_steps > 0 else 0
        })
    
    return results


def run_proportional_allocation(env, episodes=20, g_min=10, g_max=60):
    """
    Proportional Allocation Formula:
    g_i = g_min + (d_i / sum(d_j)) * (g_max - g_min)
    
    This allocates green time proportionally based on observed demand.
    """
    results = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_crossed = 0
        total_waiting = 0
        total_steps = 0
        
        while True:
            # Get demand for each phase (sum of vehicles in 3 lanes per direction)
            demands = []
            for i in range(4):
                start_idx = i * 3
                d_i = obs[start_idx] + obs[start_idx + 1] + obs[start_idx + 2]
                demands.append(d_i)
            
            total_demand = sum(demands)
            
            # Calculate proportional green times
            action = np.zeros(4, dtype=np.float32)
            for i in range(4):
                if total_demand > 0:
                    proportion = demands[i] / total_demand
                    green_time = g_min + proportion * (g_max - g_min)
                else:
                    green_time = g_min  # Equal allocation when no demand
                
                # Normalize to [0, 1] for action space
                action[i] = np.clip((green_time - g_min) / (g_max - g_min), 0, 1)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_crossed = info['total_crossed']
            total_waiting += info['waiting_vehicles']
            total_steps += 1
            
            if terminated or truncated:
                break
        
        results.append({
            'crossed': total_crossed,
            'avg_waiting': total_waiting / max(total_steps, 1),
            'throughput': total_crossed / (total_steps * env.sim_steps_per_action) if total_steps > 0 else 0
        })
    
    return results


def run_sac_agent(env, episodes=20):
    """Run trained SAC (MLP) agent."""
    agent = SimpleSACAgent(obs_dim=16, action_dim=4, hidden_dim=128, device="auto")
    
    # Try to load best model (prefer 10K extended training)
    model_paths = ["checkpoints/sac_extended_best.pt", "checkpoints/optimized_sac_best.pt", "checkpoints/best_model.pt"]
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            agent.load(path)
            print(f"  Loaded SAC model: {path}")
            loaded = True
            break
    
    if not loaded:
        print("  WARNING: No SAC model found!")
        return None
    
    results = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total_crossed = 0
        total_waiting = 0
        total_steps = 0
        
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_crossed = info['total_crossed']
            total_waiting += info['waiting_vehicles']
            total_steps += 1
            
            if terminated or truncated:
                break
        
        results.append({
            'crossed': total_crossed,
            'avg_waiting': total_waiting / max(total_steps, 1),
            'throughput': total_crossed / (total_steps * env.sim_steps_per_action) if total_steps > 0 else 0
        })
    
    return results


def run_dqn_agent(env, episodes=20):
    """Run trained Double DQN agent."""
    agent = DoubleDQNAgent(obs_dim=16, hidden_dim=128, device="auto")
    
    model_paths = ["checkpoints/dqn_model.pt", "checkpoints/dqn_best.pt"]
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            agent.load(path)
            print(f"  Loaded DQN model: {path}")
            loaded = True
            break
    
    if not loaded:
        print("  WARNING: No DQN model found!")
        return None
    
    results = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total_crossed = 0
        total_waiting = 0
        total_steps = 0
        
        while True:
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_crossed = info['total_crossed']
            total_waiting += info['waiting_vehicles']
            total_steps += 1
            
            if terminated or truncated:
                break
        
        results.append({
            'crossed': total_crossed,
            'avg_waiting': total_waiting / max(total_steps, 1),
            'throughput': total_crossed / (total_steps * env.sim_steps_per_action) if total_steps > 0 else 0
        })
    
    return results


def calculate_stats(results):
    """Calculate mean and std for results."""
    if not results:
        return None
    
    crossed = [r['crossed'] for r in results]
    waiting = [r['avg_waiting'] for r in results]
    throughput = [r['throughput'] for r in results]
    
    return {
        'crossed_mean': np.mean(crossed),
        'crossed_std': np.std(crossed),
        'waiting_mean': np.mean(waiting),
        'waiting_std': np.std(waiting),
        'throughput_mean': np.mean(throughput),
        'throughput_std': np.std(throughput)
    }


def main():
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION FOR RESEARCH PAPER")
    print("=" * 70)
    print(f"Environment: Traffic Signal Control Simulation")
    print(f"Episode Length: 300 time steps")
    print(f"Evaluation Episodes: 20 each")
    print("=" * 70)
    
    # Create environment
    env = TrafficEnv(
        render_mode=None,
        sim_steps_per_action=5,
        max_episode_steps=300,
        min_green=10,
        max_green=60
    )
    
    EVAL_EPISODES = 20
    
    # Run evaluations
    print("\n[1/4] Evaluating GST Baseline (Rule-Based)...")
    gst_results = run_gst_baseline(env, EVAL_EPISODES)
    gst_stats = calculate_stats(gst_results)
    
    print("[2/4] Evaluating Proportional Allocation (Academic Formula)...")
    prop_results = run_proportional_allocation(env, EVAL_EPISODES)
    prop_stats = calculate_stats(prop_results)
    
    print("[3/4] Evaluating SAC (MLP) Agent...")
    sac_results = run_sac_agent(env, EVAL_EPISODES)
    sac_stats = calculate_stats(sac_results) if sac_results else None
    
    print("[4/4] Evaluating Double DQN Agent...")
    dqn_results = run_dqn_agent(env, EVAL_EPISODES)
    dqn_stats = calculate_stats(dqn_results) if dqn_results else None
    
    # Print results table for research paper
    print("\n")
    print("=" * 85)
    print("RESULTS TABLE (For Research Paper)")
    print("=" * 85)
    print(f"{'Algorithm':<25} {'Vehicles Crossed':<20} {'Avg Waiting':<18} {'Throughput (v/s)':<15}")
    print("-" * 85)
    
    if gst_stats:
        print(f"{'GST (5 + 1.5v)':<25} {gst_stats['crossed_mean']:.1f} ± {gst_stats['crossed_std']:.1f}       {gst_stats['waiting_mean']:.1f} ± {gst_stats['waiting_std']:.1f}       {gst_stats['throughput_mean']:.4f}")
    
    if prop_stats:
        print(f"{'Proportional Alloc.':<25} {prop_stats['crossed_mean']:.1f} ± {prop_stats['crossed_std']:.1f}       {prop_stats['waiting_mean']:.1f} ± {prop_stats['waiting_std']:.1f}       {prop_stats['throughput_mean']:.4f}")
    
    if sac_stats:
        print(f"{'SAC (MLP)':<25} {sac_stats['crossed_mean']:.1f} ± {sac_stats['crossed_std']:.1f}       {sac_stats['waiting_mean']:.1f} ± {sac_stats['waiting_std']:.1f}       {sac_stats['throughput_mean']:.4f}")
    
    if dqn_stats:
        print(f"{'Double DQN':<25} {dqn_stats['crossed_mean']:.1f} ± {dqn_stats['crossed_std']:.1f}       {dqn_stats['waiting_mean']:.1f} ± {dqn_stats['waiting_std']:.1f}       {dqn_stats['throughput_mean']:.4f}")
    
    print("=" * 85)
    
    # Calculate improvements
    print("\n")
    print("IMPROVEMENT ANALYSIS")
    print("-" * 70)
    
    if gst_stats and sac_stats:
        sac_improvement = ((sac_stats['crossed_mean'] - gst_stats['crossed_mean']) / gst_stats['crossed_mean']) * 100
        print(f"SAC vs GST Baseline:  {sac_improvement:+.1f}% vehicles crossed")
    
    if gst_stats and dqn_stats:
        dqn_improvement = ((dqn_stats['crossed_mean'] - gst_stats['crossed_mean']) / gst_stats['crossed_mean']) * 100
        print(f"DQN vs GST Baseline:  {dqn_improvement:+.1f}% vehicles crossed")
    
    if sac_stats and dqn_stats:
        sac_vs_dqn = ((sac_stats['crossed_mean'] - dqn_stats['crossed_mean']) / dqn_stats['crossed_mean']) * 100
        print(f"SAC vs DQN:           {sac_vs_dqn:+.1f}% vehicles crossed")
    
    if prop_stats and sac_stats:
        sac_vs_prop = ((sac_stats['crossed_mean'] - prop_stats['crossed_mean']) / prop_stats['crossed_mean']) * 100
        print(f"SAC vs Proportional:  {sac_vs_prop:+.1f}% vehicles crossed")
    
    print("\n" + "=" * 85)
    
    env.close()


if __name__ == "__main__":
    main()
