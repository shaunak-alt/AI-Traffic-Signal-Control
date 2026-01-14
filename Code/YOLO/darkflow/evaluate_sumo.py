"""
SUMO Evaluation Script - Compare GST vs SAC vs DQN
==================================================
Runs headless SUMO simulations and collects proper metrics
for comparison with Pygame results.
"""

import os
import sys
import time
import random
import numpy as np

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please set SUMO_HOME environment variable")

import traci

# Import RL agents
try:
    from simple_sac import SimpleSACAgent
    from double_dqn import DoubleDQNAgent
except ImportError as e:
    print(f"Warning: Could not import RL agents: {e}")

# Configuration
SIM_TIME = 300
EVAL_EPISODES = 10
MIN_GREEN = 10
MAX_GREEN = 60

DIRECTION_TO_EDGE = {
    0: "WC",  # right
    1: "NC",  # down
    2: "EC",  # left
    3: "SC",  # up
}


def get_observation():
    """Get observation vector for RL agents."""
    obs = np.zeros(16, dtype=np.float32)
    for direction in range(4):
        edge_id = DIRECTION_TO_EDGE[direction]
        try:
            vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
            base_idx = direction * 3
            obs[base_idx] = min(len(vehicle_ids), 20)
        except:
            pass
    return obs


def run_gst_episode(seed=None):
    """Run one episode with GST formula."""
    if seed is None:
        seed = int(time.time() * 1000) % 100000
    sumoCfg = os.path.join(os.path.dirname(__file__), "sumo", "intersection.sumocfg")
    traci.start(["sumo", "-c", sumoCfg, "--start", "--quit-on-end", "--no-warnings", "--seed", str(seed)])
    
    total_waiting = 0
    total_crossed = 0
    step = 0
    
    while step < SIM_TIME:
        traci.simulationStep()
        
        # Count waiting vehicles
        for edge in DIRECTION_TO_EDGE.values():
            try:
                total_waiting += traci.edge.getWaitingTime(edge)
            except:
                pass
        
        # Count completed trips
        try:
            total_crossed += traci.simulation.getArrivedNumber()
        except:
            pass
        
        step += 1
    
    # Final count
    try:
        total_crossed += traci.simulation.getArrivedNumber()
    except:
        pass
    
    traci.close()
    
    return {
        'crossed': total_crossed,
        'avg_waiting': total_waiting / max(SIM_TIME, 1),
        'throughput': total_crossed / SIM_TIME
    }


def run_rl_episode(agent, agent_type="SAC", seed=None):
    """Run one episode with RL agent."""
    if seed is None:
        seed = int(time.time() * 1000) % 100000
    sumoCfg = os.path.join(os.path.dirname(__file__), "sumo", "intersection.sumocfg")
    traci.start(["sumo", "-c", sumoCfg, "--start", "--quit-on-end", "--no-warnings", "--seed", str(seed)])
    
    tlId = traci.trafficlight.getIDList()[0] if traci.trafficlight.getIDList() else "C"
    
    total_waiting = 0
    total_crossed = 0
    step = 0
    current_phase = 0
    phase_timer = 20
    
    while step < SIM_TIME:
        traci.simulationStep()
        
        # Count metrics
        for edge in DIRECTION_TO_EDGE.values():
            try:
                total_waiting += traci.edge.getWaitingTime(edge)
            except:
                pass
        
        try:
            total_crossed += traci.simulation.getArrivedNumber()
        except:
            pass
        
        # RL decision every 25 steps (one phase cycle)
        phase_timer -= 1
        if phase_timer <= 0:
            obs = get_observation()
            
            try:
                if agent_type == "SAC":
                    action = agent.select_action(obs, deterministic=True)
                    green_time = int(action[current_phase % 4] * (MAX_GREEN - MIN_GREEN) + MIN_GREEN)
                else:
                    action, _ = agent.select_action(obs, deterministic=True)
                    green_time = int(action[current_phase % 4] * (MAX_GREEN - MIN_GREEN) + MIN_GREEN)
            except:
                green_time = 20
            
            phase_timer = max(MIN_GREEN, min(MAX_GREEN, green_time))
            current_phase = (current_phase + 1) % 8  # 4 green + 4 yellow phases
            
            try:
                traci.trafficlight.setPhase(tlId, current_phase)
            except:
                pass
        
        step += 1
    
    traci.close()
    
    return {
        'crossed': total_crossed,
        'avg_waiting': total_waiting / max(SIM_TIME, 1),
        'throughput': total_crossed / SIM_TIME
    }


def calculate_stats(results):
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
    }


def main():
    print("=" * 70)
    print("SUMO EVALUATION - 4-Phase Traffic Light")
    print("=" * 70)
    print(f"Episodes per algorithm: {EVAL_EPISODES}")
    print(f"Simulation time: {SIM_TIME}s per episode")
    print("=" * 70)
    
    # Load RL agents
    sac_agent = None
    dqn_agent = None
    
    try:
        sac_agent = SimpleSACAgent(obs_dim=16, action_dim=4, hidden_dim=128, device="cpu")
        # SAC: Pygame-trained (better transfer learning)
        for path in ["checkpoints/sac_extended_best.pt", "checkpoints/optimized_sac_best.pt"]:
            if os.path.exists(path):
                sac_agent.load(path)
                print(f"Loaded SAC (Pygame-trained): {path}")
                break
    except Exception as e:
        print(f"SAC load failed: {e}")
    
    try:
        dqn_agent = DoubleDQNAgent(obs_dim=16, hidden_dim=128, device="cpu")
        # DQN: SUMO-trained
        for path in ["checkpoints/sumo_dqn_best.pt", "checkpoints/sumo_dqn_final.pt"]:
            if os.path.exists(path):
                dqn_agent.load(path)
                print(f"Loaded DQN (SUMO-trained): {path}")
                break
    except Exception as e:
        print(f"DQN load failed: {e}")
    
    # Run evaluations
    print("\n[1/3] Evaluating GST Baseline...")
    gst_results = []
    for ep in range(EVAL_EPISODES):
        seed = 1000 + ep  # Different seed per episode
        result = run_gst_episode(seed=seed)
        gst_results.append(result)
        print(f"  Episode {ep+1}: {result['crossed']} crossed (seed={seed})")
    gst_stats = calculate_stats(gst_results)
    
    sac_stats = None
    if sac_agent:
        print("\n[2/3] Evaluating SAC Agent...")
        sac_results = []
        for ep in range(EVAL_EPISODES):
            seed = 1000 + ep  # Same seeds as GST for fair comparison
            result = run_rl_episode(sac_agent, "SAC", seed=seed)
            sac_results.append(result)
            print(f"  Episode {ep+1}: {result['crossed']} crossed (seed={seed})")
        sac_stats = calculate_stats(sac_results)
    
    dqn_stats = None
    if dqn_agent:
        print("\n[3/3] Evaluating DQN Agent...")
        dqn_results = []
        for ep in range(EVAL_EPISODES):
            seed = 1000 + ep  # Same seeds as GST for fair comparison
            result = run_rl_episode(dqn_agent, "DQN", seed=seed)
            dqn_results.append(result)
            print(f"  Episode {ep+1}: {result['crossed']} crossed (seed={seed})")
        dqn_stats = calculate_stats(dqn_results)
    
    # Print results
    print("\n")
    print("=" * 70)
    print("SUMO RESULTS TABLE")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Vehicles Crossed':<20} {'Avg Waiting':<18} {'Throughput':<15}")
    print("-" * 70)
    
    if gst_stats:
        print(f"{'GST Baseline':<20} {gst_stats['crossed_mean']:.1f} ± {gst_stats['crossed_std']:.1f}       {gst_stats['waiting_mean']:.1f}         {gst_stats['throughput_mean']:.4f}")
    
    if sac_stats:
        print(f"{'SAC (MLP)':<20} {sac_stats['crossed_mean']:.1f} ± {sac_stats['crossed_std']:.1f}       {sac_stats['waiting_mean']:.1f}         {sac_stats['throughput_mean']:.4f}")
    
    if dqn_stats:
        print(f"{'Double DQN':<20} {dqn_stats['crossed_mean']:.1f} ± {dqn_stats['crossed_std']:.1f}       {dqn_stats['waiting_mean']:.1f}         {dqn_stats['throughput_mean']:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
