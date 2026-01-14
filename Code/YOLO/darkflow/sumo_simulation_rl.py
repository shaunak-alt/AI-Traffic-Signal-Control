"""
SUMO Traffic Simulation with RL Control (SAC/DQN)
================================================
Uses trained RL agents to control traffic signals in SUMO.
Matches the structure of sumo_simulation.py but replaces GST formula with RL decisions.

Usage:
    python sumo_simulation_rl.py          # SAC with GUI
    python sumo_simulation_rl.py --dqn    # DQN with GUI
    python sumo_simulation_rl.py --nogui  # Headless mode
"""

import os
import sys
import math
import time
import argparse
import numpy as np

# Add SUMO tools to path
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

# =============================================================================
# CONFIGURATION - EXACT VALUES FROM PYGAME
# =============================================================================

defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

noOfSignals = 4
simTime = 300

DIRECTION_TO_EDGE = {
    0: "WC",  # right → from West
    1: "NC",  # down  → from North
    2: "EC",  # left  → from East
    3: "SC",  # up    → from South
}

DIRECTION_NAMES = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}


class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0


# Global state
signals = []
currentGreen = 0
nextGreen = 1
currentYellow = 0
timeElapsed = 0

# RL Agent
rl_agent = None
rl_model_type = "SAC"


def getVehicleCounts():
    """
    Get vehicle counts for all directions.
    Returns observation vector for RL agent.
    SUMO counts ALL vehicles on each edge (not screen-limited like Pygame).
    """
    obs = np.zeros(16, dtype=np.float32)
    
    for direction in range(4):
        edge_id = DIRECTION_TO_EDGE[direction]
        try:
            # Count vehicles on incoming edge
            vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
            
            # Simple: put all vehicles in the first lane slot for that direction
            # More sophisticated: distribute by lane
            total_vehicles = len(vehicle_ids)
            
            # Put in observation (3 lanes per direction)
            base_idx = direction * 3
            obs[base_idx] = min(total_vehicles, 20)  # Cap at 20 per lane for normalization
            obs[base_idx + 1] = 0
            obs[base_idx + 2] = 0
            
        except Exception as e:
            pass
    
    # Add signal state info (last 4 elements)
    for i in range(4):
        obs[12 + i] = 1.0 if i == currentGreen else 0.0
    
    return obs


def setTimeRL():
    """
    Use RL agent to determine green time for next signal.
    Replaces GST formula with learned policy.
    
    Note: Since SAC was trained on Pygame (different dynamics),
    we use GST formula as a smart fallback when RL gives suboptimal results.
    """
    global rl_agent, nextGreen, rl_model_type
    
    obs = getVehicleCounts()
    
    # Get vehicle count for next direction
    base_idx = nextGreen * 3
    vehicle_count = int(obs[base_idx] + obs[base_idx + 1] + obs[base_idx + 2])
    
    # Calculate what GST formula would give (for comparison)
    gst_time = 5 + int(vehicle_count * 1.5)
    gst_time = max(defaultMinimum, min(defaultMaximum, gst_time))
    
    if rl_agent is not None:
        try:
            if rl_model_type == "SAC":
                action = rl_agent.select_action(obs, deterministic=True)
                rl_green_time = int(action[nextGreen] * (defaultMaximum - defaultMinimum) + defaultMinimum)
            else:  # DQN
                action, _ = rl_agent.select_action(obs, deterministic=True)
                rl_green_time = int(action[nextGreen] * (defaultMaximum - defaultMinimum) + defaultMinimum)
            
            # Smart fallback: If RL gives minimum time but there's significant traffic,
            # use the higher of RL and GST to avoid starving lanes
            if vehicle_count >= 5 and rl_green_time == defaultMinimum:
                green_time = gst_time
                method = f"GST-fallback (RL gave {rl_green_time}s)"
            elif rl_green_time < gst_time * 0.5 and vehicle_count >= 10:
                # RL is giving less than half of what GST suggests for heavy traffic
                green_time = max(rl_green_time, int(gst_time * 0.7))
                method = f"Hybrid (RL: {rl_green_time}s, GST: {gst_time}s)"
            else:
                green_time = rl_green_time
                method = "RL"
                
        except Exception as e:
            green_time = gst_time
            method = "GST (RL error)"
    else:
        green_time = gst_time
        method = "GST (no RL)"
    
    # Clamp to bounds
    green_time = max(defaultMinimum, min(defaultMaximum, green_time))
    
    signals[(currentGreen + 1) % noOfSignals].green = green_time
    
    print(f"[{method}] Signal {nextGreen+1}: {vehicle_count} cars → {green_time}s")


def printStatus():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                print(f" GREEN TS {i+1} -> r: {signals[i].red}  y: {signals[i].yellow}  g: {signals[i].green}")
            else:
                print(f"YELLOW TS {i+1} -> r: {signals[i].red}  y: {signals[i].yellow}  g: {signals[i].green}")
        else:
            print(f"   RED TS {i+1} -> r: {signals[i].red}  y: {signals[i].yellow}  g: {signals[i].green}")
    print()


def updateValues():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
                signals[i].totalGreenTime += 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


def initialize():
    global signals
    signals = []
    
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)


def getSumoPhase():
    """
    Map our 4-phase signal state to SUMO traffic light phases.
    SUMO phases (from intersection.net.xml):
      Phase 0: West green (WC)  - RIGHT direction
      Phase 1: West yellow
      Phase 2: North green (NC) - DOWN direction
      Phase 3: North yellow
      Phase 4: East green (EC)  - LEFT direction
      Phase 5: East yellow
      Phase 6: South green (SC) - UP direction
      Phase 7: South yellow
    
    Our currentGreen: 0=right (WC), 1=down (NC), 2=left (EC), 3=up (SC)
    """
    # Map our signal index to SUMO phase pairs
    phase_map = {
        0: (0, 1),  # right (WC): green=0, yellow=1
        1: (2, 3),  # down (NC): green=2, yellow=3
        2: (4, 5),  # left (EC): green=4, yellow=5
        3: (6, 7),  # up (SC): green=6, yellow=7
    }
    
    green_phase, yellow_phase = phase_map[currentGreen]
    return yellow_phase if currentYellow else green_phase


def loadRLAgent(model_type="SAC"):
    """Load the trained RL agent."""
    global rl_agent, rl_model_type
    
    rl_model_type = model_type
    
    if model_type == "SAC":
        rl_agent = SimpleSACAgent(obs_dim=16, action_dim=4, hidden_dim=128, device="auto")
        model_paths = ["checkpoints/sac_extended_best.pt", "checkpoints/optimized_sac_best.pt"]
    else:
        rl_agent = DoubleDQNAgent(obs_dim=16, hidden_dim=128, device="auto")
        model_paths = ["checkpoints/dqn_model.pt", "checkpoints/dqn_best.pt"]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                rl_agent.load(path)
                print(f"Loaded {model_type} model from: {path}")
                return True
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    
    print(f"WARNING: No {model_type} model found, using fallback GST formula")
    rl_agent = None
    return False


def runSimulation(useGui=True, model_type="SAC"):
    global currentGreen, currentYellow, nextGreen, timeElapsed
    
    # Load RL agent
    loadRLAgent(model_type)
    
    # Start SUMO
    sumoBinary = "sumo-gui" if useGui else "sumo"
    sumoCfg = os.path.join(os.path.dirname(__file__), "sumo", "intersection.sumocfg")
    
    if not os.path.exists(sumoCfg):
        print(f"ERROR: SUMO config not found: {sumoCfg}")
        return
    
    traci.start([sumoBinary, "-c", sumoCfg, "--start", "--quit-on-end"])
    
    tlIds = traci.trafficlight.getIDList()
    tlId = tlIds[0] if tlIds else "C"
    
    initialize()
    
    print("=" * 60)
    print(f"SUMO Simulation with RL Control ({rl_model_type})")
    print("=" * 60)
    print("Note: SUMO detects ALL vehicles on edge (no screen bounds)")
    print("=" * 60)
    
    total_crossed = 0
    
    while timeElapsed < simTime:
        traci.simulationStep()
        time.sleep(0.2)
        
        printStatus()
        updateValues()
        
        # RL decision instead of GST
        if signals[(currentGreen + 1) % noOfSignals].red == 5:  # detectionTime
            setTimeRL()
        
        if currentYellow == 0 and signals[currentGreen].green <= 0:
            currentYellow = 1
        elif currentYellow == 1 and signals[currentGreen].yellow <= 0:
            currentYellow = 0
            signals[currentGreen].green = defaultGreen
            signals[currentGreen].yellow = defaultYellow
            signals[currentGreen].red = defaultRed
            currentGreen = nextGreen
            nextGreen = (currentGreen + 1) % noOfSignals
            signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green
        
        try:
            traci.trafficlight.setPhase(tlId, getSumoPhase())
        except:
            pass
        
        # Count departed vehicles (crossed intersection)
        try:
            total_crossed = traci.simulation.getDepartedNumber()
        except:
            pass
        
        timeElapsed += 1
    
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"RL Model: {rl_model_type}")
    print(f"Total time: {timeElapsed}s")
    print(f"Approximate vehicles processed: {total_crossed}")
    print("\nSignal Statistics:")
    for i in range(noOfSignals):
        print(f"  Signal {i+1} ({DIRECTION_NAMES[i]}): Total green time = {signals[i].totalGreenTime}s")
    
    traci.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO Traffic Simulation with RL Control")
    parser.add_argument("--nogui", action="store_true", help="Run without GUI")
    parser.add_argument("--dqn", action="store_true", help="Use DQN instead of SAC")
    parser.add_argument("--sac", action="store_true", help="Use SAC (default)")
    args = parser.parse_args()
    
    model_type = "DQN" if args.dqn else "SAC"
    runSimulation(useGui=not args.nogui, model_type=model_type)
