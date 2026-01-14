"""
SUMO Traffic Environment for RL Training
=========================================
Gymnasium-compatible environment that uses SUMO instead of Pygame.
This allows training agents specifically for SUMO dynamics.
"""

import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    raise RuntimeError("Please set SUMO_HOME environment variable")

import traci

# Configuration
DEFAULT_CONFIG = {
    'sim_steps_per_action': 5,
    'max_episode_steps': 300,
    'min_green': 10,
    'max_green': 60,
}

DIRECTION_TO_EDGE = {
    0: "WC",  # right
    1: "NC",  # down
    2: "EC",  # left
    3: "SC",  # up
}


class SUMOTrafficEnv(gym.Env):
    """
    SUMO-based Traffic Environment for RL training.
    
    Observation Space (16 dims):
        - 12 dims: Vehicle counts per lane (4 directions × 3 lanes)
        - 4 dims: Current signal state (one-hot)
    
    Action Space (4 dims):
        - Continuous [0, 1] for each signal, scaled to [min_green, max_green]
    """
    
    metadata = {'render_modes': ['human', None]}
    
    def __init__(self, render_mode=None, config=None):
        super().__init__()
        
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.render_mode = render_mode
        
        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(16,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        
        # State variables
        self.step_count = 0
        self.episode_count = 0
        self.current_signal = 0
        self.current_phase_timer = 20
        self.total_crossed = 0
        self.total_waiting = 0
        self.traci_started = False
        
        # SUMO config path
        self.sumo_cfg = os.path.join(
            os.path.dirname(__file__), "sumo", "intersection.sumocfg"
        )
        
    def _start_sumo(self):
        """Start SUMO simulation."""
        if self.traci_started:
            try:
                traci.close()
            except:
                pass
        
        sumo_binary = "sumo-gui" if self.render_mode == "human" else "sumo"
        traci.start([
            sumo_binary, "-c", self.sumo_cfg,
            "--start", "--quit-on-end", "--no-warnings",
            "--random"  # Random seed for variety
        ])
        self.traci_started = True
        self.tl_id = traci.trafficlight.getIDList()[0] if traci.trafficlight.getIDList() else "C"
    
    def _get_observation(self):
        """Get current observation from SUMO."""
        obs = np.zeros(16, dtype=np.float32)
        
        for direction in range(4):
            edge_id = DIRECTION_TO_EDGE[direction]
            try:
                vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                base_idx = direction * 3
                obs[base_idx] = min(len(vehicle_ids), 30)  # Cap at 30
            except:
                pass
        
        # Signal state (one-hot)
        obs[12 + self.current_signal] = 1.0
        
        return obs
    
    def _get_waiting_vehicles(self):
        """Count waiting vehicles across all edges."""
        total = 0
        for edge in DIRECTION_TO_EDGE.values():
            try:
                total += traci.edge.getLastStepHaltingNumber(edge)
            except:
                pass
        return total
    
    def _set_phase(self, signal_idx, is_yellow=False):
        """Set SUMO traffic light phase."""
        # 8 phases: 4 green + 4 yellow
        phase = signal_idx * 2 + (1 if is_yellow else 0)
        try:
            traci.trafficlight.setPhase(self.tl_id, phase)
        except:
            pass
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        self._start_sumo()
        
        self.step_count = 0
        self.current_signal = 0
        self.current_phase_timer = 20
        self.total_crossed = 0
        self.total_waiting = 0
        self.episode_count += 1
        
        # Set initial phase
        self._set_phase(0, is_yellow=False)
        
        # Run a few steps to populate vehicles
        for _ in range(10):
            traci.simulationStep()
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute one step in the environment."""
        # Convert action to green time for current signal
        green_time = int(action[self.current_signal] * 
                        (self.config['max_green'] - self.config['min_green']) + 
                        self.config['min_green'])
        
        crossed_this_step = 0
        empty_green_seconds = 0
        
        # Simulate for sim_steps_per_action
        for _ in range(self.config['sim_steps_per_action']):
            traci.simulationStep()
            self.step_count += 1
            
            # Count crossed vehicles
            try:
                arrived = traci.simulation.getArrivedNumber()
                crossed_this_step += arrived
                self.total_crossed += arrived
            except:
                pass
            
            # Check for empty green light
            edge = DIRECTION_TO_EDGE[self.current_signal]
            try:
                vehicles_on_green = len(traci.edge.getLastStepVehicleIDs(edge))
                if vehicles_on_green == 0:
                    empty_green_seconds += 1
            except:
                pass
            
            # Update phase timer
            self.current_phase_timer -= 1
            
            # Handle phase transitions
            if self.current_phase_timer <= 0:
                # Yellow phase
                self._set_phase(self.current_signal, is_yellow=True)
                for _ in range(5):  # 5 second yellow
                    traci.simulationStep()
                    self.step_count += 1
                
                # Switch to next signal
                self.current_signal = (self.current_signal + 1) % 4
                self._set_phase(self.current_signal, is_yellow=False)
                self.current_phase_timer = green_time
        
        # Calculate reward
        reward = 0.0
        reward += crossed_this_step * 1.0  # +1 per vehicle crossed
        reward -= empty_green_seconds * 0.5  # -0.5 per empty green second
        
        # Current waiting for info
        current_waiting = self._get_waiting_vehicles()
        self.total_waiting += current_waiting
        
        # Check termination
        terminated = self.step_count >= self.config['max_episode_steps']
        truncated = False
        
        obs = self._get_observation()
        info = {
            'crossed': crossed_this_step,
            'waiting': current_waiting,
            'total_crossed': self.total_crossed,
            'green_time': green_time,
        }
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up resources."""
        if self.traci_started:
            try:
                traci.close()
            except:
                pass
            self.traci_started = False


if __name__ == "__main__":
    # Test the environment
    print("Testing SUMO Traffic Environment...")
    env = SUMOTrafficEnv(render_mode=None)
    
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, crossed={info['crossed']}, waiting={info['waiting']}")
        if done:
            break
    
    env.close()
    print("Test complete!")
