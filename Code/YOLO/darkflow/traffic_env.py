"""
Gymnasium Environment Wrapper for Traffic Signal Control Simulation.

This module wraps the Pygame-based traffic simulation as a Gymnasium environment,
enabling reinforcement learning agents (like SAC+GNN) to interact with it.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math
import time
import os
from typing import Optional, Tuple, Dict, Any


class TrafficEnv(gym.Env):
    """
    Gymnasium environment for adaptive traffic signal control.
    
    State Space (16-dim):
        - Vehicle counts per lane (4 directions × 3 lanes = 12)
        - Current green signal (one-hot encoded, 4)
    
    Action Space (4-dim continuous):
        - Normalized green time allocation for each signal [0, 1]
        - Mapped to actual green time: action * (max_green - min_green) + min_green
    
    Reward:
        - Negative of total waiting vehicles (encourages clearing queues)
        - Bonus for vehicles that crossed the intersection
    """
    
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        sim_steps_per_action: int = 5,
        max_episode_steps: int = 300,
        min_green: int = 10,
        max_green: int = 60
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.sim_steps_per_action = sim_steps_per_action
        self.max_episode_steps = max_episode_steps
        self.min_green = min_green
        self.max_green = max_green
        
        # Action space: continuous green time allocation for 4 signals
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Observation space: vehicle counts + current signal state
        # 12 lanes (4 directions × 3 lanes) + 4 signal states
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(16,), dtype=np.float32
        )
        
        # Simulation constants
        self.no_of_signals = 4
        self.direction_numbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
        
        # Default signal times
        self.default_red = 150
        self.default_yellow = 5
        self.default_green = 20
        
        # Vehicle speeds
        self.speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'rickshaw': 2, 'bike': 2.5}
        
        # Coordinates
        self.x_coords = {'right': [0, 0, 0], 'down': [755, 727, 697], 
                         'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
        self.y_coords = {'right': [348, 370, 398], 'down': [0, 0, 0], 
                         'left': [498, 466, 436], 'up': [800, 800, 800]}
        
        self.stop_lines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
        self.default_stop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
        
        # Gap between vehicles
        self.gap = 15
        self.gap2 = 15
        
        # Pygame setup
        self.screen = None
        self.clock = None
        self.background = None
        self.signal_images = {}
        self.font = None
        self.pygame_initialized = False
        
        # Initialize pygame immediately if render mode is human
        if self.render_mode == "human":
            self._init_pygame()
        
        # State variables (reset in reset())
        self._init_state_vars()
        
    def _init_state_vars(self):
        """Initialize/reset state variables."""
        self.signals = []
        self.vehicles = {
            'right': {0: [], 1: [], 2: [], 'crossed': 0},
            'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0},
            'up': {0: [], 1: [], 2: [], 'crossed': 0}
        }
        self.current_green = 0
        self.current_yellow = 0
        self.next_green = 1
        self.time_elapsed = 0
        self.signal_timer = 0
        self.stops = {
            'right': [580, 580, 580],
            'down': [320, 320, 320],
            'left': [810, 810, 810],
            'up': [545, 545, 545]
        }
        # Reset spawn coordinates
        self.x = {'right': [0, 0, 0], 'down': [755, 727, 697], 
                  'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
        self.y = {'right': [348, 370, 398], 'down': [0, 0, 0], 
                  'left': [498, 466, 436], 'up': [800, 800, 800]}
        self.total_crossed = 0
        self.prev_crossed = 0
        
    def _init_signals(self):
        """Initialize traffic signals."""
        self.signals = []
        
        # Signal 0: starts green
        self.signals.append({
            'red': 0,
            'yellow': self.default_yellow,
            'green': self.default_green,
            'minimum': self.min_green,
            'maximum': self.max_green
        })
        
        # Signal 1: red for (yellow + green) of signal 0
        self.signals.append({
            'red': self.default_yellow + self.default_green,
            'yellow': self.default_yellow,
            'green': self.default_green,
            'minimum': self.min_green,
            'maximum': self.max_green
        })
        
        # Signals 2 and 3: default red
        for _ in range(2):
            self.signals.append({
                'red': self.default_red,
                'yellow': self.default_yellow,
                'green': self.default_green,
                'minimum': self.min_green,
                'maximum': self.max_green
            })
    
    def _spawn_vehicle(self):
        """Spawn a random vehicle."""
        vehicle_types = {0: 'car', 1: 'bus', 2: 'truck', 3: 'rickshaw', 4: 'bike'}
        
        vehicle_type_id = random.randint(0, 4)
        vehicle_class = vehicle_types[vehicle_type_id]
        
        # Bikes go in lane 0, others in lanes 1-2
        if vehicle_type_id == 4:
            lane = 0
        else:
            lane = random.randint(1, 2)
        
        # Determine if vehicle will turn
        will_turn = 0
        if lane == 2:
            will_turn = 1 if random.randint(0, 4) <= 2 else 0
        
        # Direction based on weighted distribution
        temp = random.randint(0, 999)
        if temp < 400:
            direction_number = 0
        elif temp < 800:
            direction_number = 1
        elif temp < 900:
            direction_number = 2
        else:
            direction_number = 3
            
        direction = self.direction_numbers[direction_number]
        
        # Create vehicle dict
        vehicle = {
            'lane': lane,
            'vehicleClass': vehicle_class,
            'direction_number': direction_number,
            'direction': direction,
            'speed': self.speeds[vehicle_class],
            'x': self.x[direction][lane],
            'y': self.y[direction][lane],
            'crossed': 0,
            'willTurn': will_turn,
            'turned': 0,
            'rotateAngle': 0,
            'index': len(self.vehicles[direction][lane]),
            'stop': self.stops[direction][lane],
            'width': 50 if vehicle_class in ['bus', 'truck'] else 30,
            'height': 25 if vehicle_class in ['bus', 'truck'] else 15
        }
        
        # Adjust stop position based on vehicles ahead
        if len(self.vehicles[direction][lane]) > 0:
            prev_vehicle = self.vehicles[direction][lane][-1]
            if prev_vehicle['crossed'] == 0:
                if direction == 'right':
                    vehicle['stop'] = prev_vehicle['stop'] - prev_vehicle['width'] - self.gap
                elif direction == 'left':
                    vehicle['stop'] = prev_vehicle['stop'] + prev_vehicle['width'] + self.gap
                elif direction == 'down':
                    vehicle['stop'] = prev_vehicle['stop'] - prev_vehicle['height'] - self.gap
                elif direction == 'up':
                    vehicle['stop'] = prev_vehicle['stop'] + prev_vehicle['height'] + self.gap
        
        # Update spawn coordinates for next vehicle
        if direction == 'right':
            self.x[direction][lane] -= (vehicle['width'] + self.gap)
            self.stops[direction][lane] -= (vehicle['width'] + self.gap)
        elif direction == 'left':
            self.x[direction][lane] += (vehicle['width'] + self.gap)
            self.stops[direction][lane] += (vehicle['width'] + self.gap)
        elif direction == 'down':
            self.y[direction][lane] -= (vehicle['height'] + self.gap)
            self.stops[direction][lane] -= (vehicle['height'] + self.gap)
        elif direction == 'up':
            self.y[direction][lane] += (vehicle['height'] + self.gap)
            self.stops[direction][lane] += (vehicle['height'] + self.gap)
        
        self.vehicles[direction][lane].append(vehicle)
    
    def _move_vehicle(self, vehicle):
        """Move a single vehicle."""
        direction = vehicle['direction']
        lane = vehicle['lane']
        idx = vehicle['index']
        
        # Check if vehicle has crossed stop line
        if vehicle['crossed'] == 0:
            if direction == 'right' and vehicle['x'] + vehicle['width'] > self.stop_lines[direction]:
                vehicle['crossed'] = 1
                self.vehicles[direction]['crossed'] += 1
                self.total_crossed += 1
            elif direction == 'left' and vehicle['x'] < self.stop_lines[direction]:
                vehicle['crossed'] = 1
                self.vehicles[direction]['crossed'] += 1
                self.total_crossed += 1
            elif direction == 'down' and vehicle['y'] + vehicle['height'] > self.stop_lines[direction]:
                vehicle['crossed'] = 1
                self.vehicles[direction]['crossed'] += 1
                self.total_crossed += 1
            elif direction == 'up' and vehicle['y'] < self.stop_lines[direction]:
                vehicle['crossed'] = 1
                self.vehicles[direction]['crossed'] += 1
                self.total_crossed += 1
        
        # Check if can move
        can_move = False
        
        # If already crossed, can move
        if vehicle['crossed'] == 1:
            can_move = True
        # If signal is green
        elif self.current_green == vehicle['direction_number'] and self.current_yellow == 0:
            can_move = True
        # If not reached stop position
        elif direction == 'right' and vehicle['x'] + vehicle['width'] <= vehicle['stop']:
            can_move = True
        elif direction == 'left' and vehicle['x'] >= vehicle['stop']:
            can_move = True
        elif direction == 'down' and vehicle['y'] + vehicle['height'] <= vehicle['stop']:
            can_move = True
        elif direction == 'up' and vehicle['y'] >= vehicle['stop']:
            can_move = True
        
        # Check gap with vehicle ahead
        if can_move and idx > 0:
            prev_vehicle = self.vehicles[direction][lane][idx - 1]
            if direction == 'right':
                if vehicle['x'] + vehicle['width'] >= prev_vehicle['x'] - self.gap2:
                    can_move = False
            elif direction == 'left':
                if vehicle['x'] <= prev_vehicle['x'] + prev_vehicle['width'] + self.gap2:
                    can_move = False
            elif direction == 'down':
                if vehicle['y'] + vehicle['height'] >= prev_vehicle['y'] - self.gap2:
                    can_move = False
            elif direction == 'up':
                if vehicle['y'] <= prev_vehicle['y'] + prev_vehicle['height'] + self.gap2:
                    can_move = False
        
        # Move vehicle
        if can_move:
            if direction == 'right':
                vehicle['x'] += vehicle['speed']
            elif direction == 'left':
                vehicle['x'] -= vehicle['speed']
            elif direction == 'down':
                vehicle['y'] += vehicle['speed']
            elif direction == 'up':
                vehicle['y'] -= vehicle['speed']
    
    def _update_signals(self):
        """Update signal timers."""
        for i in range(self.no_of_signals):
            if i == self.current_green:
                if self.current_yellow == 0:
                    self.signals[i]['green'] -= 1
                    if self.signals[i]['green'] <= 0:
                        self.current_yellow = 1
                        self.signals[i]['green'] = 0
                else:
                    self.signals[i]['yellow'] -= 1
                    if self.signals[i]['yellow'] <= 0:
                        self.current_yellow = 0
                        # Reset current signal
                        self.signals[i]['green'] = self.default_green
                        self.signals[i]['yellow'] = self.default_yellow
                        self.signals[i]['red'] = self.default_red
                        # Move to next signal
                        self.current_green = self.next_green
                        self.next_green = (self.current_green + 1) % self.no_of_signals
                        # Set red time for next-next signal
                        self.signals[self.next_green]['red'] = (
                            self.signals[self.current_green]['yellow'] + 
                            self.signals[self.current_green]['green']
                        )
            else:
                self.signals[i]['red'] -= 1
                if self.signals[i]['red'] < 0:
                    self.signals[i]['red'] = 0
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        obs = []
        
        # Vehicle counts per lane (12 values)
        for direction in ['right', 'down', 'left', 'up']:
            for lane in range(3):
                # Count vehicles that haven't crossed
                count = sum(1 for v in self.vehicles[direction][lane] if v['crossed'] == 0)
                obs.append(min(count, 20))  # Cap at 20 for normalization
        
        # Current signal state (one-hot, 4 values)
        for i in range(self.no_of_signals):
            if i == self.current_green and self.current_yellow == 0:
                obs.append(1.0)
            else:
                obs.append(0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_waiting_vehicles(self) -> int:
        """Count total vehicles waiting (not crossed)."""
        total = 0
        for direction in ['right', 'down', 'left', 'up']:
            for lane in range(3):
                total += sum(1 for v in self.vehicles[direction][lane] if v['crossed'] == 0)
        return total
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self._init_state_vars()
        self._init_signals()
        
        if self.render_mode == "human":
            self._init_pygame()
        
        # Spawn some initial vehicles
        for _ in range(random.randint(5, 15)):
            self._spawn_vehicle()
        
        observation = self._get_observation()
        info = {"time_elapsed": 0, "total_crossed": 0}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Apply action: set green time for the NEXT signal only
        # Action[0] controls how long the next green phase will be
        next_green_time = int(action[self.next_green] * (self.max_green - self.min_green) + self.min_green)
        self.signals[self.next_green]['green'] = next_green_time
        
        # Run simulation for multiple steps
        prev_crossed = self.total_crossed
        reward = 0.0
        
        for _ in range(self.sim_steps_per_action):
            # Spawn vehicles randomly
            if random.random() < 0.3:
                self._spawn_vehicle()
            
            # Move all vehicles
            for direction in ['right', 'down', 'left', 'up']:
                for lane in range(3):
                    for vehicle in self.vehicles[direction][lane]:
                        self._move_vehicle(vehicle)
            
            # Update signals
            self._update_signals()
            
            self.time_elapsed += 1
            
            # Check for empty green light (penalize inefficiency)
            if self.current_yellow == 0:  # Green light active
                direction = self.direction_numbers[self.current_green]
                # Check if there are ANY uncrossed vehicles in this direction
                active_vehicles = 0
                for lane in range(3):
                    active_vehicles += sum(1 for v in self.vehicles[direction][lane] if v['crossed'] == 0)
                
                # If green light is on but NO cars are waiting/crossing, penalize
                if active_vehicles == 0:
                    reward -= 1.0
            
            # Render if needed
            if self.render_mode == "human":
                self._render_frame()
        
        # Calculate reward components
        crossed_this_step = self.total_crossed - self.prev_crossed
        self.prev_crossed = self.total_crossed
        
        # Calculate waiting (needed for info)
        current_waiting = self._get_waiting_vehicles()
        
        # Reward: +1 per vehicle crossed (User specification)
        reward += (crossed_this_step * 1.0)
        
        # Check termination
        terminated = False
        truncated = self.time_elapsed >= self.max_episode_steps
        
        observation = self._get_observation()
        info = {
            "time_elapsed": self.time_elapsed,
            "total_crossed": self.total_crossed,
            "waiting_vehicles": current_waiting
        }
        
        return observation, reward, terminated, truncated, info
    
    def _init_pygame(self):
        """Initialize Pygame for rendering."""
        if not self.pygame_initialized:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Traffic Simulation - RL Environment")
            self.screen = pygame.display.set_mode((1400, 800))
            self.clock = pygame.time.Clock()
            
            # Load images
            image_dir = os.path.join(os.path.dirname(__file__), 'images')
            self.background = pygame.image.load(os.path.join(image_dir, 'mod_int.png'))
            self.signal_images = {
                'red': pygame.image.load(os.path.join(image_dir, 'signals', 'red.png')),
                'yellow': pygame.image.load(os.path.join(image_dir, 'signals', 'yellow.png')),
                'green': pygame.image.load(os.path.join(image_dir, 'signals', 'green.png'))
            }
            self.font = pygame.font.Font(None, 30)
            self.pygame_initialized = True
    
    def _render_frame(self):
        """Render one frame."""
        if not self.pygame_initialized or self.screen is None:
            self._init_pygame()
            if self.screen is None:
                return
        
        # Handle events
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
        except pygame.error:
            return
        
        # Draw background
        self.screen.blit(self.background, (0, 0))
        
        # Draw signals
        signal_coords = [(530, 230), (810, 230), (810, 570), (530, 570)]
        for i in range(self.no_of_signals):
            if i == self.current_green:
                if self.current_yellow == 1:
                    self.screen.blit(self.signal_images['yellow'], signal_coords[i])
                else:
                    self.screen.blit(self.signal_images['green'], signal_coords[i])
            else:
                self.screen.blit(self.signal_images['red'], signal_coords[i])
        
        # Draw vehicles (as colored rectangles)
        colors = {
            'car': (0, 100, 255),
            'bus': (255, 165, 0),
            'truck': (139, 69, 19),
            'rickshaw': (255, 255, 0),
            'bike': (0, 255, 0)
        }
        
        for direction in ['right', 'down', 'left', 'up']:
            for lane in range(3):
                for vehicle in self.vehicles[direction][lane]:
                    color = colors.get(vehicle['vehicleClass'], (255, 255, 255))
                    rect = pygame.Rect(
                        int(vehicle['x']), int(vehicle['y']),
                        vehicle['width'], vehicle['height']
                    )
                    pygame.draw.rect(self.screen, color, rect)
        
        # Draw info text
        info_text = self.font.render(
            f"Time: {self.time_elapsed}  Crossed: {self.total_crossed}  Waiting: {self._get_waiting_vehicles()}",
            True, (255, 255, 255), (0, 0, 0)
        )
        self.screen.blit(info_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_frame()
    
    def close(self):
        """Clean up resources."""
        if self.pygame_initialized:
            try:
                pygame.quit()
            except:
                pass
            self.screen = None
            self.pygame_initialized = False


# Quick test
if __name__ == "__main__":
    print("Testing TrafficEnv...")
    
    # Test without rendering
    env = TrafficEnv(render_mode=None)
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    # Take a few random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, waiting={info['waiting_vehicles']}, crossed={info['total_crossed']}")
    
    env.close()
    print("Test completed successfully!")
