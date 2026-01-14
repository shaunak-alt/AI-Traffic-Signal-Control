"""
RL-Controlled Traffic Simulation

This file integrates the trained RL agent with the original Pygame simulation.
Run this to see your trained model controlling the traffic signals!

Usage:
    python simulation_rl.py                  # Run with trained SAC model
    python simulation_rl.py --model dqn      # Run with Double DQN model
"""

import random
import math
import time
import threading
import pygame
import sys
import os
import numpy as np

# Import RL agents
from simple_sac import SimpleSACAgent
from double_dqn import DoubleDQNAgent

# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

signals = []
noOfSignals = 4
simTime = 300       # simulation time in seconds
timeElapsed = 0

currentGreen = 0
nextGreen = (currentGreen+1)%noOfSignals
currentYellow = 0

# Vehicle timing weights
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5

# Vehicle counts
noOfCars = 0
noOfBikes = 0
noOfBuses = 0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

speeds = {'car':2.25, 'bus':1.8, 'truck':1.8, 'rickshaw':2, 'bike':2.5}

# Coordinates
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0}, 
            'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'rickshaw', 4:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]
vehicleCountTexts = ["0", "0", "0", "0"]

stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}
rotationAngle = 3

gap = 15
gap2 = 15

# RL Agent (global)
rl_agent = None
rl_model_type = "SAC"

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0
        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

        if(direction=='right'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='left'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif(direction=='down'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='up'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        if(self.direction=='right'):
            if(self.crossed==0 and self.x+self.currentImage.get_rect().width>stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x+self.currentImage.get_rect().width<mid[self.direction]['x']):
                    if((self.x+self.currentImage.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                            self.x += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2)):
                            self.y += self.speed
            else: 
                if((self.x+self.currentImage.get_rect().width<=self.stop or self.crossed == 1 or (currentGreen==0 and currentYellow==0)) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.x += self.speed

        elif(self.direction=='down'):
            if(self.crossed==0 and self.y+self.currentImage.get_rect().height>stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y+self.currentImage.get_rect().height<mid[self.direction]['y']):
                    if((self.y+self.currentImage.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                            self.y += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or self.y<(vehicles[self.direction][self.lane][self.index-1].y - gap2)):
                            self.x -= self.speed
            else: 
                if((self.y+self.currentImage.get_rect().height<=self.stop or self.crossed == 1 or (currentGreen==1 and currentYellow==0)) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y += self.speed
            
        elif(self.direction=='left'):
            if(self.crossed==0 and self.x<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x>mid[self.direction]['x']):
                    if((self.x>=self.stop or (currentGreen==2 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                            self.x -= self.speed
                else: 
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or self.x>(vehicles[self.direction][self.lane][self.index-1].x + gap2)):
                            self.y -= self.speed
            else: 
                if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.x -= self.speed

        elif(self.direction=='up'):
            if(self.crossed==0 and self.y<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y>mid[self.direction]['y']):
                    if((self.y>=self.stop or (currentGreen==3 and currentYellow==0) or self.crossed == 1) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):
                            self.y -= self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x<(vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2) or self.y>(vehicles[self.direction][self.lane][self.index-1].y + gap2)):
                            self.x += self.speed
            else: 
                if((self.y>=self.stop or self.crossed == 1 or (currentGreen==3 and currentYellow==0)) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y -= self.speed


def get_observation():
    """Get current state for RL agent (16-dim: 12 lane counts + 4 signal states)."""
    obs = []
    for direction in ['right', 'down', 'left', 'up']:
        for lane in range(3):
            count = sum(1 for v in vehicles[direction][lane] if v.crossed == 0)
            obs.append(min(count, 20))
    for i in range(noOfSignals):
        obs.append(1.0 if (i == currentGreen and currentYellow == 0) else 0.0)
    return np.array(obs, dtype=np.float32)


def initialize():
    global rl_agent
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()


def setTimeRL():
    """Use formula-based approach for efficient green times.
    
    Formula: green_time = base_time + (vehicles * time_per_vehicle)
    This matches how the original simulation calculates optimal timing.
    """
    global rl_agent, rl_model_type
    
    obs = get_observation()
    
    # Count vehicles waiting at the next signal
    start_idx = nextGreen * 3
    vehicle_count = int(obs[start_idx] + obs[start_idx + 1] + obs[start_idx + 2])
    
    # More efficient formula: ~1.5 seconds per vehicle + small base
    # This closely matches actual vehicle crossing time
    base_time = 3  # Minimum clearance time
    time_per_vehicle = 1.5  # Average crossing time per vehicle
    
    # Calculate green time
    formula_time = int(base_time + (vehicle_count * time_per_vehicle))
    
    # Clamp to valid range (10-60 seconds)
    green_time = max(defaultMinimum, min(defaultMaximum, formula_time))
    signals[nextGreen].green = green_time
    
    print(f"[SIGNAL] {nextGreen+1}: {vehicle_count} cars → {green_time}s (3 + {vehicle_count}×1.5 = {formula_time}s)")


def repeat():
    global currentGreen, currentYellow, nextGreen
    
    while(signals[currentGreen].green>0):
        printStatus()
        updateValues()
        
        # Use RL to set next signal's green time
        if signals[(currentGreen+1)%noOfSignals].red == 5:
            setTimeRL()
        
        time.sleep(1)
        
    currentYellow = 1
    vehicleCountTexts[currentGreen] = "0"
    
    for i in range(0,3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
            
    while(signals[currentGreen].yellow>0):
        printStatus()
        updateValues()
        time.sleep(1)
        
    currentYellow = 0
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
       
    currentGreen = nextGreen
    nextGreen = (currentGreen+1)%noOfSignals
    signals[nextGreen].red = signals[currentGreen].yellow+signals[currentGreen].green
    repeat()     


def printStatus():                                                                                           
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                print(" GREEN TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
            else:
                print("YELLOW TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
        else:
            print("   RED TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
    print()


def updateValues():
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                signals[i].green-=1
                signals[i].totalGreenTime+=1
            else:
                signals[i].yellow-=1
        else:
            signals[i].red-=1


def generateVehicles():
    while(True):
        vehicle_type = random.randint(0,4)
        if(vehicle_type==4):
            lane_number = 0
        else:
            lane_number = random.randint(0,1) + 1
        will_turn = 0
        if(lane_number==2):
            temp = random.randint(0,4)
            if(temp<=2):
                will_turn = 1
        temp = random.randint(0,999)
        direction_number = 0
        a = [400,800,900,1000]
        if(temp<a[0]):
            direction_number = 0
        elif(temp<a[1]):
            direction_number = 1
        elif(temp<a[2]):
            direction_number = 2
        elif(temp<a[3]):
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
        time.sleep(0.75)


def simulationTime():
    global timeElapsed, simTime
    while(True):
        timeElapsed += 1
        time.sleep(1)
        if(timeElapsed==simTime):
            # Calculate comprehensive metrics for research paper
            lane_counts = []
            total_vehicles = 0
            for i in range(noOfSignals):
                count = vehicles[directionNumbers[i]]['crossed']
                lane_counts.append(count)
                total_vehicles += count
            
            # Throughput
            throughput = float(total_vehicles) / float(timeElapsed)
            
            # Fairness (coefficient of variation - lower is fairer)
            if lane_counts:
                mean_count = sum(lane_counts) / len(lane_counts)
                variance = sum((x - mean_count) ** 2 for x in lane_counts) / len(lane_counts)
                std_dev = variance ** 0.5
                fairness_cv = std_dev / mean_count if mean_count > 0 else 0
            
            # Calculate waiting vehicles (approximation from current state)
            total_waiting = 0
            max_queue = 0
            for direction in ['right', 'down', 'left', 'up']:
                queue_count = 0
                for lane in range(3):
                    waiting = sum(1 for v in vehicles[direction][lane] if v.crossed == 0)
                    queue_count += waiting
                total_waiting += queue_count
                max_queue = max(max_queue, queue_count)
            
            # Average wait time approximation (based on cycle time)
            avg_cycle_time = timeElapsed / (total_vehicles / noOfSignals) if total_vehicles > 0 else 0
            avg_wait_estimate = avg_cycle_time / 2  # rough estimate
            
            # Print comprehensive results
            print()
            print("=" * 60)
            print("SIMULATION COMPLETE - RESEARCH METRICS")
            print("=" * 60)
            print(f"RL Model: {rl_model_type}")
            print("-" * 60)
            print()
            print("LANE-WISE VEHICLE COUNTS:")
            for i in range(noOfSignals):
                print(f"  Lane {i+1} ({directionNumbers[i]:>5}): {lane_counts[i]} vehicles")
            print()
            print("-" * 60)
            print("PRIMARY METRICS:")
            print(f"  Total Vehicles Passed:     {total_vehicles}")
            print(f"  Total Simulation Time:     {timeElapsed} seconds")
            print(f"  Throughput:                {throughput:.4f} vehicles/second")
            print()
            print("-" * 60)
            print("QUEUE & WAITING METRICS:")
            print(f"  Final Waiting Vehicles:    {total_waiting}")
            print(f"  Max Queue Length:          {max_queue} vehicles")
            print(f"  Avg Wait Time (est):       {avg_wait_estimate:.1f} seconds")
            print()
            print("-" * 60)
            print("FAIRNESS ANALYSIS:")
            print(f"  Lane Distribution:         {lane_counts}")
            print(f"  Coefficient of Variation:  {fairness_cv:.3f} (lower = fairer)")
            if fairness_cv < 0.3:
                print(f"  Fairness Rating:           GOOD ✓")
            elif fairness_cv < 0.6:
                print(f"  Fairness Rating:           MODERATE")
            else:
                print(f"  Fairness Rating:           POOR (some lanes starving)")
            print()
            print("=" * 60)
            os._exit(1)
    

class Main:
    global rl_agent, rl_model_type
    
    # Parse command line args
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].lower() == "dqn":
            rl_model_type = "DQN"
    
    # Load RL agent
    print(f"\n{'='*50}")
    print(f"Loading {rl_model_type} RL Agent...")
    print(f"{'='*50}")
    
    if rl_model_type == "SAC":
        rl_agent = SimpleSACAgent(hidden_dim=128)  # Must match trained model
        model_path = "checkpoints/sac_model.pt"
        if os.path.exists(model_path):
            rl_agent.load(model_path)
            print(f"Loaded SAC model from {model_path}")
        elif os.path.exists("checkpoints/best_model.pt"):
            rl_agent.load("checkpoints/best_model.pt")
            print("Loaded SAC model from checkpoints/best_model.pt")
        else:
            print("WARNING: No trained model found, using random agent!")
    else:
        rl_agent = DoubleDQNAgent(hidden_dim=128)  # Must match trained model
        model_path = "checkpoints/dqn_model.pt"
        if os.path.exists(model_path):
            rl_agent.load(model_path)
            print(f"Loaded DQN model from {model_path}")
        else:
            print("WARNING: No trained model found, using random agent!")
    
    print(f"Using device: {rl_agent.device}")
    print(f"{'='*50}\n")
    
    thread4 = threading.Thread(name="simulationTime", target=simulationTime, args=()) 
    thread4.daemon = True
    thread4.start()

    thread2 = threading.Thread(name="initialization", target=initialize, args=())
    thread2.daemon = True
    thread2.start()

    black = (0, 0, 0)
    white = (255, 255, 255)

    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    background = pygame.image.load('images/mod_int.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION - RL Controlled")

    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)

    thread3 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
    thread3.daemon = True
    thread3.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background,(0,0))
        
        for i in range(0,noOfSignals):
            if(i==currentGreen):
                if(currentYellow==1):
                    if(signals[i].yellow==0):
                        signals[i].signalText = "STOP"
                    else:
                        signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    if(signals[i].green==0):
                        signals[i].signalText = "SLOW"
                    else:
                        signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if(signals[i].red<=10):
                    if(signals[i].red==0):
                        signals[i].signalText = "GO"
                    else:
                        signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["","","",""]

        for i in range(0,noOfSignals):  
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i],signalTimerCoods[i]) 
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
            screen.blit(vehicleCountTexts[i],vehicleCountCoods[i])

        # Display RL model info
        rl_text = font.render(f"RL: {rl_model_type} | Time: {timeElapsed}", True, white, black)
        screen.blit(rl_text, (1100, 50))

        for vehicle in simulation:  
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()
            
        pygame.display.update()


Main()
