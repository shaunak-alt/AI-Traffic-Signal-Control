"""
SUMO Traffic Simulation - Pygame Replacement
=============================================
This is a DIRECT replacement of simulation.py using SUMO instead of Pygame.
ALL signal control logic, GST formula, and behavior remain EXACTLY the same.

Pygame → SUMO Mapping:
- vehicles[direction][lane] → traci.edge.getLastStepVehicleIDs()
- vehicle.vehicleClass → traci.vehicle.getTypeID()
- vehicle.crossed == 0 → vehicles on incoming edge
- Pygame rendering → SUMO-GUI handles this
- Vehicle movement → SUMO handles this

Usage:
    python sumo_simulation.py          # With SUMO-GUI
    python sumo_simulation.py --nogui  # Headless mode
"""

import os
import sys
import math
import time
import argparse

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please set SUMO_HOME environment variable")

import traci

# =============================================================================
# CONFIGURATION - EXACT VALUES FROM PYGAME simulation.py (lines 27-58)
# =============================================================================

# Default signal times (lines 27-31)
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

# Simulation parameters (lines 34-36)
noOfSignals = 4
simTime = 300

# Average times for vehicles to pass intersection (lines 43-47)
carTime = 2
bikeTime = 1
rickshawTime = 2.25
busTime = 2.5
truckTime = 2.5

# Lane configuration (line 55)
noOfLanes = 2

# Detection time before green (line 58)
detectionTime = 5

# Direction mapping: Pygame direction number → SUMO incoming edge (line 68)
# directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}
DIRECTION_TO_EDGE = {
    0: "WC",  # right → from West
    1: "NC",  # down  → from North
    2: "EC",  # left  → from East
    3: "SC",  # up    → from South
}

DIRECTION_NAMES = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}


# =============================================================================
# TRAFFIC SIGNAL CLASS - EXACT FROM PYGAME (lines 91-99)
# =============================================================================

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0


# =============================================================================
# GLOBAL STATE - EXACT FROM PYGAME (lines 33-40)
# =============================================================================

signals = []
currentGreen = 0
nextGreen = 1
currentYellow = 0
timeElapsed = 0


# =============================================================================
# VEHICLE COUNTING - REPLACES PYGAME vehicle iteration (lines 293-313)
# =============================================================================

def getVehicleCounts(direction):
    """
    Count vehicles by type on the incoming edge for a direction.
    This replaces the Pygame logic that iterated over vehicles[direction][lane].
    Only counts vehicles that haven't crossed (equivalent to vehicle.crossed == 0).
    """
    edge_id = DIRECTION_TO_EDGE[direction]
    
    noOfCars = 0
    noOfBikes = 0
    noOfBuses = 0
    noOfTrucks = 0
    noOfRickshaws = 0
    
    try:
        # Get all vehicles on incoming edge (equivalent to vehicles not crossed)
        vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
        
        for veh_id in vehicle_ids:
            vtype = traci.vehicle.getTypeID(veh_id)
            
            # Match Pygame logic: lane 0 is bikes only (lines 294-299)
            # Lanes 1-2 are cars, buses, trucks, rickshaws (lines 300-313)
            if vtype == "car":
                noOfCars += 1
            elif vtype == "bike":
                noOfBikes += 1
            elif vtype == "bus":
                noOfBuses += 1
            elif vtype == "truck":
                noOfTrucks += 1
            elif vtype == "rickshaw":
                noOfRickshaws += 1
    except:
        pass
    
    return noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws


# =============================================================================
# GST CALCULATION - EXACT FROM PYGAME (lines 280-323)
# =============================================================================

def setTime():
    """
    Calculate green signal time using EXACT GST formula from Pygame (line 315).
    Called when signals[(currentGreen+1)%noOfSignals].red == detectionTime
    """
    global nextGreen
    
    # Get vehicle counts for next green direction (lines 293-313)
    noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws = getVehicleCounts(nextGreen)
    
    print(f"  Detecting vehicles for {DIRECTION_NAMES[nextGreen]}:")
    print(f"    Cars: {noOfCars}, Bikes: {noOfBikes}, Buses: {noOfBuses}, "
          f"Trucks: {noOfTrucks}, Rickshaws: {noOfRickshaws}")
    
    # EXACT GST FORMULA FROM LINE 315:
    # greenTime = math.ceil(((noOfCars*carTime) + (noOfRickshaws*rickshawTime) + 
    #              (noOfBuses*busTime) + (noOfTrucks*truckTime) + (noOfBikes*bikeTime))/(noOfLanes+1))
    greenTime = math.ceil(
        ((noOfCars * carTime) + 
         (noOfRickshaws * rickshawTime) + 
         (noOfBuses * busTime) + 
         (noOfTrucks * truckTime) + 
         (noOfBikes * bikeTime)) / (noOfLanes + 1)
    )
    
    print(f"    Green Time: {greenTime}")
    
    # Apply bounds (lines 318-321)
    if greenTime < defaultMinimum:
        greenTime = defaultMinimum
    elif greenTime > defaultMaximum:
        greenTime = defaultMaximum
    
    # Set green time for next signal (line 323)
    signals[(currentGreen + 1) % noOfSignals].green = greenTime


# =============================================================================
# PRINT STATUS - EXACT FROM PYGAME (lines 360-369)
# =============================================================================

def printStatus():
    """Print signal status in EXACT same format as Pygame."""
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                print(f" GREEN TS {i+1} -> r: {signals[i].red}  y: {signals[i].yellow}  g: {signals[i].green}")
            else:
                print(f"YELLOW TS {i+1} -> r: {signals[i].red}  y: {signals[i].yellow}  g: {signals[i].green}")
        else:
            print(f"   RED TS {i+1} -> r: {signals[i].red}  y: {signals[i].yellow}  g: {signals[i].green}")
    print()


# =============================================================================
# UPDATE VALUES - EXACT FROM PYGAME (lines 372-381)
# =============================================================================

def updateValues():
    """Decrement signal timers - EXACT from Pygame."""
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
                signals[i].totalGreenTime += 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


# =============================================================================
# INITIALIZE - EXACT FROM PYGAME (lines 268-277)
# =============================================================================

def initialize():
    """Initialize signals - EXACT from Pygame."""
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


# =============================================================================
# SUMO PHASE MAPPING
# =============================================================================

def getSumoPhase():
    """
    Map our 4-phase signal state to SUMO traffic light phases.
    SUMO phases (from intersection.net.xml - 4-phase):
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
    phase_map = {
        0: (0, 1),  # right (WC): green=0, yellow=1
        1: (2, 3),  # down (NC): green=2, yellow=3
        2: (4, 5),  # left (EC): green=4, yellow=5
        3: (6, 7),  # up (SC): green=6, yellow=7
    }
    
    green_phase, yellow_phase = phase_map[currentGreen]
    return yellow_phase if currentYellow else green_phase


# =============================================================================
# MAIN SIMULATION LOOP - STRUCTURE FROM PYGAME repeat() (lines 325-357)
# =============================================================================

def runSimulation(useGui=True):
    """
    Main simulation loop matching Pygame structure.
    SUMO replaces: vehicle movement, rendering, collision handling.
    Python keeps: signal timing logic, GST calculation, state management.
    """
    global currentGreen, currentYellow, nextGreen, timeElapsed
    
    # Start SUMO
    sumoBinary = "sumo-gui" if useGui else "sumo"
    sumoCfg = os.path.join(os.path.dirname(__file__), "sumo", "intersection.sumocfg")
    
    traci.start([sumoBinary, "-c", sumoCfg, "--start", "--quit-on-end"])
    
    # Get traffic light ID
    tlIds = traci.trafficlight.getIDList()
    tlId = tlIds[0] if tlIds else "C"
    
    # Initialize signals
    initialize()
    
    print("=" * 60)
    print("SUMO Simulation (Pygame Logic Preserved)")
    print("=" * 60)
    
    # Main loop - runs for simTime seconds (matching Pygame simulationTime)
    while timeElapsed < simTime:
        # Step SUMO forward
        traci.simulationStep()
        
        # Real-time delay (matching Pygame time.sleep(1) in repeat())
        time.sleep(0.2)  # Faster: 5x speed
        
        # Print current status (matching Pygame)
        printStatus()
        
        # Update signal timers (matching Pygame)
        updateValues()
        
        # Check detection time trigger (line 330 in Pygame)
        if signals[(currentGreen + 1) % noOfSignals].red == detectionTime:
            setTime()
        
        # Handle signal switching (lines 327-357 in Pygame)
        if currentYellow == 0 and signals[currentGreen].green <= 0:
            # Switch to yellow (line 336)
            currentYellow = 1
        elif currentYellow == 1 and signals[currentGreen].yellow <= 0:
            # Yellow finished, switch to next green (lines 347-356)
            currentYellow = 0
            
            # Reset current signal (lines 350-352)
            signals[currentGreen].green = defaultGreen
            signals[currentGreen].yellow = defaultYellow
            signals[currentGreen].red = defaultRed
            
            # Move to next signal (lines 354-356)
            currentGreen = nextGreen
            nextGreen = (currentGreen + 1) % noOfSignals
            signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green
        
        # Update SUMO traffic light to match our state
        try:
            traci.trafficlight.setPhase(tlId, getSumoPhase())
        except:
            pass
        
        timeElapsed += 1
    
    # Print final stats (matching Pygame simulationTime lines 418-425)
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total time passed: {timeElapsed}")
    print("\nSignal Statistics:")
    for i in range(noOfSignals):
        print(f"  Signal {i+1} ({DIRECTION_NAMES[i]}): Total green time = {signals[i].totalGreenTime}s")
    
    traci.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO Traffic Simulation (Pygame Replacement)")
    parser.add_argument("--nogui", action="store_true", help="Run without GUI")
    args = parser.parse_args()
    
    runSimulation(useGui=not args.nogui)
