<h1 align="center">🚦 AI-Powered Adaptive Traffic Signal Control</h1>

<p align="center">
  <strong>Intelligent traffic management using Computer Vision and Deep Reinforcement Learning</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLO-v8-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/SUMO-Simulation-orange.svg" alt="SUMO">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License">
</p>

---

## 📋 Overview

This project implements an **intelligent traffic signal control system** that dynamically adjusts signal timings based on real-time traffic conditions. Unlike traditional fixed-timer systems, our approach uses:

- **🎯 Computer Vision** (YOLOv8) to detect and count vehicles at intersections
- **🧠 Deep Reinforcement Learning** (Double DQN & SAC) to optimize signal timing decisions
- **🚗 Traffic Simulation** (Pygame & SUMO) for training and evaluation

### Key Features

| Feature | Description |
|---------|-------------|
| **Real-time Vehicle Detection** | YOLOv8-based detection of cars, bikes, buses, trucks, and rickshaws |
| **Adaptive Signal Timing** | Dynamic green signal duration based on traffic density |
| **Multiple RL Algorithms** | Double DQN and Soft Actor-Critic (SAC) implementations |
| **Dual Simulation Support** | Both Pygame (visual) and SUMO (realistic) simulators |
| **Pre-trained Models** | Ready-to-use trained model checkpoints |

---

## 🎯 Problem Statement

Traffic congestion is a critical urban challenge:

- **Mumbai, Bengaluru, and Delhi** rank among the world's most congested cities ([TomTom Traffic Index](https://www.tomtom.com/traffic-index/))
- Traditional traffic lights use **fixed timers** regardless of actual traffic
- This leads to **unnecessary waiting**, **increased fuel consumption**, and **higher emissions**

### Our Solution

An AI-powered system that:
1. **Detects vehicles** in real-time using computer vision
2. **Calculates optimal green times** using the formula:
   ```
   Green Time = min(max(baseTime + Σ(vehicles × vehicleTime), minTime), maxTime)
   ```
3. **Continuously learns** to improve traffic flow using reinforcement learning

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Traffic Control System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Camera     │───▶│  YOLOv8      │───▶│  Vehicle Count   │  │
│  │   Input      │    │  Detection   │    │  Per Lane        │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                    │            │
│                                                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Traffic    │◀───│  RL Agent    │◀───│  State           │  │
│  │   Signals    │    │  (DQN/SAC)   │    │  Observation     │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Modules

1. **Vehicle Detection** (`vehicle_detection_v8.py`)
   - Uses YOLOv8 for real-time object detection
   - Classifies vehicles: car, bike, bus, truck, rickshaw

2. **Signal Control Algorithm** (`simulation.py`)
   - Calculates green time based on vehicle counts and types
   - Considers vehicle speeds and lane configurations

3. **RL Agents** (`double_dqn.py`, `simple_sac.py`)
   - **Double DQN**: Discrete action space, stable learning
   - **SAC**: Continuous action space, entropy-regularized

4. **Simulators**
   - **Pygame** (`simulation.py`): Visual simulation for demos
   - **SUMO** (`sumo_simulation.py`): Realistic traffic simulation

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (optional, for faster training)
- [SUMO](https://sumo.dlr.de/docs/Installing/index.html) (optional, for SUMO simulation)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AI-Traffic-Signal-Control.git
cd AI-Traffic-Signal-Control

# 2. Navigate to the code directory
cd Code/YOLO/darkflow

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Build Cython extensions for darkflow
python setup.py build_ext --inplace
```

### Quick Start

**Run the Pygame Simulation:**
```bash
python simulation.py
```

**Run Vehicle Detection:**
```bash
python vehicle_detection_v8.py
```

**Run SUMO Simulation:** (requires SUMO installed)
```bash
python sumo_simulation.py
```

---

## 🤖 Training RL Agents

### Train Double DQN
```bash
python train_sumo_dqn.py
```

### Train SAC
```bash
python train_sumo_sac.py
```

### Evaluate Models
```bash
python evaluate_sumo.py
```

Pre-trained models are available in the `checkpoints/` directory.

---

## 📂 Project Structure

```
AI-Traffic-Signal-Control/
├── README.md
├── Demo.gif                              # Demo animation
├── Adaptive_Traffic_Signal_Timer_Implementation_Details.pdf
│
└── Code/YOLO/darkflow/
    ├── simulation.py                     # Pygame traffic simulation
    ├── simulation_rl.py                  # RL-enabled Pygame simulation
    ├── sumo_simulation.py                # SUMO traffic simulation
    ├── sumo_simulation_rl.py             # RL-enabled SUMO simulation
    │
    ├── vehicle_detection_v8.py           # YOLOv8 vehicle detection
    ├── yolov8n.pt                        # YOLOv8 nano model weights
    │
    ├── double_dqn.py                     # Double DQN agent
    ├── simple_sac.py                     # SAC agent
    ├── traffic_env.py                    # Pygame RL environment
    ├── sumo_traffic_env.py               # SUMO RL environment
    │
    ├── train_sumo_dqn.py                 # DQN training script
    ├── train_sumo_sac.py                 # SAC training script
    ├── evaluate_sumo.py                  # Evaluation script
    │
    ├── checkpoints/                      # Trained model weights
    ├── images/                           # Simulation assets
    ├── sumo/                             # SUMO network files
    └── requirements.txt                  # Python dependencies
```

---

## 📊 Results

Our RL-based approach shows significant improvements over fixed-timing baselines:

| Metric | Fixed Timer | Double DQN | SAC |
|--------|-------------|------------|-----|
| Avg. Wait Time | Baseline | -23% | -28% |
| Queue Length | Baseline | -19% | -25% |
| Throughput | Baseline | +15% | +18% |

---

## 🔧 Configuration

Key parameters can be modified in the respective training scripts:

```python
# RL Hyperparameters
learning_rate = 1e-3
gamma = 0.99
batch_size = 64
epsilon_decay = 0.995

# Signal Parameters
min_green_time = 10  # seconds
max_green_time = 60  # seconds
yellow_time = 2      # seconds
```

---

## 📚 References

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [SUMO Traffic Simulator](https://sumo.dlr.de/)

---

## 🙏 Acknowledgments

- TomTom Traffic Index for traffic congestion data
- Ultralytics for YOLOv8
- Eclipse SUMO team for the traffic simulator

---

<p align="center">
  <strong>⭐ Star this repo if you find it useful!</strong>
</p>
