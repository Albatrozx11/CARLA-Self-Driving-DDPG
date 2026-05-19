# EchoDrive – CARLA Self-Driving Car using DDPG

## Overview

EchoDrive is a Deep Reinforcement Learning based autonomous driving system built using the CARLA Simulator and the Deep Deterministic Policy Gradient (DDPG) algorithm.

The project focuses on training a self-driving agent capable of:

- Lane following
- Obstacle avoidance
- Safe navigation
- Continuous steering and acceleration control
- Generalization to unseen environments

Unlike traditional DQN-based approaches that work with discrete actions, EchoDrive uses DDPG to operate in a continuous action space, making it more suitable for real-world autonomous driving control.

The system combines multiple sensors including:

- RGB Camera
- LiDAR
- IMU
- GNSS / Navigation features

The trained agent learns directly through interaction with the CARLA environment.

---

## Features

- End-to-end autonomous driving in CARLA
- Deep Reinforcement Learning using DDPG
- Multi-modal sensor fusion
- Continuous steering and acceleration control
- Dynamic traffic and pedestrian simulation
- Randomized weather conditions
- A* based route planning
- Replay buffer based off-policy training
- Lane invasion and collision monitoring
- Zero-shot testing on unseen CARLA towns

---

# System Architecture

The project follows an Actor–Critic Reinforcement Learning architecture.

## Main Components

### 1. Simulation Environment

CARLA provides:

- Urban roads
- Dynamic traffic
- Pedestrians
- Weather conditions
- Vehicle physics

### 2. Perception Pipeline

Sensor data is collected and processed from:

- RGB Camera
- LiDAR
- IMU
- Navigation vectors

### 3. State Representation

The final state consists of:

- Grayscale image tensor `(80 × 80 × 1)`
- LiDAR polar obstacle vector `(32)`
- Navigation + IMU vector `(29)`

### 4. DDPG Agent

The DDPG agent contains:

#### Actor Network

Predicts:

- Steering
- Acceleration

#### Critic Network

Evaluates:

- State-action quality (Q-value)

### 5. Reward Function

The reward function encourages:

- Lane keeping
- Progress toward destination
- Smooth steering
- Collision avoidance

---

# Tech Stack

| Component | Technology |
|---|---|
| Simulator | CARLA |
| Game Engine | Unreal Engine |
| RL Algorithm | DDPG |
| Language | Python |
| Deep Learning | TensorFlow |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |

---

# Hardware Requirements

## Minimum

- Quad-core CPU
- 8 GB RAM
- GPU support recommended

## Recommended

- NVIDIA GPU with 8 GB VRAM
- 16 GB RAM
- CUDA support

---

# Software Requirements

- Python 3.8+
- TensorFlow 2+
- CARLA 0.9.x
- CUDA Toolkit (recommended)
- NVIDIA Drivers

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Albatrozx11/CARLA-Self-Driving-DDPG.git
cd CARLA-Self-Driving-DDPG
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Installing CARLA

## Download CARLA

Download CARLA 0.9.x from:

- https://carla.org/
- https://github.com/carla-simulator/carla/releases

---

## Extract CARLA

Example:

```bash
C:\CARLA_0.9.15
```

or

```bash
~/CARLA_0.9.15
```

---

## Launch CARLA

### Windows

```bash
CarlaUE4.exe
```

### Linux

```bash
./CarlaUE4.sh
```

---

# Running the Project

## Step 1 — Start CARLA Simulator

Launch CARLA before running the training script.

Example:

```bash
./CarlaUE4.sh
```

---

## Step 2 — Run Training

```bash
python train.py
```

The training pipeline:

- Initializes the CARLA environment
- Spawns traffic and pedestrians
- Attaches sensors
- Generates routes
- Collects experiences
- Updates Actor and Critic networks

---

## Step 3 — Evaluate Trained Model

```bash
python evaluate.py
```

Evaluation is performed in:

- Unseen towns
- Random weather
- Random traffic conditions

---

# Project Structure

```text
CARLA-Self-Driving-DDPG/
│
├── train.py
├── evaluate.py
├── env/
├── models/
├── sensors/
├── utils/
├── logs/
├── checkpoints/
├── requirements.txt
└── README.md
```

---

# Sensor Pipeline

## RGB Camera

- Resolution: `128 × 128`
- Converted to grayscale
- Resized to `80 × 80`
- Normalized to `[0,1]`

## LiDAR

- Front 180° view
- Encoded into 32 angular bins
- Stores nearest obstacle distances

## IMU

Captures:

- Acceleration
- Gyroscope data

## Navigation Features

Includes:

- Distance to goal
- Speed
- Cross-track error
- Future waypoints

---

# Reinforcement Learning Pipeline

## Replay Buffer

Stores:

- State
- Action
- Reward
- Next State

## Training Loop

1. Observe environment
2. Predict action using Actor
3. Execute action in CARLA
4. Receive reward
5. Store transition
6. Update Actor and Critic networks

---

# Neural Network Architecture

## Actor Network

### Inputs

- Camera tensor
- LiDAR vector
- Navigation/Physics vector

### Outputs

- Steering
- Acceleration

### Layers

- CNN layers for visual features
- Dense layers for sensor fusion
- Fully connected layers for control prediction

---

## Critic Network

### Inputs

- State
- Action

### Output

- Q-value

### Purpose

Evaluates how good an action is in a given state.

---

# Training Strategy

The model is trained in a randomized CARLA environment with:

- Dynamic traffic
- Pedestrians
- Multiple weather conditions
- Random spawn locations

To improve throughput:

- Synchronous simulation is used
- Rendering can be disabled during training

The reward function combines:

- Goal progress
- Lane discipline
- Collision penalties
- Steering smoothness

---

# Results

## Training Results

The DDPG agent successfully learned:

- Lane following
- Smooth steering control
- Collision avoidance
- Navigation toward target waypoints

The model showed improved stability over training episodes through:

- Reduced collision frequency
- Reduced lane invasions
- Smoother control outputs
- Better trajectory adherence

---

## Robustness Testing

The trained agent was evaluated in:

- Completely unseen CARLA towns
- Randomized weather conditions
- Dynamic traffic environments

The testing was performed without any additional fine-tuning.

This demonstrated:

- Generalization capability
- Adaptive driving behavior
- Reduced overfitting to training routes

---

## Key Evaluation Metrics

- Episode reward trends
- Collision count
- Lane invasion frequency
- Driving smoothness
- Route completion consistency

---

# State Representation

The final multi-modal observation space:

```text
State =
{
    Grayscale Image Tensor (80 × 80 × 1)
    LiDAR Polar Vector (32)
    Navigation + IMU Vector (29)
}
```

This allows the model to jointly reason about:

- Visual scene understanding
- Obstacle proximity
- Vehicle dynamics
- Route planning

---

# Algorithms Used

## Route Planning

- A* Global Route Planner

## Reinforcement Learning

- Deep Deterministic Policy Gradient (DDPG)

## Sensor Processing

- Polar LiDAR encoding
- Image normalization
- IMU feature encoding

---

# Testing Strategy

The trained model was tested using:

- Zero-shot deployment
- Unseen maps
- No additional training

Performance was monitored using:

- Collision events
- Lane invasions
- Reward consistency

This ensured the agent learned generalized driving policies instead of memorizing routes.

---

# Research Contributions

This project demonstrates:

- End-to-end autonomous driving using DDPG
- Multi-modal sensor fusion in CARLA
- Continuous control for autonomous navigation
- Generalization testing in unseen environments
- Robust RL training using randomized simulation conditions

---

# Future Improvements

Planned future work includes:

- TD3 implementation
- TCAMD architecture integration
- Multi-agent traffic learning
- Attention mechanisms
- Improved reward shaping
- Sim-to-real transfer learning
- Transformer-based perception models

---

# Literature Inspiration

The project was inspired by multiple research works in:

- DDPG autonomous driving
- CARLA-based RL systems
- Multi-agent reinforcement learning
- Traffic randomization
- Sensor fusion
- TD3 and TCAMD architectures

Key ideas adopted include:

- Continuous action control
- Randomized environments
- Multi-modal perception
- Robust reward engineering

---

# References

1. CARLA Simulator  
   https://carla.org/

2. Deep Deterministic Policy Gradient (DDPG)  
   Lillicrap et al.

3. TensorFlow Documentation  
   https://www.tensorflow.org/

4. Reinforcement Learning for Autonomous Driving Research Papers

---

# Citation

```bibtex
@project{echodrive2026,
  title={EchoDrive: CARLA Self-Driving using DDPG},
  author={Adithyan A and Ann Mariyam Prakash and Karthik Manoj and Sachin Umendran},
  year={2026},
  institution={Model Engineering College}
}
```

---

# Team

- Adithyan A
- Ann Mariyam Prakash
- Karthik Manoj
- Sachin Umendran

Guide:

- Ms. Aysha Fymin Majeed

Department of Computer Science Engineering  
Model Engineering College, Kochi

---

# Acknowledgements

- CARLA Simulator Team
- TensorFlow
- Unreal Engine
- OpenDRIVE
- Research papers referenced during implementation
