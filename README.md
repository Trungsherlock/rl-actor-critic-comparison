# RL Actor-Critic Comparison

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Start](#start)
- [Running Experiments](#running-experiments)
- [Results](#results)
- [Algorithms](#algorithms)
- [Environments](#environments)

## Overview

This project implements and compares two fundamental policy gradient algorithms:
- **REINFORCE with Baseline**: Monte Carlo policy gradient algorithm
- **Actor-Critic**: Temporal Difference policy gradient algorithm

Both algorithms are tested on two OpenAI Gymnasium environments:
- **CartPole-v1** (Discrete action space)
- **MountainCarContinuous-v0** (Continuous action space)

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Trungsherlock/rl-actor-critic-comparison.git
cd rl-actor-critic-comparison
```

### Step 2: Create Virtual Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch 2.9.1
- Gymnasium 1.2.2
- NumPy 2.3.5
- Matplotlib 3.10.7
- tqdm 4.67.1

## Start
## Running Experiments

### CartPole Experiments

#### 1. REINFORCE on CartPole
```bash
python experiments/cartpole/re_cartpole.py
```

#### 2. Actor-Critic on CartPole
```bash
python experiments/cartpole/ac_cartpole.py
```

#### 3. Hyperparameter Sweep
```bash
python experiments/cartpole/hyperparameter_sweep_reinforce.py
```
Tests learning rate combinations:
- Policy LR: [1e-4, 3e-4, 1e-3, 3e-3]
- Value LR: [1e-3, 3e-3, 1e-2]
- Saves best config to `experiments/cartpole/results/best_config.json`

### MountainCar Experiments

#### 1. REINFORCE on MountainCar
```bash
python experiments/mountaincar/re_mountaincar.py
```

#### 2. Actor-Critic on MountainCar
```bash
python experiments/mountaincar/ac_mountaincar.py
```

#### 3. Hyperparameter Sweep
```bash
python experiments/mountaincar/hyperparameter_sweep_reinforce.py
```
Tests learning rate combinations with multiprocessing:
- Policy LR: [3e-4, 5e-4, 1e-3]
- Value LR: [1e-3, 3e-3, 5e-3]
- Trains 1000 episodes per config
- Uses 3 seeds per configuration

## Results

### Best Performance - MountainCar REINFORCE

**Configuration:**
```json
{
  "lr_policy": 0.0003,
  "lr_value": 0.005,
  "hidden_dim": 64,
  "gamma": 0.99
}
```

## Algorithms

### REINFORCE with Baseline
**File:** [src/algorithms/reinforce.py](src/algorithms/reinforce.py)

**Features:**
- Monte Carlo policy gradient
- Baseline (value function) to reduce variance
- Supports discrete and continuous actions
- Episode-level updates

**Usage:**
```python
agent = REINFORCE(
    state_dim=4,
    action_dim=2,
    hidden_dim=128,
    lr_policy=3e-4,
    lr_value=1e-3,
    gamma=0.99,
    continuous=False  # True for continuous actions
)
```

### Actor-Critic
**File:** [src/algorithms/actor_critic.py](src/algorithms/actor_critic.py)

**Features:**
- Temporal Difference learning
- Online updates (step-by-step)
- Separate actor and critic networks
- Bootstrapping from value estimates

**Usage:**
```python
# Discrete actions (CartPole)
agent = DiscreteActorCritic(
    state_dim=4,
    action_dim=2,
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=0.99
)

# Continuous actions (MountainCar)
agent = ContinuousActorCritic(
    state_dim=2,
    action_dim=1,
    actor_lr=1e-4,
    critic_lr=5e-4,
    gamma=0.99
)
```

## Environments

### CartPole-v1
**File:** [src/environments/cartpole.py](src/environments/cartpole.py)

**Specifications:**
- **State**: [position, velocity, angle, angular_velocity]
- **Action**: Discrete(2) - push left or right
- **Goal**: Keep pole upright for 500 timesteps
- **Reward**: +1 per timestep

### MountainCarContinuous-v0
**File:** [src/environments/mountaincarcontinuous.py](src/environments/mountaincarcontinuous.py)

**Specifications:**
- **State**: [position, velocity]
  - Position: [-1.2, 0.6]
  - Velocity: [-0.07, 0.07]
- **Action**: Continuous force in [-1.0, 1.0]
- **Goal**: Reach position ≥ 0.45
- **Reward**: +100 for goal, penalty for action²

**Features:**
- Built-in state normalization
- 999 step episode limit

## Save and Load Models

```python
# Save
agent.save('models/my_agent.pth')

# Load
agent.load('models/my_agent.pth')

# Evaluate
avg_reward, std = agent.evaluate(env, num_episodes=100)
```

## Contact

**Repository:** https://github.com/Trungsherlock/rl-actor-critic-comparison
