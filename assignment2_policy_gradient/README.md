# Assignment 2 — Policy Gradient Reinforcement Learning

This assignment focuses on implementing policy-based Reinforcement Learning methods using neural network function approximation and stochastic optimization techniques.

The predator-prey gridworld environment developed in Assignment 1 is extended to train a neural policy using Policy Gradient methods in PyTorch.

The project explores how neural networks can directly parameterize policies and learn optimal behaviors through sampled trajectories and gradient-based optimization.

---

# Problem Description

A predator-prey game is modeled on an \(N \times N\) gridworld environment.

- The predator acts as the learning agent.
- The prey moves randomly within the environment.
- The predator receives a reward of +1 whenever it catches the prey.
- Once captured, a new prey is generated randomly and the game continues.

Unlike Assignment 1, where exact transition kernels and Dynamic Programming methods were used, this assignment focuses on learning policies directly from interactions with the environment using Policy Gradient methods.

---

# Objectives

The major objectives of this assignment are:

- Represent the predator policy using a neural network
- Estimate policy gradients from sampled trajectories
- Train the policy using stochastic gradient ascent
- Compare different optimization techniques
- Analyze convergence using learning curves

---

# Neural Policy Representation

The predator policy is represented using a neural network implemented in PyTorch.

The neural network maps environment states to probability distributions over actions.

## Input

The network takes the current environment state as input:

\[
((x_p, y_p), (x_r, y_r))
\]

where:
- \((x_p, y_p)\) = predator position
- \((x_r, y_r)\) = prey position

---

## Output

The network outputs action probabilities for:

- move up
- move down
- move left
- move right
- remain stationary

The output layer uses a probability distribution to sample actions stochastically.

---

# Policy Gradient Learning

The assignment implements the REINFORCE-style Policy Gradient algorithm.

The policy parameters are updated using sampled trajectories generated from the simulator environment.

The objective is to maximize the expected cumulative reward:

:contentReference[oaicite:0]{index=0}

The policy gradient is estimated using Monte Carlo sampling:

:contentReference[oaicite:1]{index=1}

PyTorch automatic differentiation is used to compute gradients of the log-policy efficiently.

---

# Optimization Methods

Two optimization approaches are implemented and compared.

---

# 1. Simple Stochastic Gradient Ascent (SGA)

The neural policy parameters are updated manually using the stochastic gradient ascent update rule:

:contentReference[oaicite:2]{index=2}

## Features

- Manual parameter updates
- Direct implementation of policy gradient learning
- Useful for understanding optimization fundamentals

---

# 2. Adam Optimizer

The assignment also evaluates the performance of the Adam optimizer available in PyTorch.

## Features

- Adaptive learning rates
- Faster convergence
- Improved training stability
- Better performance in noisy optimization settings

The learning curves of Adam and simple SGA are compared experimentally.

---

# Experimental Analysis

The assignment includes:

- Training neural policies over multiple iterations
- Reward tracking during training
- Learning curve visualization
- Comparison of optimization techniques
- Convergence behavior analysis

---

# Files and Components

| File | Description |
|---|---|
| `policy_network.py` | Neural network policy architecture |
| `gradient_estimate.py` | Policy gradient estimation |
| `simple_sga.py` | Manual stochastic gradient ascent updates |
| `train_adam.py` | Training using Adam optimizer |
| `episode.py` | Trajectory generation and simulation |

---

# Environment Details

## Gridworld

- Discrete \(N \times N\) environment
- Random prey movement
- Boundary-aware predator actions

---

## Reward Structure

| Condition | Reward |
|---|---|
| Predator catches prey | +1 |
| Otherwise | 0 |

---

# Technologies Used

- Python
- NumPy
- PyTorch
- Matplotlib

---

# Key Reinforcement Learning Concepts

This assignment demonstrates practical implementation of:

- Policy Gradient Methods
- REINFORCE Algorithm
- Neural Policy Approximation
- Monte Carlo Gradient Estimation
- Stochastic Optimization
- Deep Reinforcement Learning

---

# Learning Outcomes

Through this assignment, the following concepts are explored:

- Direct policy optimization
- Neural network based RL
- Automatic differentiation in PyTorch
- Gradient estimation from sampled trajectories
- Optimization algorithm comparison
- Exploration through stochastic policies

---

# Repository Structure

```text
assignment2_policy_gradient/
│
├── README.md
├── assignment2_report.pdf
│
├── policy_network.py
├── gradient_estimate.py
├── simple_sga.py
├── train_adam.py
└── episode.py
