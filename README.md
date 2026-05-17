# Predator-Prey Reinforcement Learning

This repository contains implementations of classical and deep Reinforcement Learning algorithms developed for a predator-prey gridworld environment. The project explores both model-based and policy-based learning approaches, beginning from exact Markov Decision Process (MDP) formulations and extending to neural policy optimization using PyTorch.

The environment consists of a predator attempting to capture a randomly moving prey in an \(N \times N\) discrete gridworld. The project investigates how optimal decision-making strategies can be learned through Dynamic Programming methods, transition kernel estimation, and Policy Gradient based Reinforcement Learning techniques.

The repository is divided into two major assignments.

---

# Assignment 1 — Classical Reinforcement Learning

This section focuses on exact environment modeling and Dynamic Programming approaches for solving the predator-prey problem.

The following concepts are implemented:

- Predator-prey simulator
- State-transition kernel generation
- Reward function construction
- Stochastic policy generation
- Policy-induced kernels and rewards
- State-value and Q-value evaluation
- Value Iteration
- Policy Iteration
- Transition kernel estimation from simulations
- Approximate planning using estimated models

The assignment also includes runtime analysis, convergence studies, and comparisons between exact and estimated solutions for different grid sizes.

---

# Assignment 2 — Policy Gradient Reinforcement Learning

This section extends the environment into a policy-based Reinforcement Learning setting using neural network function approximation.

The following concepts are implemented:

- Neural policy representation using PyTorch
- Monte Carlo policy gradient estimation
- REINFORCE-style learning
- Stochastic Gradient Ascent (SGA)
- Adam optimization
- Learning curve analysis
- Optimizer comparison

The assignment demonstrates how policies can be learned directly through interaction with the environment without requiring explicit transition probability models.

---

# Key Reinforcement Learning Concepts Covered

This project demonstrates practical implementation and experimentation with:

- Markov Decision Processes (MDPs)
- Dynamic Programming
- Bellman Equations
- Policy Evaluation
- Value Iteration
- Policy Iteration
- Transition Kernel Estimation
- Approximate Reinforcement Learning
- Policy Gradient Methods
- Neural Policy Approximation
- Monte Carlo Estimation
- Deep Reinforcement Learning

---

# Technologies Used

- Python
- NumPy
- PyTorch
- Matplotlib

---

# Repository Structure

```text
predator-prey-reinforcement-learning/
│
├── assignment1_classical_rl/
│   ├── part1_environment_and_policy/
│   ├── part2_dynamic_programming/
│   └── part3_kernel_estimation/
│
├── assignment2_policy_gradient/
│
├── README.md
