# Assignment 1 — Classical Reinforcement Learning

This assignment focuses on modeling, analyzing, and solving a predator-prey gridworld environment using classical Reinforcement Learning (RL) and Dynamic Programming techniques.

The project explores the complete RL pipeline starting from environment simulation and transition kernel generation to optimal policy computation using Value Iteration and Policy Iteration algorithms.

---

# Problem Description

A predator-prey game is modeled on an \(N \times N\) gridworld environment.

- The predator acts as the learning agent.
- The prey moves randomly within the grid.
- The objective of the predator is to catch the prey as efficiently as possible.
- A reward of +1 is obtained whenever the predator catches the prey.
- The environment follows a discounted reward setting with:

\[
\gamma = 0.99
\]

The project investigates both exact and estimated transition models of the environment.

---

# Assignment Structure

The assignment is divided into three major parts.

---

# Part 1 — Environment and Policy Modeling

This section focuses on constructing the complete Markov Decision Process (MDP) representation of the predator-prey environment.

## Implementations

- Predator-prey simulator
- Transition kernel generation
- Reward function generation
- Stochastic policy generation
- Policy-induced transition kernels
- Policy-induced reward functions
- State-value evaluation
- Q-value evaluation

## Key Concepts

- Markov Decision Processes (MDPs)
- State transition probabilities
- Bellman expectation equations
- Policy evaluation
- Stochastic policies

## Main Files

| File | Description |
|---|---|
| `simulator.py` | Simulates one environment step |
| `kernel.py` | Generates exact transition kernel |
| `reward.py` | Constructs reward function |
| `sample_policy.py` | Generates stochastic predator policy |
| `induced_kernel.py` | Computes policy-induced transition kernel |
| `induced_reward.py` | Computes policy-induced rewards |
| `state_value_evaluation.py` | Computes state-value function |
| `q_value_evaluation.py` | Computes Q-value function |

---

# Part 2 — Dynamic Programming

This section implements classical Dynamic Programming methods to compute approximately optimal policies.

## Implementations

- Value Iteration
- Policy Iteration
- Greedy policy extraction
- Runtime comparison
- Optimal value function comparison

## Key Concepts

- Bellman optimality equations
- Optimal control
- Dynamic Programming
- Convergence analysis

## Main Files

| File | Description |
|---|---|
| `value_iteration.py` | Implements Value Iteration |
| `policy_iteration.py` | Implements Policy Iteration |
| `induced_policy.py` | Extracts greedy policy |
| `compare_algorithms.py` | Compares algorithms and runtimes |

---

# Part 3 — Kernel Estimation

This section studies model estimation when exact transition probabilities are unavailable.

## Implementations

- Transition kernel estimation using simulations
- Approximate value iteration using estimated kernels
- Error analysis between exact and estimated solutions
- Statistical comparison across multiple random seeds

## Key Concepts

- Model estimation
- Approximate RL
- Sampling-based estimation
- Monte Carlo estimation

## Main Files

| File | Description |
|---|---|
| `estimate_kernel.py` | Estimates transition probabilities |
| `value_iteration_estimated.py` | Runs value iteration on estimated kernels |

---

# Environment Details

## State Space

Each state is represented using:

\[
((x_p, y_p), (x_r, y_r))
\]

where:
- \((x_p, y_p)\) = predator position
- \((x_r, y_r)\) = prey position

---

## Action Space

The predator can:
- move up
- move down
- move left
- move right
- remain stationary

Invalid boundary actions are handled appropriately.

---

# Experimental Analysis

The assignment includes experiments for multiple grid sizes:

\[
N = 5, 10, 15, 20, 25
\]

The following analyses are performed:

- State-value trends
- Runtime analysis
- L1 difference comparisons
- Kernel estimation accuracy
- Mean and standard deviation analysis across random seeds

---

# Technologies Used

- Python
- NumPy
- Matplotlib

---

# Reports

Detailed theoretical explanations, derivations, implementation details, and plots are included in the corresponding report files inside each part directory.

---

# Learning Outcomes

This assignment demonstrates practical implementation of:

- Markov Decision Processes
- Dynamic Programming
- Policy Evaluation
- Value Iteration
- Policy Iteration
- Transition Kernel Estimation
- Approximate Reinforcement Learning

---

# Repository Structure

```text
assignment1_classical_rl/
│
├── README.md
│
├── part1_environment_and_policy/
├── part2_dynamic_programming/
└── part3_kernel_estimation/
