# EE675 ASSIGNMENT 1 - Q1

## Author 
Aayush Dubey 

## Description
This part of assignment implements a predator-prey reinforcement learning environment on an N × N grid. The predator aims to catch the prey while maximizing its expected discounted reward.

## File Structure 

- simulator.py → Environment simulation  
- kernel.py → State generation utilities  
- reward.py → Reward function  
- policy.py → Sample policy  
- induced_kernel.py → Policy-induced kernel  
- induced_reward.py → Policy-induced reward  
- value_eval.py → State value evaluation (sampling-based)  
- q_eval.py → Q-value evaluation  
- main.py → Runs experiments and generates plots

## How to run 
1. Open terminal in the project folder 
2. Install dependencies : pip install numpy matplotlib
3. Run the code : python main.py

## Output 
This program generates two plots 
1. **State Value vs N**
This shows how the expected value of the initial state changes with grid size.

2. **Runtime vs N**
This shows how computation time increases as the grid size increases.

## Implementation Details

The size of the state space grows as O(N^4), which makes exact computation of the transition kernel impractical for larger values of N. To address this, a sampling-based approximation is used. Instead of constructing the full kernel, transitions are simulated using the environment model.

To further reduce computational complexity, only a subset of the full state space was used during evaluation. In this implementation, approximately 1500 states were sampled from the complete state space. The initial state was always included to ensure correct evaluation of the required value.

This reduction significantly decreases runtime and memory usage while still preserving the overall trends in the results.

## Observations

The runtime increases with N due to the rapid growth of the state space. The state value generally decreases with increasing N, as the predator requires more time to reach the prey in larger grids and future rewards are discounted.

Due to the use of approximation and state sampling, the computed values may not be exact but still capture the overall trends effectively.

## Conclusion

The implementation demonstrates how reinforcement learning techniques can be applied to a predator-prey problem. It highlights the challenges associated with large state spaces and shows how sampling-based methods can be used to obtain approximate yet meaningful results.

