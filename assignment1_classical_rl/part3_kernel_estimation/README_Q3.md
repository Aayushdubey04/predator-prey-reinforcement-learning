# EE675 Assignment 1 - Q3

## Author
Aayush Dubey

## Description
This part of the assignment focuses on estimating the transition dynamics of the predator-prey environment using sampling and analyzing how the number of samples affects the accuracy and stability of the results.

Instead of explicitly computing the full transition kernel, which is computationally expensive due to the large state space, a sampling-based approach is used.

## Overview

For each state-action pair, multiple simulations are performed to estimate the expected outcomes. The number of samples used is denoted by K.

The goal is to study how varying K affects:
- The accuracy of the estimated value function
- The stability of the results across different random seeds

## File Structure 
- estimate_kernel.py
- value_iteration_estimated.py
- main_q3.py 

Note: The implementation reuses components from earlier parts of the assignment, 
particularly the simulator and value iteration modules.

## How to Run

1. Open a terminal in the project directory

2. Install dependencies: pip install numpy matplotlib

3. Run : python main_Q3.py

## Output

This program generates two plots:

1. **Mean Difference vs K**  
This plot shows how the average difference between the estimated Q-function and the reference Q-function varies with the number of samples K.

2. **Std Deviation vs K**  
This plot shows how the variability of the estimation changes with K across different random seeds. 

## Observations

- Increasing K improves the stability of the estimates  
- The standard deviation decreases with larger K  
- The mean difference shows fluctuations for small K and improves for larger values  
- Larger K leads to better approximation of the underlying dynamics  

## Conclusion

This part demonstrates the impact of sampling on estimating transition dynamics in reinforcement learning. Increasing the number of samples improves both the accuracy and stability of the results, highlighting the importance of sufficient sampling in approximate methods.