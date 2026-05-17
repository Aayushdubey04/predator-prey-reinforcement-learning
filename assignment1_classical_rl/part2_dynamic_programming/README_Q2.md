# EE675 Assignment 1 - Q2

## Author
Aayush Dubey

## Description
This part extends the predator-prey environment to compute optimal policies using Value Iteration and Policy Iteration. The goal is to compare both methods in terms of accuracy and runtime.

## File Structure

- value_iteration.py  
  Implements Value Iteration using sampling-based updates.

- induced_policy.py  
  Extracts the greedy policy from Q-values.

- policy_iteration.py  
  Implements Policy Iteration with evaluation and improvement steps.

- main_Q2.py  
  Runs experiments and generates plots.

## How to Run

1. Open a terminal in the project directory

2. Install dependencies: pip intall numpy matplotlib

3. Run : python main_Q2.py

## Output

This program generates two plots:

1. **L1 Difference between Value Iteration and Policy Iteration**  
   This plot shows the difference between the Q-values obtained using Value Iteration and Policy Iteration. Due to sampling-based approximation and reduced state space, the difference is non-zero, but it generally decreases for larger values of N.

2. **Runtime Comparison**  
   This plot compares the execution time of Value Iteration and Policy Iteration. Value Iteration is generally faster, while Policy Iteration takes more time due to repeated policy evaluation and improvement steps.

 ## Output

This program generates two plots:

1. **L1 Difference between Value Iteration and Policy Iteration**  
   This plot shows the difference between the Q-values obtained using Value Iteration and Policy Iteration.

2. **Runtime Comparison**  
   This plot compares the execution time of Value Iteration and Policy Iteration. 

## Observations

- Value Iteration is significantly faster than Policy Iteration
- Policy Iteration is computationally more expensive due to repeated evaluations
- The L1 difference between methods is non-zero due to approximation
- The difference decreases for larger N due to sampling effects

## Conclusion

This implementation demonstrates the trade-off between Value Iteration and Policy Iteration in large state-space problems. Sampling-based approximations enable practical computation while maintaining meaningful comparisons.
