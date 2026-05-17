# EE675 ASSIGNMENT 2 - Q2

## Author 
Aayush Dubey 

## Description
This part of assignment implements a policy gradient method (REINFORCE) for a predator-prey grid environment.
The objective is to train a policy that maximizes the expected reward (capturing the prey).

## File Structure 
- policy_network.py      
- simulator.py             
- gradient_estimate.py
- episode.py              
- simple_SGA.py
- train_adam.py             
- compare.py               
- README.md


## How to Run

1. Open a terminal in the project directory 

2. Install dependencies : pip install numpy matplotlib torch
3. Run the code : python simple_SGA.py
                : python train_adam.py
                : python compare.py


## Output
- SGA
1. High variance
2. Noisy learning curve
3. .Unstable updates
-Adam
1. Smoother learning curve
2. Faster convergence
3. More stable performance

## Observations 
- SGA struggles due to fixed learning rate and high variance
- Adam adapts learning rates → better stability
- Reward curves are not strictly increasing due to stochasticity

## Conclusion 
Adam significantly outperforms simple stochastic gradient ascent in:

- Stability
- Convergence speed
- Final reward

This demonstrates the importance of adaptive optimization in reinforcement learning.