import numpy as np
from kernel import get_states
from simulator import simulator, actions

def reward_function(N):
    states = get_states(N)
    S = len(states)
    A = len(actions)
    
    R = np.zeros((S, A))
    
    for i, s in enumerate(states):
        predator, prey = s
        
        for a in range(A):
            _, _, reward = simulator(N, predator, prey, a)
            R[i, a] = reward
    
    return R