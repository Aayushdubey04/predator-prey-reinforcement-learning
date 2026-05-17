import numpy as np
from simulator import simulator, actions

def get_states(N):
    states = []
    for px in range(1, N+1):
        for py in range(1, N+1):
            for qx in range(1, N+1):
                for qy in range(1, N+1):
                    states.append(((px, py), (qx, qy)))
    return states

def kernel(N):
    states = get_states(N)
    S = len(states)
    A = len(actions)
    
    state_to_idx = {s:i for i,s in enumerate(states)}
    
    P = np.zeros((S*A, S))
    
    for i, s in enumerate(states):
        predator, prey = s
        
        for a in range(A):
            row = i*A + a
            
            # Monte Carlo approx for transition
            counts = np.zeros(S)
            K = 20

        
            
            for _ in range(K):
                p_next, prey_next, _ = simulator(N, predator, prey, a)
                s_next = (p_next, prey_next)
                j = state_to_idx[s_next]
                counts[j] += 1
            
            P[row] = counts / K
    
    return P, states