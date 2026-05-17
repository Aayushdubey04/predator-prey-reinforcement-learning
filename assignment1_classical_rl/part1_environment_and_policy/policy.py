import numpy as np
from kernel import get_states
from simulator import actions

def manhattan(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def sample_policy(N):
    states = get_states(N)
    S = len(states)
    A = len(actions)
    
    policy = np.zeros((S, A))
    
    for i, s in enumerate(states):
        predator, prey = s
        
        distances = []
        for a in range(A):
            dx, dy = actions[a]
            new_pos = (predator[0]+dx, predator[1]+dy)
            dist = manhattan(new_pos, prey)
            distances.append(dist)
        
        best_action = np.argmin(distances)
        
        for a in range(A):
            if a == best_action:
                policy[i, a] = 0.5
            else:
                policy[i, a] = 0.5 / (A-1)
    
    return policy