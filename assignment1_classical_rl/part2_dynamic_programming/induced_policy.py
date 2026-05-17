import numpy as np

def induced_policy(Q):
    S, A = Q.shape
    policy = np.zeros((S, A))
    
    for s in range(S):
        best_action = np.argmax(Q[s])
        policy[s, best_action] = 1.0
    
    return policy