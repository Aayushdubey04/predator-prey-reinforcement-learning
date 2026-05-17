import numpy as np

def induced_kernel(P, policy):
    S = policy.shape[0]
    A = policy.shape[1]
    
    P_pi = np.zeros((S, S))
    
    for s in range(S):
        for a in range(A):
            P_pi[s] += policy[s, a] * P[s*A + a]
    
    return P_pi