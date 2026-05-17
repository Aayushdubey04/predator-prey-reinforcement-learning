import numpy as np

def q_value_eval(policy, P, R, gamma=0.99, theta=1e-6):
    S, A = R.shape
    Q = np.zeros((S, A))
    
    V = np.zeros(S)
    
    while True:
        delta = 0
        
        for s in range(S):
            for a in range(A):
                q = R[s,a] + gamma * np.dot(P[s*A + a], V)
                delta = max(delta, abs(Q[s,a] - q))
                Q[s,a] = q
        
        for s in range(S):
            V[s] = np.dot(policy[s], Q[s])
        
        if delta < theta:
            break
    
    return Q