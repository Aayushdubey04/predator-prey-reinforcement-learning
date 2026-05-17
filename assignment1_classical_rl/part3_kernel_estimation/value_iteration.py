import numpy as np
from simulator import simulator

def value_iteration(states, state_to_idx, N, gamma=0.99, theta=1e-2, K=2, max_iter=20):
    S = len(states)
    A = 5  # number of actions
    
    Q = np.zeros((S, A))
    V = np.zeros(S)
    
    for iteration in range(max_iter):
        delta = 0
        
        for s in range(S):
            predator, prey = states[s]
            v = V[s]
            
            q_values = []
            
            for a in range(A):
                total = 0
                
                for _ in range(K):
                    p_next, prey_next, reward = simulator(N, predator, prey, a)
                    s_next = (p_next, prey_next)
                    
                    if s_next in state_to_idx:
                        idx = state_to_idx[s_next]
                        total += reward + gamma * V[idx]
                    else:
                        total += reward + gamma * np.mean(V)
                
                q = total / K
                Q[s, a] = q
                q_values.append(q)
            
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
        
        print(f"VI Iteration {iteration+1}, delta = {delta:.4f}")
        
        if delta < theta:
            break
    
    return Q