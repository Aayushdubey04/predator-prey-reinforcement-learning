import numpy as np
from simulator import simulator

def state_value_eval(policy, states, state_to_idx, N, gamma=0.99, theta=1e-6, K=3, max_iter=20):
    S, A = policy.shape
    V = np.zeros(S)
    
    for iteration in range(max_iter):
        delta = 0
        
        for s in range(S):
            predator, prey = states[s]
            v = V[s]
            
            new_v = 0
            
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
                
                new_v += policy[s, a] * (total / K)
            
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
        
        print(f"Iteration {iteration+1}, delta = {delta:.4f}")
        
        if delta < theta:
            break
    
    return V.reshape(-1, 1)