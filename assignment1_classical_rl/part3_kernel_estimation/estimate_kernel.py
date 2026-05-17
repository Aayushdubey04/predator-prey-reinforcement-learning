import numpy as np
from simulator import simulator

def estimate_kernel(states, state_to_idx, N, K=5):
    S = len(states)
    A = 5
    
    P_est = {}
    
    for s in range(S):
        predator, prey = states[s]
        
        for a in range(A):
            counts = {}
            
            for _ in range(K):
                p_next, prey_next, _ = simulator(N, predator, prey, a)
                s_next = (p_next, prey_next)
                
                if s_next in state_to_idx:
                    idx = state_to_idx[s_next]
                else:
                    idx = np.random.randint(0, len(states))
                
                counts[idx] = counts.get(idx, 0) + 1
            
            # Normalize
            total = sum(counts.values())
            probs = {k: v/total for k, v in counts.items()} if total > 0 else {}
            
            P_est[(s, a)] = probs
    
    return P_est