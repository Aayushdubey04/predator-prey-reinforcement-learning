import numpy as np

def value_iteration_estimated(states, state_to_idx, P_est, gamma=0.99, theta=1e-2, max_iter=20):
    S = len(states)
    A = 5
    
    V = np.zeros(S)
    Q = np.zeros((S, A))
    
    for _ in range(max_iter):
        for s in range(S):
            q_values = []
            
            for a in range(A):
                total = 0
                
                for next_s, prob in P_est.get((s,a), {}).items():
                    total += prob * (gamma * V[next_s])
                    reward = 1 if states[next_s][0] == states[next_s][1] else 0
                    total += prob * (reward + gamma * V[next_s])
                
                Q[s,a] = total
                q_values.append(Q[s,a])
            
            V[s] = max(q_values)
    
    return Q