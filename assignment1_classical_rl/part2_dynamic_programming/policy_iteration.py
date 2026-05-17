import numpy as np
from simulator import simulator
from value_eval import state_value_eval

def policy_iteration(states, state_to_idx, N, gamma=0.99, K=2, max_iter=10):
    S = len(states)
    A = 5
    
    # Start with random policy
    policy = np.ones((S, A)) / A
    
    for iteration in range(max_iter):
        print(f"PI Iteration {iteration+1}")
        
        # Policy Evaluation
        V = state_value_eval(policy, states, state_to_idx, N).flatten()
        
        stable = True
        
        # Policy Improvement
        for s in range(S):
            predator, prey = states[s]
            
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
                
                q_values.append(total / K)
            
            best_action = np.argmax(q_values)
            
            if np.argmax(policy[s]) != best_action:
                stable = False
            
            policy[s] = np.eye(A)[best_action]
        
        if stable:
            break
    
    # Compute final Q
    Q = np.zeros((S, A))
    for s in range(S):
        predator, prey = states[s]
        
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
            
            Q[s, a] = total / K
    
    return Q