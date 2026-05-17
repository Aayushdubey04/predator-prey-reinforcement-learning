import time
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

from kernel import get_states
from value_iteration import value_iteration
from policy_iteration import policy_iteration

def run_q2():
    Ns = [5, 10, 15, 20, 25]
    
    differences = []
    times_vi = []
    times_pi = []
    
    for N in Ns:
        print(f"\nRunning Q2 for N = {N}")
        
        all_states = get_states(N)
        init_state = ((1,1),(N,N))
        
        states = [init_state] + [s for s in all_states if s != init_state][:1499]
        state_to_idx = {s:i for i,s in enumerate(states)}
        
        # Value Iteration
        start = time.time()
        Q_vi = value_iteration(states, state_to_idx, N)
        times_vi.append(time.time() - start)
        
        # Policy Iteration
        start = time.time()
        Q_pi = policy_iteration(states, state_to_idx, N)
        times_pi.append(time.time() - start)
        
        # L1 difference
        diff = np.sum(np.abs(Q_vi - Q_pi))
        differences.append(diff)
    
    # Plot difference
    plt.figure()
    plt.plot(Ns, differences, marker='o')
    plt.xlabel("N")
    plt.ylabel("L1 Difference")
    plt.title("Difference between VI and PI")
    plt.grid()
    plt.show()
    
    # Plot runtime
    plt.figure()
    plt.plot(Ns, times_vi, marker='o', label="Value Iteration")
    plt.plot(Ns, times_pi, marker='o', label="Policy Iteration")
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.title("Runtime Comparison")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_q2()
