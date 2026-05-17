import numpy as np
import matplotlib.pyplot as plt
import time
import random

from kernel import get_states
from value_iteration import value_iteration
from estimate_kernel import estimate_kernel
from value_iteration_estimated import value_iteration_estimated

def run_q3():
    N = 5
    Ks = [5, 10, 15, 20, 25]
    
    all_states = get_states(N)
    init_state = ((1,1),(N,N))
    
    states = [init_state] + [s for s in all_states if s != init_state][:1499]
    state_to_idx = {s:i for i,s in enumerate(states)}
    
    # Exact Q using simulator-based VI
    Q_exact = value_iteration(states, state_to_idx, N)
    
    all_means = []
    all_stds = []
    
    for K in Ks:
        diffs = []
    
        for seed in range(10):
             np.random.seed(seed)
        
             Q_est = value_iteration(states, state_to_idx, N, K=50)
        
             diff = np.sum(np.abs(Q_exact - Q_est))
             diffs.append(diff)
    
        all_means.append(np.mean(diffs))
        all_stds.append(np.std(diffs))
    
    # Plot mean
    plt.figure()
    plt.plot(Ks, all_means, marker='o')
    plt.xlabel("K")
    plt.ylabel("Mean L1 Difference")
    plt.title("Mean Difference vs K")
    plt.grid()
    plt.show()
    
    # Plot std
    plt.figure()
    plt.plot(Ks, all_stds, marker='o')
    plt.xlabel("K")
    plt.ylabel("Std Deviation")
    plt.title("Std Deviation vs K")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_q3()