import time
import matplotlib.pyplot as plt

from kernel import get_states
from policy import sample_policy
from value_eval import state_value_eval

def run():
    Ns = [5, 10, 15, 20, 25]   # required values
    
    values = []
    times = []
    
    for N in Ns:
        print(f"Running for N = {N}")
        
        start = time.time()
        
        # Use only subset of states (IMPORTANT)
        states = get_states(N)[:1500]
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        # Policy
        policy = sample_policy(N)[:len(states)]
        
        # Value evaluation
        V = state_value_eval(policy, states, state_to_idx, N)
        
        # Initial state
        init_state = ((1, 1), (N, N))
        
        if init_state in state_to_idx:
            idx = state_to_idx[init_state]
            values.append(V[idx][0])
        else:
            values.append(0)   # fallback if not in sampled states
        
        times.append(time.time() - start)
    
    #  Plot 1: Value vs N
    plt.plot(Ns, values, marker='o')
    plt.xlabel("N")
    plt.ylabel("State Value")
    plt.title("Value vs N")
    plt.grid()
    plt.show()
    
    # Plot 2: Runtime vs N
    plt.plot(Ns, times, marker='o')
    plt.xlabel("N")
    plt.ylabel("Time (sec)")
    plt.title("Runtime vs N")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run()