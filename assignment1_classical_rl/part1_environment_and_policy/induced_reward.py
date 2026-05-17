import numpy as np

def induced_reward(R, policy):
    return np.sum(policy * R, axis=1).reshape(-1,1)