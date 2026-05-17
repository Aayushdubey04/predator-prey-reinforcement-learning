import torch
import random
from simulator import simulator

def run_episode(policy_net, N=5, max_steps=50):
    
    predator_pos = (random.randint(1, N), random.randint(1, N))
    prey_pos = (random.randint(1, N), random.randint(1, N))
    
    states, actions_taken, rewards = [], [], []
    
    for _ in range(max_steps):
        
        state = torch.tensor([
            predator_pos[0], predator_pos[1],
            prey_pos[0], prey_pos[1]
        ], dtype=torch.float32) / N
        
        probs = policy_net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        predator_next, prey_next, reward = simulator(
            N, predator_pos, prey_pos, action.item()
        )
        
        states.append(state)
        actions_taken.append(action)
        rewards.append(reward)
        
        predator_pos, prey_pos = predator_next, prey_next
        
        if reward == 1:
            break
    
    return states, actions_taken, rewards