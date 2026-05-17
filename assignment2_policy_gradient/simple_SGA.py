import torch
from policy_network import PolicyNetwork
from gradient_estimate import gradient_estimate
import matplotlib.pyplot as plt

def train_sga():
    policy_net = PolicyNetwork()
    rewards_history = []
    lr = 0.005
    
    for it in range(500):
        loss, avg_reward = gradient_estimate(policy_net)
        
        for param in policy_net.parameters():
            param.grad = None
        
        loss.backward()
        
        with torch.no_grad():
            for param in policy_net.parameters():
                param += lr * param.grad
        
        rewards_history.append(avg_reward)
        
        if it % 50 == 0:
           print(f"[SGA] Iter {it}: Reward {avg_reward:.4f}")
    
    plt.plot(rewards_history)
    plt.title("SGA Learning Curve")
    plt.show()

if __name__ == "__main__":
    train_sga()