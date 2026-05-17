import torch
import torch.optim as optim
from policy_network import PolicyNetwork
from gradient_estimate import gradient_estimate
import matplotlib.pyplot as plt
import numpy as np

def train_adam():
    policy_net = PolicyNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    rewards_history = []

    for it in range(500):
        optimizer.zero_grad()

        loss, avg_reward = gradient_estimate(policy_net)
        loss.backward()

        optimizer.step()

        rewards_history.append(avg_reward)

        if it % 50 == 0:
            print(f"[Adam] Iter {it}: Reward {avg_reward:.4f}")

    # ---- Plot ----
    plt.plot(rewards_history, label="Raw")

    # Smoothed curve
    smoothed = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
    plt.plot(smoothed, label="Smoothed")

    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Adam Learning Curve")
    plt.legend()
    plt.show()

    return rewards_history


if __name__ == "__main__":
    train_adam()