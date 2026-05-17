import torch
from episode import run_episode

def gradient_estimate(policy_net, episodes=30, gamma=0.99):
    total_loss = 0
    total_reward = 0

    for _ in range(episodes):
        states, actions, rewards = run_episode(policy_net)
        
        total_reward += sum(rewards)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:

            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # REINFORCE loss
        for s, a, Gt in zip(states, actions, returns):
            probs = policy_net(s)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(a)
            total_loss += -log_prob * Gt

    avg_reward = total_reward / episodes
    return total_loss / episodes, avg_reward