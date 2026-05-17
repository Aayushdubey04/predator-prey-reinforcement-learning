import matplotlib.pyplot as plt
import numpy as np

from simple_SGA import train_sga
from train_adam import train_adam

sga_rewards = train_sga()
adam_rewards = train_adam()

# Smooth both
sga_smooth = np.convolve(sga_rewards, np.ones(20)/20, mode='valid')
adam_smooth = np.convolve(adam_rewards, np.ones(20)/20, mode='valid')

plt.plot(sga_smooth, label="SGA")
plt.plot(adam_smooth, label="Adam")

plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("SGA vs Adam Comparison")
plt.legend()
plt.show()