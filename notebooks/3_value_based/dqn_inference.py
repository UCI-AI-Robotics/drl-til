from matplotlib import pyplot as plt
from collections import deque
import gymnasium as gym
import numpy as np
import collections
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# learning_rate = 0.005
learning_rate = 0.001
gamma         = 0.99
buffer_limit  = 50000
batch_size    = 32

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = x.to(device)
        return self.layer(x)

    def sample_action(self, obs, epsilon):
        obs = obs.to(device)
        pred = self.forward(obs)
        coin = random.random()

        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return pred.argmax().item()

# load model
model_path = "dqn_20241027_143435.pth"
q = Qnet().to(device)
q.load_state_dict(torch.load(model_path))

env = gym.make("CartPole-v1", render_mode = "human")
env.metadata['render_fps'] = 30
epsilon = 0.01

while True:
    s, _ = env.reset()
    a = q.sample_action(torch.from_numpy(s).float(), epsilon)
    s_prime, r, done, truncated, _ = env.step(a)

    if done or truncated:
        break

    s = s_prime

env.close()