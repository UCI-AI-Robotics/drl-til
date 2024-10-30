# Core libraries
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gymnasium
import gymnasium as gym

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

env = gym.make('CartPole-v0', render_mode = "human")
env.metadata['render_fps'] = 30

print('observation space:', env.observation_space)
print('action space:', env.action_space)

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        ).to(device)

    def forward(self, x):
        x = x.to(device)
        output = self.layer(x)
        return output

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        output = self.forward(state)  # No need for extra .to(device) since forward handles it
        prob = Categorical(output)

        # Caution : action is stochastic
        # if you call sample() multiple times, you will get different actions
        action = prob.sample()
        return action.item(), prob.log_prob(action)

# load model
model_path = "reinforce_20241030_105048.pth"
policy_net = PolicyNet().to(device)
policy_net.load_state_dict(torch.load(model_path))

while True:
    s, _ = env.reset()
    a, _ = policy_net.act(s)
    s, r, done, truncated, _ = env.step(a)

    if done or truncated:
        break

env.close()