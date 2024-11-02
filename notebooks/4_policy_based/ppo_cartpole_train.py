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

# Set random seed for reproducibility
torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


class ReplayBuffer():
    def __init__(self, buffer_limit=1000):
        self.buffer = deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self):
        s_list, a_list, r_list, s_prime_list, old_prob_list, done_list = [], [], [], [], [], []

        for transition in self.buffer:
            s, a, r, s_prime, old_p, dm = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            old_prob_list.append([old_p])
            done_list.append([dm])

        # Warning : Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        # Please consider converting the list to a single numpy.ndarray
        return torch.tensor(np.array(s_list), dtype=torch.float).to(device), torch.tensor(a_list).to(device), \
               torch.tensor(r_list).to(device), torch.tensor(np.array(s_prime_list), dtype=torch.float).to(device), \
               torch.tensor(old_prob_list).to(device), torch.tensor(done_list).to(device)

class PPONet(nn.Module):
    def __init__(self):
        super(PPONet, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(device)
        output = self.layer(x)
        return output
    
    def act(self, s):
        # s = torch.from_numpy(s).float().unsqueeze(0).to(device)
        model_output = self.forward(s)
        prob = Categorical(model_output)

        action = prob.sample().item()
        old_prob = model_output[0, action].item()

        return action, old_prob

def train_ppo(ppo_net, buffer, optimizer, epsilon=0.1, discount=0.99):

    for i in range(4):
        s, a, r, s_prime, old_p, dm = buffer.sample()

        # Get new action probabilities
        model_output = ppo_net(s)
        dist = Categorical(model_output)
        new_p = model_output.gather(1, a)  # Get probabilities for taken actions
        
        # Calculate probability ratio
        ratios = new_p / old_p
        surrogate = torch.clamp(ratios, 1-epsilon, 1+epsilon)

        # calculate future rewards
        discounts = [ discount ** i for i in range(len(r)+1) ]

        exp_ratios = []
        for j, ratio in enumerate(ratios):
            discounted_return = sum(r[j:] * discounts[:-(1+j)])
            exp_ratios.append(ratio * discounted_return)

        exp_r = torch.tensor(exp_ratios)

        surrogate = torch.min(exp_r * surrogate, exp_r * ratios)

        # optim
        loss = -surrogate

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


env = gym.make("CartPole-v1")
print(f"env.action_space: {env.action_space}")
print(f"env.observation_space: {env.observation_space}")

epoch = 500
print_every = 100
learning_rate = 0.01

buffer = ReplayBuffer()
ppo_net = PPONet().to(device)
optimizer = optim.Adam(ppo_net.parameters(), lr=learning_rate)

for i in range(epoch):
    
    s, _ = env.reset()
    score = 0.0
    score_queue, avg_scores = deque(maxlen=100), deque(maxlen=100)

    while True:
        a, old_prob = ppo_net.act( torch.from_numpy(s).float().unsqueeze(0).to(device) )
        s_p, r, done, trunc, info = env.step(a)
        
        buffer.put((s, a, r, s_p, old_prob, done))

        if done or trunc:
            break

        s = s_p
        score += r

    score_queue.append(score)

    train_ppo(ppo_net, buffer, optimizer)

    # TODO: preformance check

    if i % print_every == 0 and i != 0:
        avg_scores.append(np.mean(score_queue))
        print(f"Episode {i}: Average Score = {np.mean(score_queue)}")