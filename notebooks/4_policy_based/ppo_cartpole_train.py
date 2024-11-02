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
            r_list.append([r/100.0])
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

def train_ppo(ppo_net, buffer, optimizer, rollout_length=4, epsilon=0.1, discount=0.99):

    for i in range(rollout_length):
        s, a, r, s_prime, old_p, dm = buffer.sample()

        # Get new action probabilities
        model_output = ppo_net(s)
        dist = Categorical(model_output)
        new_p = model_output.gather(1, a)  # Get probabilities for taken actions
        
        # Calculate probability ratio
        ratios = new_p / old_p.detach()
        surrogate = torch.clamp(ratios, 1-epsilon, 1+epsilon)

        # calculate future rewards
        discounts = torch.tensor([ discount ** i for i in range(len(r)+1) ])
        discounts = discounts.unsqueeze(1).to(device)
        exp_r = sum([a*b for a,b in zip(discounts, r)])

        # normalize rewards

        tensor_loss = torch.min(exp_r * surrogate, exp_r * ratios)

        # include a regularization term
        # this steers new_policy towards 0.5
        # prevents policy to become exactly 0 or 1 helps exploration
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_p*torch.log(old_p+1.e-10)+ \
            (1.0-new_p)*torch.log(1.0-old_p+1.e-10))

        # optim
        loss = - torch.mean(0.01*entropy + tensor_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


env = gym.make("CartPole-v1")
print(f"env.action_space: {env.action_space}")
print(f"env.observation_space: {env.observation_space}")

epoch = 5000
T_horizon = 20
print_every = 20
rollout_length = 3
learning_rate = 0.0005

buffer = ReplayBuffer()
ppo_net = PPONet().to(device)
optimizer = optim.Adam(ppo_net.parameters(), lr=learning_rate)

for i in range(epoch):
    
    s, _ = env.reset()
    score = 0.0
    done = False
    score_queue, avg_scores = deque(maxlen=100), deque(maxlen=100)

    while not done:
        for t in range(T_horizon):

            a, old_prob = ppo_net.act( torch.from_numpy(s).float().unsqueeze(0).to(device) )
            s_p, r, done, trunc, info = env.step(a)
            
            buffer.put((s, a, r, s_p, old_prob, done))

            if done or trunc:
                break

            s = s_p
            score += r

    score_queue.append(score)

    train_ppo(ppo_net, buffer, optimizer, rollout_length)

    # TODO: preformance check
    avg_score = np.mean(score_queue)
    if i % print_every == 0 and i != 0:
        avg_scores.append(avg_score)
        print(f"Episode {i}: Average Score = {np.mean(score_queue)}")

    if avg_score >= 195.0:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, avg_score))
        break