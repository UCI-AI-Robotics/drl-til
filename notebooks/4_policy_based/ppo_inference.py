# Standard library imports
from matplotlib import pyplot as plt
from collections import deque  # For efficient queue operations
import time

# Third-party imports
import gymnasium as gym  # OpenAI Gym environment
import numpy as np  # For numerical computations

# PyTorch imports
import torch  # Main PyTorch package
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Neural network functions
import torch.optim as optim  # Optimization algorithms
from torch.distributions import Categorical  # For sampling actions from probability distributions


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPOCPU(nn.Module):
    def __init__(self):
        super(PPOCPU, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                # advantage = gamma * lmbda * advantage + delta_t[0]
                advantage = gamma * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # # calculate future rewards
            # discounts = torch.tensor([ gamma ** i for i in range(len(r)+1) ])
            # discounts = discounts.unsqueeze(1)
            # advantage = sum([a*b for a,b in zip(discounts, r)])

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = pi_a / prob_a

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())
            # loss = -torch.min(surr1, surr2)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

class PPOGPU(nn.Module):
    def __init__(self):
        super(PPOGPU, self).__init__()

        self.data = []
        
        self.pi_layer = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        self.v_layer = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def pi(self, x, softmax_dim = 0):
        x = x.to(device)
        layer_output = self.pi_layer(x)
        output = F.softmax(layer_output, dim=softmax_dim)
        return output
    
    def v(self, x):
        x = x.to(device)
        layer_output = self.v_layer(x)
        return layer_output
    
    def put_data(self, data):
        self.data.append(data)
        return
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst).to(device), \
                                          torch.tensor(r_lst).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(device), torch.tensor(prob_a_lst).to(device)
        
        self.data = []
        return s, a, r, s_prime, prob_a, done_mask 

    def train_net(self, optimizer, gamma, epsilon, importance_iter):

        s, a, r, sp, prob, dm  = self.make_batch()
        gamma = torch.tensor(gamma, dtype=torch.float).to(device)

        for i in range(importance_iter):

            td_target = r + gamma * self.v(sp) * dm
            delta = td_target - self.v(s)

            delta = delta.cpu().detach().numpy()
            gamma = gamma.cpu()

            # calc advantage
            discounts = np.array([ [gamma ** i] for i in range(len(delta)+1) ])
            advantage_lst2 = []
            for j in range(len(delta)):
                discounted_return = sum(delta[j:] * discounts[:-(1+j)])
                advantage_lst2.append(discounted_return)
            advantage = torch.tensor(advantage_lst2, dtype=torch.float).to(device)

            pi = self.pi(s, softmax_dim=1)
            prob_new = pi.gather(1, a)
            ratio = prob_new / prob

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio * advantage, 1-epsilon, 1+epsilon)
            surr  = torch.min(surr1, surr2)

            loss = -surr + F.smooth_l1_loss(self.v(s), td_target.detach())

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

pth_file = 'ppo_cartpole_gpu_20241102_185742.pth'
model = PPOGPU().to(device)

# pth_file = 'ppo_cartpole_cpu_20241102_185658.pth'
# model = PPOCPU().to(device)

env = gym.make('CartPole-v1', render_mode="human")
env.metadata['render_fps'] = 30

model.load_state_dict(torch.load(pth_file))

s, _ = env.reset()
while True:
    prob = model.pi(torch.from_numpy(s).float().to(device))
    m = Categorical(prob)
    a = m.sample().item()
    s_prime, r, done, truncated, info = env.step(a)
    s = s_prime
    if done or truncated:
        break

env.close()