from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import collections
import random

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


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        s_list, a_list, r_list, s_prime_list, done_list = [], [], [], [], []
        mini_batch = random.sample(self.buffer, n)

        for transition in mini_batch:
            s, a, r, s_prime, dm = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_list.append([dm])

        # Warning : Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        # Please consider converting the list to a single numpy.ndarray
        return torch.tensor(np.array(s_list), dtype=torch.float).to(device), torch.tensor(a_list).to(device), \
               torch.tensor(r_list).to(device), torch.tensor(np.array(s_prime_list), dtype=torch.float).to(device), \
               torch.tensor(done_list).to(device)
    
    def size(self):
        return len(self.buffer)


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


def train(Qnet, Qnet_target, buffer, optim):
    for i in range(10):
        s,a,r,s_prime,dm = buffer.sample(batch_size)

        q_out = Qnet(s)
        q_pred = q_out.gather(1, a)

        q_target_out = Qnet_target(s_prime)
        q_target_pred = r + gamma * q_target_out.max(1)[0].unsqueeze(1) * dm

        loss = F.smooth_l1_loss(q_pred, q_target_pred)

        optim.zero_grad()
        loss.backward()
        optim.step()


# env = gym.make("CartPole-v1", render_mode = "human")
env = gym.make("CartPole-v1")
print(f"env.action_space: {env.action_space}")
print(f"env.observation_space: {env.observation_space}")

q = Qnet().to(device)
q_target = Qnet().to(device)
q_target.load_state_dict(q.state_dict())

memory = ReplayBuffer()

optimizer = optim.Adam(q.parameters(), lr=learning_rate)
target_update_interval = 20
print_interval = 100
epoch = 5000

score_list = []

for i in range(epoch):
    # epsilon = 1.0 / (i + 1)
    epsilon = max(0.01, 0.08 - 0.01*(i/200)) #Linear annealing from 8% to 1%

    s, _ = env.reset()
    score = 0.0

    while True:
        a = q.sample_action(torch.from_numpy(s).float(), epsilon)
        s_prime, r, done, truncated, _ = env.step(a)
        dm = 0.0 if done else 1.0
        memory.put((s, a, r/100.0, s_prime, dm))

        score += r
        s = s_prime
        if done or truncated:
            break

    if memory.size() > 2000:
        train(q, q_target, memory, optimizer)
    
    if i % target_update_interval == 0 :
        q_target.load_state_dict(q.state_dict())

    if i % print_interval == 0 and i != 0:
        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                        i, score, memory.size(), epsilon*100))

        # TODO: Plot the average score using matplotlib
        score_list.append(score)

env.close()

# Save trained model weights and optimizer state
torch.save(q.state_dict(), "dqn.pth")