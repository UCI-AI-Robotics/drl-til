import gymnasium as gym
import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

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

        return torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
               torch.tensor(r_list), torch.tensor(s_prime_list, dtype=torch.float), \
               torch.tensor(done_list)


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
        return self.layer(x)

    def sample_action(self, obs, epsilon):
        pred = self.forward(obs)
        coin = random.random()

        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return pred.argmax().item()

env = gym.make("CartPole-v1", render_mode = "human")
print(f"env.action_space: {env.action_space}")
print(f"env.observation_space: {env.observation_space}")

s, _ = env.reset()

replay_buffer = ReplayBuffer()
q_net = Qnet()

# while True:
#     a = env.action_space.sample()
#     s, r, done, truncated, info = env.step(a)

#     if done or truncated:
#         break

#     env.render()

env.close()