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

env = gym.make('CartPole-v0')
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


n_episodes = 1000
learning_rate = 0.0075
avg_scores = deque(maxlen=1000)

policy_net = PolicyNet().to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=50):
    
    score_queue = deque(maxlen=100)
    
    for i in range(n_episodes):
        
        log_probs, rewards = [], []
        state, _ = env.reset()
        
        while True:
            action, log_prob = policy_net.act(state)
            state, reward, done, trunc, _ = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)

            if done or trunc:
                break
        
        score_queue.append(sum(rewards))

        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        R = sum([a*b for a,b in zip(discounts, rewards)])

        probs = []
        for j, log_prob in enumerate(log_probs):
            discounted_return = sum(rewards[j:] * discounts[:-(1+j)])
            probs.append(-discounted_return * log_prob)
            
            # print(discounted_return, R)
            # probs.append(-log_prob * R)
        probs_loss = torch.cat(probs).sum()

        optimizer.zero_grad()
        probs_loss.backward()
        optimizer.step()

        avg_score = np.mean(score_queue)

        if i % print_every == 0 and i != 0:
            avg_scores.append(avg_score)
            print(f"Episode {i}: Average Score = {avg_score}")
        
        if avg_score >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, avg_score))
            break

reinforce(n_episodes=1000, gamma=0.99)

env.close()

# Save trained model weights and optimizer state with current timestamp
torch.save(policy_net.state_dict(), f"reinforce_{time.strftime('%Y%m%d_%H%M%S')}.pth")

# plot performance
plt.plot(np.linspace(0, n_episodes, len(avg_scores), endpoint=False), np.asarray(avg_scores))
plt.xlabel('Episode Number')
plt.ylabel('Average Reward (Over Next %d Episodes)' % 50)
plt.show()

# Environment solved in 641 episodes!     Average Score: 195.88