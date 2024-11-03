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

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

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

            # advantage_lst = []
            # advantage = 0.0
            # for delta_t in delta[::-1]:
            #     # advantage = gamma * lmbda * advantage + delta_t[0]
            #     advantage = gamma * advantage + delta_t[0]
            #     advantage_lst.append([advantage])
            # advantage_lst.reverse()
            # advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # calc advantage
            discounts = np.array([ [gamma ** i] for i in range(len(delta)+1) ])
            advantage_lst2 = []
            for j in range(len(delta)):
                discounted_return = sum(delta[j:] * discounts[:-(1+j)])
                advantage_lst2.append(discounted_return)
            advantage = torch.tensor(advantage_lst2, dtype=torch.float).to(device)

            # print(f"advantage_lst: {advantage_lst}")
            # print(f"advantage_lst2: {advantage_lst2}")

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

gamma = 0.98
epsilon = 0.1
importance_iter = 3

epoch = 5000
T_horizon = 20
print_interval = 20
learning_rate = 0.0005

env = gym.make('CartPole-v1')
model = PPO().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

score_queue = deque(maxlen=100)
avg_scores = deque(maxlen=100)

for n_epi in range(epoch):

    score = 0.0
    done = False
    s, _ = env.reset()

    while not done:
        for t in range(T_horizon):
            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()

            sp, r, done, trunc, info = env.step(a)
            model.put_data( (s, a, r/100.0, sp, prob[a].item(), done ) )
            s = sp

            score += r
            if done or trunc:
                break

        model.train_net(optimizer=optimizer, gamma=gamma, epsilon=epsilon, importance_iter=importance_iter)

    score_queue.append(score)
    avg_score = np.mean(score_queue)
    avg_scores.append(avg_score)

    if avg_score >= 195.0:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(n_epi - print_interval, avg_score))
        break

    if n_epi % print_interval == 0 and n_epi != 0:
        print(f"Episode {n_epi}: Average Score = {avg_score}")

env.close()

# Save trained model weights and optimizer state with current timestamp
torch.save(model.state_dict(), f"ppo_cartpole_gpu_{time.strftime('%Y%m%d_%H%M%S')}.pth")

# plot performance
plt.plot(np.linspace(0, epoch, len(avg_scores), endpoint=False), np.asarray(avg_scores))
plt.xlabel('Episode Number')
plt.ylabel('Average Reward (Over Next %d Episodes)' % print_interval)
plt.show()

# Environment solved in 350 episodes!     Average Score: 195.85