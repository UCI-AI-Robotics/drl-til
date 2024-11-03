import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

gamma = 0.99
epsilon = 0.1
T_horizon = 20
importance_iter = 4
learning_rate = 0.0005

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
        layer_output = self.pi_layer(x)
        output = F.softmax(layer_output, dim=softmax_dim)
        return output
    
    def v(self, x):
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
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self, optimizer):

        s, a, r, sp, dm, prob = self.make_batch()

        for i in range(importance_iter):

            td_target = r + gamma * self.v(sp) * dm
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


            # # calc advantage
            # discounts = torch.tensor([ gamma ** i for i in range(len(r)) ])
            # discounts = discounts.unsqueeze(1)
            # advantage = sum([a*b for a,b in zip(discounts, r)])

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


def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    print_interval = 20
    score_queue = deque(maxlen=100)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        score = 0.0

        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break
            
            model.train_net(optimizer)

        score_queue.append(score)
        avg_score = np.mean(score_queue)

        if avg_score >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(n_epi - print_interval, avg_score))
            break

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"Episode {n_epi}: Average Score = {avg_score}")

    env.close()

if __name__ == '__main__':
    main()

