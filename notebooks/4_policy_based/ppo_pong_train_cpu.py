# Standard library imports
from collections import deque  # For efficient queue operations
import time  # For timestamps and timing operations
import random

# Visualization imports 
import matplotlib  # Base matplotlib library
import matplotlib.pyplot as plt  # For plotting training metrics and visualizations

# Third-party imports
# Caution: Gym-Atari required
# Check availablility: python3 -m pip install "gymnasium[atari, accept-rom-license]"
import gymnasium as gym  # OpenAI Gym environment
import numpy as np  # For numerical computations

# PyTorch imports
import torch  # Main PyTorch package
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Neural network functions
import torch.optim as optim  # Optimization algorithms
from torch.distributions import Categorical  # For sampling actions from probability distributions

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.conv1 = nn.Conv2d(3, 6, kernel_size=2, stride=2) # 210x160x3 -> 105x80x6
        self.conv2 = nn.Conv2d(6, 12, kernel_size=2, stride=2) # 105x80x6 -> 52x40x12   
        self.conv3 = nn.Conv2d(12, 24, kernel_size=2, stride=2) # 52x40x12 -> 26x20x24
        self.conv4 = nn.Conv2d(24, 48, kernel_size=2, stride=2) # 26x20x24 -> 13x10x48 = 6240

        self.fc1 = nn.Linear(13*10*48, 64) # 6240 -> 64
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 1)

        self.fc4 = nn.Linear(210*160*3, 256) # 100800 -> 256
        self.fc5 = nn.Linear(256, 64) # 256 -> 64
        self.fc6 = nn.Linear(64, 1) # 64 -> 1

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = np.swapaxes(x, 0, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, 13*10*48)
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)
        prob = F.softmax(x, dim=softmax_dim)

        return prob
    
    def v(self, x):
        x = x.view(-1, 210*160*3)
        
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        v = self.fc6(x)

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


# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
    print(np.shape(img)) # (80, 80)
    return img

def plot_preprocessing(env):

    frame, _ = env.reset() # (210, 160, 3)
    for n_epi in range(30):
        frame, _, _, _, _ = env.step(1)

    plt.subplot(1,2,1)
    plt.imshow(frame)
    plt.title('original image')

    plt.subplot(1,2,2)
    plt.title('preprocessed image')

    # 80 x 80 black and white image
    plt.imshow(preprocess_single(frame), cmap='Greys')
    plt.show()

def main():
    # print(gym.envs.registration.registry.keys())
    RIGHT = 4
    LEFT  = 5

    env = gym.make('ALE/Pong-v5')
    model = PPO()
    
    print_interval = 20
    score_queue = deque(maxlen=100)

    # plot_preprocessing(env)

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        score = 0.0

        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                a = RIGHT if random.random() < prob else LEFT
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob, done))
                s = s_prime

                score += r
                if done:
                    break
            
            model.train_net()

        score_queue.append(score)
        avg_score = np.mean(score_queue)

        if avg_score >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(n_epi - print_interval, avg_score))
            break

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"Episode {n_epi}: Average Score = {avg_score}")

    env.close()

    # # Save trained model weights and optimizer state with current timestamp
    # torch.save(model.state_dict(), f"ppo_cartpole_cpu_{time.strftime('%Y%m%d_%H%M%S')}.pth")


if __name__ == '__main__':
    main()