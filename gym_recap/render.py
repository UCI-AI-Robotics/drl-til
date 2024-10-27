import gymnasium as gym
import numpy as np

env_train = gym.make('FrozenLake-v1')

# initialize Q table
Q = np.zeros([env_train.observation_space.n, env_train.action_space.n])
print(Q)

# parameter
lr = 0.8
gamma = 0.95
num_episodes = 2000

# learning
for i in range(num_episodes):
    state = env_train.reset()[0]
    done = False

    while not done:
        # epsilon-greedy policy
        if np.random.uniform(0, 1) < 0.5:
            action = env_train.action_space.sample()  
        else:
            action = np.argmax(Q[state,:])  

        next_state, reward, done, _, _ = env_train.step(action)

        Q[state, action] = (1 - lr) * Q[state, action] + \
            lr * (reward + gamma * np.max(Q[next_state, :]))

        state = next_state

print(Q)
env_train.close()

# test
env_test = gym.make('FrozenLake-v1', render_mode="human")
state = env_test.reset()[0]
done = False
while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, done, _, _ = env_test.step(action)
    state = next_state
    env_test.render()

env_test.close()