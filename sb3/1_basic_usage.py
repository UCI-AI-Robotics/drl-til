# referenced from https://github.com/johnnycode8/gym_solutions

import gymnasium as gym
import os
import argparse

import stable_baselines3

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def get_model_class(algo_name):
    """Retrieve the SB3 algorithm class dynamically."""
    try:
        return getattr(stable_baselines3, algo_name)
    except AttributeError:
        raise ValueError(f"Invalid algorithm: {algo_name}. Available options: A2C, DDPG, DQN, PPO, SAC, TD3")

def train(env, sb3_class, algo_name, env_name):

    os.makedirs(f"{model_dir}/{env_name}/{algo_name}", exist_ok=True)
    os.makedirs(f"{log_dir}/{env_name}/{algo_name}", exist_ok=True)

    # PPO
    # model = sb3_class(
    #     "MlpPolicy", 
    #     env, 
    #     verbose=1,
    #     gamma=0.9,
    #     learning_rate=1e-3,
    #     use_sde=True,
    #     sde_sample_freq=4,
    #     device="cuda", 
    #     tensorboard_log=f"{log_dir}/{env_name}/{algo_name}"
    # )
    # TIMESTEPS = 100000

    # SAC
    model = sb3_class(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-3,
        device="cuda", 
        tensorboard_log=f"{log_dir}/{env_name}/{algo_name}"
    )
    TIMESTEPS = 20000

    # TIMESTEPS = 25000
    iters = 0

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{model_dir}/{env_name}/{algo_name}/{algo_name}_{TIMESTEPS*iters}")

    # while True:
    #     iters += 1

    #     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    #     model.save(f"{model_dir}/{env_name}/{algo_name}/{algo_name}_{TIMESTEPS*iters}")

def test(env, sb3_class, path_to_model):

    model = sb3_class.load(path_to_model, env=env)

    obs, _ = env.reset()
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        # print(f"obs: {obs}")

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break

if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. A2C, DDPG, DQN, PPO, SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    try:
        sb3_class = get_model_class(args.sb3_algo)
    except ValueError as e:
        print(e)
        exit(1)

    if args.train:
        gymenv = gym.make(args.gymenv, render_mode=None)
        train(gymenv, sb3_class, args.sb3_algo, args.gymenv)

    if(args.test):
        if os.path.isfile(args.test):
            gymenv = gym.make(args.gymenv, render_mode='human')
            test(gymenv, sb3_class, args.test)
        else:
            print(f'{args.test} not found.')

# python3 1_basic_usage.py Pendulum-v1 SAC -t
# python3 1_basic_usage.py Pendulum-v1 DDPG -t
# python3 1_basic_usage.py Pendulum-v1 PPO -t

# python3 1_basic_usage.py Pendulum-v1 PPO -s ./models/Pendulum-v1/PPO/PPO_0.zip
# python3 1_basic_usage.py Pendulum-v1 SAC -s ./models/Pendulum-v1/SAC/SAC_200000.zip

# tensorboard --logdir Pendulum-v1