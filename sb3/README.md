

```bash
conda create --name drl-til python=3.10
conda activate drl-til

pip install --upgrade pip
pip install flask
pip install tensorboard
pip install gymnasium[mujoco]
pip install 'stable-baselines3[extra]'
```

```
# check version

```bash
gymnasium==0.29.1
mujoco==2.3.3
mujoco-py==2.1.2.14
stable_baselines3==2.5.0
```

```bash
# For Humanoid-v4 SAC works well
python3 1_basic_usage.py Humanoid-v4 SAC -t 
python3 1_basic_usage.py Humanoid-v4 TD3 -t 
python3 1_basic_usage.py Humanoid-v4 A2C -t

# After achieving at least 4000 ep_rew_mean
python3 1_basic_usage.py Humanoid-v4 SAC -s ./models/SAC_350000.zip

# For Pendulum-v1 TD3 works well
python3 1_basic_usage.py Pendulum-v1 TD3 -t 
python3 1_basic_usage.py Pendulum-v1 PPO -t 
python3 1_basic_usage.py Pendulum-v1 DDPG -t 

# After ep_rew_mean converges
python3 1_basic_usage.py Pendulum-v1 TD3 -s ./models/TD3_75000.zip

# Check learning status
tensorboard --logdir logs
```

## rl-zoo


```bash
# Installation - check your sim env and find best release from 
# https://github.com/DLR-RM/rl-baselines3-zoo/releases

apt-get install swig cmake ffmpeg
# full dependencies
pip install -r requirements.txt
# minimal dependencies
pip install -e .
```

```bash
# Train
python -m rl_zoo3.train --algo ppo --env CartPole-v1 --eval-freq 10000 --save-freq 50000 -f zoo_logs

# Play
python -m rl_zoo3.enjoy --algo ppo --env CartPole-v1 --folder zoo_logs/ -n 5000
```

## rl-zoo (Hyperparameter Optimization)

```bash 
# this will takes long time 
python -m rl_zoo3.train --algo ppo --env MountainCar-v0 --eval-freq 10000 --save-freq 50000 -f zoo_logs

# Hyperparameter Optimization
python -m rl_zoo3.train --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler random --pruner median -f zoo_logs

# rl-zoo example output
Best trial:
Value:  -104.2
Params: 
    batch_size: 16
    n_steps: 32
    gamma: 0.99
    learning_rate: 0.006960048474522834
    ent_coef: 3.5797062373970776e-08
    clip_range: 0.2
    n_epochs: 1
    gae_lambda: 0.8
    max_grad_norm: 0.9
    vf_coef: 0.4873662205806325
    net_arch: medium
    activation_fn: relu

# modify ppo.yml and then train again or hard code into your python file
python -m rl_zoo3.train --algo ppo --env MountainCar-v0 --eval-freq 10000 --save-freq 50000 -f zoo_logs
# or
python3 2_rl_zoo.py MountainCar-v0 PPO -t 

```