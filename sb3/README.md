

```bash
conda create --name drl-til python=3.10
conda activate drl-til

pip install --upgrade pip
pip install flask
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