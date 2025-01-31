

```
conda create --name dlr-til python=3.10
conda activate dlr-til

pip install --upgrade pip
pip install flask
pip install gymnasium[mujoco]
pip install 'stable-baselines3[extra]'
```

```
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

# Check learning status
tensorboard --logdir logs



```