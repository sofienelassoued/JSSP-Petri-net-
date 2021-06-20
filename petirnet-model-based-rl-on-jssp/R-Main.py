import gym

import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

#%% load , make , chek custom petrinet env 
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'petri-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
import gym_petrinet

# Create environment
env = gym.make("petri-v0")
check_env(env, warn=True)
#env = make_vec_env(lambda: env, n_envs=1)
#%% Train model 

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=int(1e6))
# Save the agent
model.save("Trained models\dqn_model3_dot3.zip")
#del model  # delete trained model to demonstrate loading

#%% Test model 

episodes=100
model = DQN.load("Trained models\dqn_model3_dot3.zip")

for ep in range (episodes):
    
    dones =False
    ep_reward=0
    obs = env.reset()
    
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action,False,ep) 
        ep_reward+=rewards

    print(ep_reward)
    
env.render()
    
#%%Ntesting 
Nxmarking,Timefeatures,fired,inprocess=env.fire_transition (0)
print()