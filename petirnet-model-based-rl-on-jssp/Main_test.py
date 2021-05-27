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

#%% Test saved model 

model = DQN.load("Trained models\dqn_Petrinet_1.zip")

obs = env.reset()
for i in range(50):
    action, _states = model.predict(obs)
    #print(action)
    obs, rewards, dones, info = env.step(action,True)
    
env.render()

#%%Ntesting 

