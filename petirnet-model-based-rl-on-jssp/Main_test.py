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


episodes=3
model = DQN.load("Trained models\dqn_Petrinet_1.zip")



for ep in range (episodes):
    
    dones =False
    ep_reward=0
    obs = env.reset()
    
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action,True,ep) 
        ep_reward+=rewards

    print(ep_reward)
    
env.render()
    


#%%Ntesting 

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#%%
#env.render(continues=False)