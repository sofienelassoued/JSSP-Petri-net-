# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:42:45 2020

@author: Sharafath
"""
import matplotlib.pyplot as plt
from jsspenv_gym import Jssp_env
import torch
# import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


#%%

# Instantiate the env
env = Jssp_env()
env = DummyVecEnv([lambda:env])
env = VecNormalize(env, norm_reward=False, norm_obs=False)

pk = dict(activation_fn = torch.nn.ReLU, net_arch=[9, dict(vf=[9, 9], pi=[8, 4])])
 
model = PPO('MlpPolicy', env, policy_kwargs=pk,  verbose=0, learning_rate=1e-4, \
            seed=0, n_steps=16, n_epochs=2)
    
model.learn(12000)

plt.figure()
plt.title('Makespan over episodes')
plt.plot(env.get_attr('running_counter_list')[0])
plt.show()

plt.figure()
plt.title('Rewards collected over episodes')
plt.plot(env.get_attr('ep_rew_list')[0])
plt.show()

plt.figure()
plt.title('Output over episodes')
plt.plot(env.get_attr('output_list')[0])
plt.show()


#%%
# Deterministic Action for environment chekcing
# env.reset()
# done = False
# action = np.array([1])
# while not done:
#     obs, _, done, _ = env.step(action)
#     try:
#         action = np.where(obs[0][:2]>0)[0][0]
#         action = np.array([action])
#     except:
#         print('Action not possible')
        