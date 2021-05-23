import gym

import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
#%%
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

#%%

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2))
# Save the agent
model.save("dqn_Petri")
del model  # delete trained model to demonstrate loading

#%%
# Load the trained agent
model = DQN.load("dqn_Petri")
# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(30):
    action, _states = model.predict(obs)
    #print(action)
    obs, rewards, dones, info = env.step(action)
    
env.render()
    

#%%Ntesting 
