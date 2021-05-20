import gym
import gym_petrinet
from stable_baselines3 import DQN



env = gym.make("petri-v0")

for i in range (20):
    action=env.action_space.sample()
    env.step(action)
    

env.render()


#%%
