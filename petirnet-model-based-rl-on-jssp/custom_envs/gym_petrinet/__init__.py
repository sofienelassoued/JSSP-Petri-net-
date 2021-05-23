import gym
from gym.envs.registration import register


register(
    id='petri-v0',
    entry_point='gym_petrinet.envs:PetriEnv',
)