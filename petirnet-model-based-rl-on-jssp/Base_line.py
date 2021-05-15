import gym
import gym_petrinet


env = gym.make("petri-v0")

done=False
obs=env.reset()

while done==False:


    action=env.action_space.sample()
    obs,reward,done,info=env.step(action)
    print(tuple (obs),reward,done)



#%% 


















