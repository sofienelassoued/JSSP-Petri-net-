import gym
import gym_petrinet


env = gym.make("petri-v0")

done=False
obs=env.reset()


#while done==False:


    #action=env.action_space.sample()
   # obs,reward,done,info=env.step(action)
   # print( (obs),reward,done,info)



#%% 


obs,reward,done,info=env.step(1)

for i in env.Places_obj:
    print(i.enabled)

print(env.Places_obj[2])

