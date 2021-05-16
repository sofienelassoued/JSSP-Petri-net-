import gym
import gym_petrinet
from stable_baselines3 import DQN


env = gym.make("petri-v0")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

#model.save("dqn_petri")
#del model # remove to demonstrate saving and loading
#model = DQN.load("dqn_petri")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
    #env.render()
    if done:
      obs = env.reset()

#%%




