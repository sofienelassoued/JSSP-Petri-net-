import gym
import gym_petrinet
from stable_baselines3 import DQN


#%%

# Create environment
env = gym.make("petri-v0")

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
# Save the agent
model.save("dqn_Petri")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("dqn_Petri")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action,True,i)

env.render()
    
#%%


env = gym.make("petri-v0")
#%%

for i in range (100):
    action=env.action_space.sample()
    env.step(action,True,i)


env.render()

#%% print