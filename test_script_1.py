import gym, gym_donkeycar

from stable_baselines3 import PPO

env = gym.make("donkey-mountain-track-v0")

model = PPO("CnnPolicy", env, verbose=1)

obs = env.reset()

model.learn(10000)