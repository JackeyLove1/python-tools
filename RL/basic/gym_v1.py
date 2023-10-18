import gym
import numpy as np
import torch
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
state, infos = env.reset()
print(f"state:{state}, infos:{infos}")
for _ in range(2):
    action = env.action_space.sample()
    print("action:", action)
    observation, reward, terminated, truncated, info = env.step(action)
    observation = np.array(observation)
    observation = torch.from_numpy(observation)
    print(f"observation:{observation}, reward:{reward}, terminated:{terminated}, info:{info}")
    if terminated or truncated:
        observation, info = env.reset()
env.close()