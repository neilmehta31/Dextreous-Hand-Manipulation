"""ShadowHand Environment Wrappers."""
import os

import numpy as np
import gym
from gym import spaces

env = gym.make("HandManipulateBlock-v0")
done, observation = False, env.reset()

while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()
    # trial comment
