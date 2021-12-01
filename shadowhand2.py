# DISCLAMER: The hyperparameters control how fast 
# and convergent is the learning of parameters
# Changing it above your cpu capacity may make 
# the process to crash.😕

# ⏹⏹⏹ Also make sure to increment the model 
# number in the saving process so that you dont 
# overwrite the previously saved model. 

"""ShadowHand Environment Wrappers."""
import os

from numpy.core.shape_base import block
import numpy as np

import gym
from gym import spaces
from ppo import PPO
from matplotlib import pyplot as plt

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

# Initialising the environment
env = gym.make("HandManipulateBlock-v0", reward_type="dense")
env.rotation_threshold = 0.4
env.distance_threshold = 0.01
env.relative_control = True

done, observation = False, env.reset()
rewards = []
done_cntr = 0
action = env.action_space.sample()
observation, reward, done, info = env.step(action)

obs_inputs = np.array(np.shape(observation["observation"]))[0]  # 61
num_outputs = np.array(np.shape(action))[0]  # 20
goal_size = np.array(np.shape(observation["desired_goal"]))[0]
num_inputs = obs_inputs + goal_size

"""
Hyperparameters used for PPO by OpenAI's implementation
discount factor γ                   0.998
Generalized Advantage Estimation λ  0.95
entropy regularization coefficient  0.01
PPO clipping parameter              0.2
optimizer Adam [28]
learning rate                       3e-4
batch size (per GPU)                80k chunks x 10 transitions = 800k transitions
minibatch size (per GPU)            25.6k transitions
number of minibatches per step      60
network architecture                dense layer with ReLU + LSTM
size of dense hidden layer          1024
LSTM size                           512
"""

# HYPERPARAMETERS
# Model
lstm_nh = 64  # Hidden layer size in LSTM
dense_na = 128  # Size of dense hidden layer
action_dist_size = 20  # Action distribution size
value_output_size = 1  # Single output

hyperparameters = {
    "timesteps_per_batch": 500,
    "max_timesteps_per_episode": 200,
    "gamma": 0.95,
    "n_updates_per_iteration": 10,
    "lr": 3e-4,
    "clip": 0.2,
    "render": False,
    "render_every_i": 1,
}

def test_env(rndr=True):
    print("Running a test")
    obs = env.reset()
    if rndr:
        env.render()
    done = False
    total_reward = 0
    while not done:
        obs = np.concatenate(
                    [
                        np.reshape(obs["observation"], (61)),
                        np.reshape(obs["desired_goal"], (7)),
                    ]
                )
        obs = torch.tensor(obs).reshape((1, 1, num_inputs)).float()        
        action, _ = model.get_action(obs)
        obs, reward, done, _ = env.step(action.reshape((20)))
        if rndr:
            env.render()
        total_reward += reward
    return total_reward

# Actor and Critic models
class PPO_model(nn.Module):
    def __init__(self, nX, nY):
        super(PPO_model, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(nX, dense_na),
            nn.ReLU(),
            nn.LSTM(dense_na, lstm_nh, batch_first=True),
        )
        self.Lin = nn.Linear(lstm_nh, nY)

    def forward(self, obs):
        # obs = F.normalize(obs)
        obs, _ = self.stack(obs)
        out = self.Lin(obs)
        return out


load_model = False  # load model from the desired pathfile
extra_train = False  # Toggle for more training

model = PPO(policy_class=PPO_model, env=env, load_model=load_model, **hyperparameters)

if load_model and not extra_train:
    [test_env() for _ in range(15)]
    exit()
else:
    model.learn(total_timesteps=1250)

print(model.actor)