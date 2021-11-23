"""ShadowHand Environment Wrappers."""
import os
import const
from PIL import Image  # Will need to make sure PIL is installed
import numpy as np
import gym
from gym import spaces
import mss
import cv2
from matplotlib import pyplot as plt

#Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# Initialising the environment
env = gym.make("HandManipulateBlock-v0")
done, observation = False, env.reset()
rewards=[]
done_cntr=0
action = env.action_space.sample()
observation, reward, done, info = env.step(action)

# print("env.observation_space.shape :",np.array(np.shape(observation["observation"]))[0])
# print("env.observation_space.shape :",np.array(np.shape(action))[0])
num_inputs  = np.array(np.shape(observation["observation"]))[0] #61
num_outputs = np.array(np.shape(action))[0]  #20


# print("\nactionsize :", np.shape(action),"\naction :",action)

"""
Hyperparameters used for PPO
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

#HYPERPARAMETERS
#Model
lstm_nh = 512             #Hidden layer size in LSTM
dense_na = 1024           #Size of dense hidden layer
policy_input_size = 61    #Policy network input size
value_input_size = 61     #Policy network input size
lr = 3e-4                 #Adam optimizer learning rate
action_dist_size = 20     #Action distribution size
value_output_size = 1     #Single output

#PPO
discount_factor = 0.998   #Discount factor Gamma
gae_gamma = 0.95          #Generalized Advantage Estimation λ
ppo_clipping_param = 0.2  #PPO clipping parameter
num_steps        = 100
mini_batch_size  = 5
ppo_epochs       = 40
threshold_reward = -200 

#Policy model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Policy_Value_NN(nn.Module):
    def __init__(self, std=0.0):
        super(Policy_Value_NN, self).__init__()
        self.policy_stack = nn.Sequential(
            nn.Linear(policy_input_size, dense_na),
            nn.ReLU(),
            # nn.LSTM(dense_na, lstm_nh),
            nn.Linear(dense_na, action_dist_size)
            )

        self.value_stack = nn.Sequential(
            nn.Linear(value_input_size, dense_na),
            nn.ReLU(),
            # nn.LSTM(dense_na, lstm_nh),
            nn.Linear(dense_na, 1)
        )

        self.log_std = nn.Parameter(torch.ones(1, 1, num_outputs) * std)
        self.apply(init_weights)


    def forward(self, obs):
        obs = F.normalize(obs, dim = 2)
        value = self.value_stack(obs)
        actions = self.policy_stack(obs)
        # print(actions.cpu().detach())
        # exit()
        std  = self.log_std.exp().expand_as(actions.cpu().detach())
        # print("actions :",actions)
        dist = Normal(actions, std)
        return dist, value

# model = Policy_Value_NN()
# print(model)
def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=True):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor([state['observation']]).unsqueeze(0).to(device)
        dist, _ = model.forward(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy().ravel())
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

# gae algo for PPO
def compute_gae(next_value, rewards, masks, values, discount_factor=0.99, gae_gamma=0.95):
    """next_value, rewards, masks, values, discount_factor=0.998, gae_gamma=0.95"""
    # next_value = next_value.detach().numpy()
    values = values + [next_value]
    # print('you gay',type(values[0]))
    gae = 0
    returns = []
    # print("\n\n (rewards): ",rewards[0].detach().numpy())
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + discount_factor*values[step+1]*masks[step] - values[step]
        gae = delta + discount_factor*gae_gamma*masks[step]*gae
        returns.insert(0, gae+values[step])
    return returns

# PPO algo

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        advantage = torch.reshape(advantage, (num_steps, 1, 1))
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
                

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, ppo_clipping_param=0.2):
    """ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, ppo_clipping_param=0.2"""
    print("HOLAAAAAAA")
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model.forward(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage                                                                    #PPO equation Lclip part one
            surr2 = torch.clamp(ratio, 1.0 - ppo_clipping_param, 1.0 + ppo_clipping_param) * advantage   #PPO equation Lclip part two

            policy_loss  = - torch.min(surr1, surr2).mean()
            value_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * value_loss + policy_loss - 0.001 * entropy
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

model = Policy_Value_NN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 15000
frame_idx  = 0
test_rewards = []


state = env.reset()
# print(state)
# exit()
state = state['observation']
early_stop = False
i=0
while frame_idx < max_frames and not early_stop:

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0
    # count=0
    for i in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        state = torch.reshape(state, (1, 1, num_inputs))
        dist, value = model.forward(state)
        action = dist.sample().to(device)
        next_state, reward, done, _ = env.step(action.cpu().detach().numpy().ravel())
        # print("\n\ndone",done,i)
        next_state = next_state['observation']
        # next_state, reward, done, _ = env.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value[0][0])
        rewards.append(torch.FloatTensor([reward]).to(device))
        masks.append(torch.FloatTensor([1 - done]).to(device))
        
        states.append(state)
        actions.append(action)
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            test_reward = np.mean([test_env() for _ in range(10)])
            test_rewards.append(test_reward)
            plot(frame_idx, test_rewards)
            if test_reward > threshold_reward: early_stop = True
            

    next_state = torch.FloatTensor([next_state]).to(device)
    next_state = torch.reshape(next_state, (1, 1, num_inputs))
    _, next_value = model.forward(next_state)
    # print("last mask shape : ", np.shape(masks[-1]))

    returns = compute_gae(next_value, rewards, masks, values)
    # print(i)
    i+=1
    # print("Returns shape", np.shape(returns))
    # print("Returns sample: ", returns[10][0][0])
    # print("\n")
    # if np.shape(returns)[1] == 0:
    #     continue
    returns   = torch.cat(returns).detach().to(device)
    log_probs = torch.cat(log_probs).detach().to(device)
    values    = torch.cat(values).detach().to(device)
    states    = torch.cat(states).to(device)
    actions   = torch.cat(actions).to(device)
    # print(f"returns shape : {returns.shape}\nvalues shape : {values.shape}\nrewards shape : {np.shape(rewards)}")
    # print(returns)
    # print("\n\n")
    advantage = returns[:, 0, 0] - values[0]

    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
    

# while True:
#     env.render()
#     observation, reward, done, info = env.step(action)

#     # We give the reward at timestamp t as the difference between the 
#     # rotation angles between the desired and the current/achieved otientation 
#     # before and after the transition, respectively.
#     obs = torch.Tensor(observation['observation'])
#     obs = torch.reshape(obs, (1, 1, 61))
#     # break
#     reward = observation["desired_goal"] - observation["achieved_goal"]
#     # print("\nobservations: ",obs)
#     # additional reward of +5 is given if the goal is achieved 
#     # within a tolerance of 0.4 rad
#     actions, value  = model.forward(obs)

#     # print("\n\nactions by NN :", actions.sample().detach().numpy().ravel())
#     # print("\n\nactions by NN :", actions.detach().numpy().ravel())
#     # action = actions.detach().numpy().ravel()
#     action = actions.sample().detach().numpy().ravel()
#     # print(action)
#     # exit()

#     if info["is_success"]==1.0 and observation["achieved_goal"]<0.4: # if np.array is obserbvation["achieved_goal"] then no need to change anything
#         reward+=5

#     # add the code for the condition when the object drops from the hand
#     # and give it a reward of -20
    
#     # if object dropped:
#     #     reward-=20

#     rewards.append(reward)          #We append the reward value to the rewards 
#                                     #for future reference

#     # print(reward)
#     if done:
#         done_cntr+=1
#         # print("done")
#         env.reset()
#         if done_cntr==const.LIMIT_STEPS:
#             break

# print(rewards)
# print(np.shape(rewards))