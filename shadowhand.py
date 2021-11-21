"""ShadowHand Environment Wrappers."""
import os
import const
from PIL import Image  # Will need to make sure PIL is installed
import numpy as np
import gym
from gym import spaces
import mss
import cv2

# function to define the PPO RL algotithm
def proximal_pol_opti():
    return action

# Initialising the environment
env = gym.make("HandManipulateBlock-v0")
done, observation = False, env.reset()
rewards=[]
done_cntr=0
action = env.action_space.sample()


while True:
    env.render()
    observation, reward, done, info = env.step(action)

    # We give the reward at timestamp t as the difference between the 
    # rotation angles between the desired and the current/achieved otientation 
    # before and after the transition, respectively.
    reward = observation["desired_goal"] - observation["achieved_goal"]

    # additional reward of +5 is given if the goal is acchieved 
    # within a tolerance of 0.4 rad
    if info["is_success"]==1.0 and observation["achieved_goal"]<0.4:
        reward+=5


    # add the code for the condition when the object drops from the hand
    # and give it a reward of -20
    
    # if object dropped:
    #     reward-=20

    rewards.append(reward)          #We append the reward value to the rewards 
                                    #for future reference
    
    action = proximal_pol_opti()    #calculate the action using the PPO RL algorithm

    # print(reward)
    if done:
        done_cntr+=1
        print("done")
        env.reset()
        if done_cntr==const.LIMIT_STEPS:
            break

print(rewards)
print(np.shape(rewards))