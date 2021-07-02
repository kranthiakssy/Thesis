# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:34:09 2021

@author: kranthi
"""
# Importing packages
#import os
#import torch as T
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt

# Importing local functions
#from td3_module import CriticNetwork, ActorNetwork, OUActionNoise, ReplayBuffer
from td3_agent import Agent
from Custom_PIDEnv import PIDEnv


# Defining gym environment
env = PIDEnv()
#gym.make('Pendulum-v0') # 'LunarLanderContinuous-v2', 'MountainCarContinuous-v0',
                                # 'Pendulum-v0'

# Define all the state and action dimensions, and the bound of the action
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
print("State Dim: {0}\n Action Dim: {1}\n Action Bound: {2}"\
      .format(state_dim, action_dim, action_bound))

# Agent creaation    
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[state_dim], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=action_dim,
              action_bound=action_bound)

agent.load_models()
np.random.seed(0)

# specify number of steps
ns = 300
# define time points
t = np.linspace(0,ns/10,ns+1)
delta_t = t[1]-t[0]
# process model
Kp = 2.0
taup = 5.0
    
score_history = []
end_state = []
for i in range(1): #1000 episodes
    obs = env.reset()
    done = False
    score = 0
    for k in range(0,ns):
        act = agent.test_action(obs)
        new_state, reward, done, info, pv, pterm, iterm, dterm = env.step(act, delta_t, Kp, taup, k)
        #agent.remember(obs, act, reward, new_state, int(done))
        #print("Action:{0}, State:{1}, Reward:{2}".format(act, new_state, reward))
        #agent.learn()
        score += reward
        obs = new_state
        #env.render()
    #score_history.append(score)
    #end_state.append(new_state)

    #if i % 25 == 0:
    #    agent.save_models()
        
    #if i % 25 == 0:
    plt.figure()
    plt.plot(pv)
    plt.title("Process Value")
    plt.savefig("tmp/graphs/Test_PV"+str(i)+".png")

    plt.figure()
    plt.plot(pterm)
    plt.title("Proportional Term")
    plt.savefig("tmp/graphs/Test_Pterm"+str(i)+".png")

    plt.figure()
    plt.plot(iterm)
    plt.title("Integral Term")
    plt.savefig("tmp/graphs/Test_Iterm"+str(i)+".png")

    plt.figure()
    plt.plot(dterm)
    plt.title("Derivative Term")
    plt.savefig("tmp/graphs/Test_Dterm"+str(i)+".png")


    print('episode ', i, 'score %.2f' % score)
    print("State at the end of episode: {}".format(obs))

t = np.linspace(0,len(pv),len(pv))
t[1:] = 48
plt.figure()
plt.plot(t)
plt.plot(pv)
plt.title("Process Value")
plt.savefig("tmp/graphs/Test_PV1"+str(i)+".png")