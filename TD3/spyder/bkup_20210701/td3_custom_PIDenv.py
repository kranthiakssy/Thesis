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
#from ddpg_module import CriticNetwork, ActorNetwork, OUActionNoise, ReplayBuffer
from td3_agent import Agent
from Custom_PIDEnv import PIDEnv

# Function for plotting scores
def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.figure()
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

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
agent = Agent(alpha=0.001, beta=0.001, input_dims=[state_dim], tau=0.005, env=env,
              batch_size=100,  layer1_size=256, layer2_size=256, n_actions=action_dim,
              action_bound=action_bound)

#agent.load_models()
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
for i in range(21): #1000 episodes
    obs = env.reset()
    done = False
    score = 0
    k = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info, pv, _, _, _ = env.step(act, delta_t, Kp, taup, k)
        agent.remember(obs, act, reward, new_state, int(done))
        #print("Action:{0}, State:{1}, Reward:{2}".format(act, new_state, reward))
        agent.learn()
        score += reward
        k += 1
        obs = new_state
        #env.render()
    score_history.append(score)
    end_state.append(new_state)

    if i % 25 == 0:
        agent.save_models()
        
    if i % 25 == 0:
        plt.figure()
        plt.plot(pv)
        plt.savefig("tmp/graphs/PV"+str(i)+".png")


    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
    print("State at the end of episode: {}".format(obs))
        
filename1 = 'tmp/graphs/Closed_Loop_system_score.png'
filename2 = 'tmp/graphs/Closed_Loop_system_state.png'
plotLearning(score_history, filename1, window=100)   
plotLearning(end_state, filename2, window=100) 

    
      