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
from ddpg_agent import Agent

# Function for plotting scores
def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

# Defining gym environment
env = gym.make('Pendulum-v0') # 'LunarLanderContinuous-v2', 'MountainCarContinuous-v0',
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

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(10): #1000 episodes
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        env.render()
    score_history.append(score)

    if i % 25 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)        