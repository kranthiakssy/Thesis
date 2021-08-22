# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:34:09 2021

@author: kranthi
"""
# Importing packages
#import os
from math import ceil
from os import stat
import time 

from numpy.random import rand
#import torch as T
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
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


agent.load_models()
# np.random.seed(0)
# random.seed(1)

# Tensorboard Initialization for visualization
tb = SummaryWriter()

# adding models graphs to tensorboard
# tb.add_graph(agent.actor,T.tensor([0,0,0,0]))

# Iteration parameters
episodes = 1500 # no of episodes
update_tb = 300 # Update episode no for tensorboard
ns = 300 # no of steps to run in each episode    
t = np.linspace(0,ns/100,ns+1) # define time points
dt = t[1]-t[0] # time step duration

# initial controller parameters
def initialize(episode):
        global tune_param, pv, sp, sp_data, e, delta_e, ie, dpv, statevec, smoothing
        # sd = ceil(episode/10)
        # random.seed(sd)
        tune_param = [0.1, 1, 0] # [random.uniform(0,4),
                        # random.uniform(1,10),
                        # random.uniform(0.01,1)]
        pv = [0] #[random.uniform(0,100)]
        sp = 30 #random.uniform(0,100)
        # print(tune_param,pv,sp)
        sp_data = [sp] # setpoint track
        e = [0] # error list
        delta_e = [0] # change in error list
        ie = [0] # integral error list
        dpv = [0] # change in pv list
        statevec = np.array([0,0,0,0]) # e, delta_e, ie, dpv
        smoothing = 1 # smoothing factor


# action space function
def statevectorfunc(pv, sp, dt):
        e.append(sp-pv[-1])
        delta_e.append(e[-2]-e[-1])
        ie.append(ie[-1]+e[-1]*dt)
        dpv.append((pv[-1]-pv[-2])/dt)
        out = np.array([e[-1],delta_e[-1],ie[-1],dpv[-1]])
        smoothing = abs(e[-1]/sp)
        return out, smoothing

str_time = time.time()

score_history = []
best_score = -1e10
cnt = 0
# running for specific no of episodes
for episode in range(1, episodes+1):
    state = env.reset()
    initialize(episode)
    done = False
    score = 0

    # add data to tensorboard
    if episode % update_tb == 0:
        tb.add_scalars("Tune_Param/episode-"+str(episode),{"Kp":tune_param[0],
                                                            "Ti":tune_param[1],
                                                            "Td":tune_param[2]},0)
        tb.add_scalars("Process_Value/episode-"+str(episode),{"PV":pv[0],
                                                            "SP":sp},0)
        tb.add_scalar("Process_Input/episode-"+str(episode),0,0)
        tb.add_scalars("StateVector/episode-"+str(episode),{"Error":statevec[0],
                                                            "dE":statevec[1],
                                                            "IE":statevec[2],
                                                            "dpv":statevec[3]},0)
        tb.add_scalars("Actor_Actions/episode-"+str(episode),{"dKp":0,
                                                            "dTi":0,
                                                            "dTd":0},0)      
    for k in range(0,ns):
        #env.render()
        action = agent.choose_action(statevec)
        tune_param += action * smoothing
        tune_param = np.maximum([0,1,0],tune_param)
        new_state, reward, done, info, cout, csat  = env.step(tune_param, statevec, dt, pv[-1])
        pv.append(new_state)
        sp_data.append(sp)
        new_statevec, smoothing = statevectorfunc(pv, sp, dt)
        if csat == True:
            ie[-1] = ie[-2] # anti-reset windup
        agent.remember(statevec, action, reward, new_statevec, int(done))
        agent.learn()
        score += reward
        statevec = new_statevec

        # add data to tensorboard
        if episode % update_tb == 0:
            tb.add_scalars("Tune_Param/episode-"+str(episode),{"Kp":tune_param[0],
                                                                "Ti":tune_param[1],
                                                                "Td":tune_param[2]*0},k+1)
            tb.add_scalars("Process_Value/episode-"+str(episode),{"PV":new_state,
                                                                "SP":sp},k+1)
            tb.add_scalar("Process_Input/episode-"+str(episode),cout,k+1)
            tb.add_scalars("StateVector/episode-"+str(episode),{"Error":statevec[0],
                                                                "dE":statevec[1],
                                                                "IE":statevec[2],
                                                                "dpv":statevec[3]},k+1)
            tb.add_scalars("Actor_Actions/episode-"+str(episode),{"dKp":action[0],
                                                                "dTi":action[1],
                                                                "dTd":action[2]},k+1)
    
    # Calculation of closed loop response parameters
    # Calculate ITAE (Integral of time weighted absolute error)
    itae = dt * np.dot(t.T,abs(np.subtract(sp_data, pv)))
    # Calculate maximum overshoot
    mos = np.max(pv) - sp
    # Calculation of rise time
    try:
        rt = t[np.array(pv) >= (sp * 0.9)][0]
    except:
        rt = 0
    # Calculation of steady state error
    ess = statevec[0]

    # Averaging the reward score over last 100 episodes
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    print('episode ', episode, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % avg_score)
    
    # add data to tensorboard
    tb.add_scalar("Reward",avg_score,episode)
    tb.add_scalar("End_State",new_state,episode)
    tb.add_scalar("ITAE",itae,episode)
    tb.add_scalar("OverShoot",mos,episode)
    tb.add_scalar("RiseTime",rt,episode)
    tb.add_scalar("SteadyStateError",ess,episode)

    # Adding Network parameters to Tensorboard
    for name, weight in agent.actor.named_parameters():
        tb.add_histogram(name, weight, episode)
        tb.add_histogram(f'{name}.grad', weight.grad, episode)

    # Saving Models whent the average score improves atleast 5 times
    # if (score > best_score) and (episode > 8): # bypassing intital episodes
        # cnt += 1
        # best_score = score
        # if cnt >= 5:
            # cnt = 0
            # agent.save_models()
            # print("Time taken upto episode {} in seconds: {}"\
                    # .format(episode,round(time.time()-str_time)))

    # Saving Models
    if episode % update_tb == 0:
        agent.save_models()
        print("Time taken upto episode {} in seconds: {}"\
            .format(episode,round(time.time()-str_time)))

print("Total time taken in seconds: ", round(time.time()-str_time))

tb.close()

