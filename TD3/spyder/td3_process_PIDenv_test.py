# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:34:09 2021

@author: kranthi
"""
# Importing packages
#import os
from os import stat

from numpy.core.shape_base import block
import torch as T
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

# Importing local functions
from ProcessModel import ProcessModel
from td3_agent import Agent
from Process_PIDEnv import PIDEnv

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
np.random.seed(0)

# Iteration parameters
episodes = 1 # no of episodes
update_tb = 1 # Update episode no for tensorboard
ns = 300 # no of steps to run in each episode    
t = np.linspace(0,ns/10,ns+1) # define time points
dt = t[1]-t[0] # time step duration

# initial controller parameters
def initialize():
        global tune_param, pv, sp, sp_data, e, delta_e, ie, dpv, statevec
        tune_param = [0.1, 1.5, 0.1]  # Kp, Ti, Td respectively
        pv = [0] # process value list
        sp = 30 # setpoint
        sp_data = [sp] # setpoint track
        e = [0] # error list
        delta_e = [0] # change in error list
        ie = [0] # integral error list
        dpv = [0] # change in pv list
        statevec = np.array([0,0,0,0]) # e, delta_e, ie, dpv


# action space function
def statevectorfunc(pv, sp, dt):
        e.append(sp-pv[-1])
        delta_e.append(e[-2]-e[-1])
        ie.append(ie[-1]+e[-1]*dt)
        dpv.append((pv[-1]-pv[-2])/dt)
        out = np.array([e[-1],delta_e[-1],ie[-1],dpv[-1]])
        return out

# State Space parameters of Process Model
prm = ProcessModel(1,[1,1.6,1],0,dt) # num, dnum, delay, time_step

param_Kp = [0]
param_Ti = [0]
param_Td = [0]
param_e = [0]
param_delta_e = [0]
param_ie = [0]
param_dpv = [0]
param_cout = [0]
# running for specific no of episodes
for episode in range(1, episodes+1):
    state = env.reset()
    initialize()
    done = False
    score = 0 
    X = prm[6]
    U = [0]
    for k in range(0,ns):
        #env.render()
        action = agent.test_action(statevec)
        tune_param += action
        tune_param = np.maximum([0,1,0],tune_param)
        new_state, reward, done, info, cout, csat, X, U  = env.step(tune_param, statevec, dt, prm, X, U)
        pv.append(new_state)
        sp_data.append(sp)
        param_Kp.append(tune_param[0])
        param_Ti.append(tune_param[1])
        param_Td.append(tune_param[2])
        param_e.append(statevec[0])
        param_delta_e.append(statevec[1])
        param_ie.append(statevec[2])
        param_dpv.append(statevec[3])
        param_cout.append(cout)
        # if k >= 150 and k < 200:
            # sp = 50
        # elif k >= 200:
            # sp = 30
        sp = 5*np.sin(2*np.pi*k/100)+30
        new_statevec = statevectorfunc(pv, sp, dt)
        score += reward
        statevec = new_statevec

    
    # Calculation of closed loop response parameters
    # Calculate ITAE (Integral of time weighted absolute error)
    itae = dt * np.dot(t.T,abs(np.subtract(sp_data, pv)))
    # Calculate maximum overshoot
    mos = np.max(pv) - sp
    # Calculation of rise time
    try:
        rt = t[np.array(pv) >= (pv[0]+abs(pv[0]-sp) * 0.9)][0]
    except:
        rt = 0
    # Calculation of steady state error
    ess = statevec[0]

    print("Score: {}, ITAE: {}, OverShoot: {}, RiseTime: {}, Ess: {}"\
        .format(score, itae, mos, rt, ess))

    tm = str(round(time.time()))
    # Plot tunning parameters
    plt.figure()
    plt.plot(t, param_Kp, label="Kp")
    plt.plot(t, param_Ti, label="Ti")
    plt.plot(t, param_Td, label="Td")
    plt.title("Tuning Parameters")
    plt.grid()
    plt.legend()
    plt.savefig("tmp/graphs/Tuning_param_"+tm+".png")
    plt.show(block=True)

    # Plot State parameters
    plt.figure()
    plt.plot(t, param_e, label="E")
    plt.plot(t, param_delta_e, label="dE")
    plt.plot(t, param_ie, label="IE")
    plt.plot(t, param_dpv, label="dPV")
    plt.title("State Parameters")
    plt.grid()
    plt.legend()
    plt.savefig("tmp/graphs/State_param_"+tm+".png")
    plt.show(block=True)

    # Plot Process parameters
    plt.figure()
    plt.plot(t, pv, label="PV")
    plt.plot(t, sp_data, label="SP")
    # plt.plot(t, param_cout, label="Process_In")
    plt.title("Process Parameters")
    plt.grid()
    plt.legend()
    plt.savefig("tmp/graphs/Process_param_"+tm+".png")
    plt.show(block=True)

    # Plot Process Input parameter
    plt.figure()
    plt.plot(t, param_cout, label="Process_In")
    plt.title("Process Input Parameter")
    plt.grid()
    plt.legend()
    plt.savefig("tmp/graphs/Process_Input_param_"+tm+".png")
    plt.show(block=True)
    

