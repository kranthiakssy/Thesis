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
np.random.seed(0)

# Tensorboard Initialization for visualization
# tb = SummaryWriter()

# adding models graphs to tensorboard
# tb.add_graph(agent.actor,T.tensor([0,0,0,0]))

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
        sp = 32 # setpoint
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
    for k in range(0,ns):
        #env.render()
        action = agent.test_action(statevec)
        tune_param += action
        tune_param = np.maximum([0,0.000001,0],tune_param)
        new_state, reward, done, info, cout  = env.step(tune_param, statevec, dt, pv[-1])
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
        new_statevec = statevectorfunc(pv, sp, dt)
        score += reward
        statevec = new_statevec

        # add data to tensorboard
        """ if episode % update_tb == 0:
            tb.add_scalars("Tune_Param/Test-"+str(episode),{"Kp":tune_param[0],
                                                                "Ti":tune_param[1],
                                                                "Td":tune_param[2]},k)
            tb.add_scalars("Process_Value/Test-"+str(episode),{"PV":new_state,
                                                                "SP":sp},k)
            tb.add_scalars("StateVector/Test-"+str(episode),{"Error":statevec[0],
                                                                "dE":statevec[1],
                                                                "IE":statevec[2],
                                                                "dpv":statevec[3]},k) """
    
    # Calculation of closed loop response parameters
    # Calculate ITAE (Integral of time weighted absolute error)
    itae = dt * np.dot(t.T,abs(np.subtract(sp_data, pv)))
    # Calculate maximum overshoot
    mos = np.max(pv) - sp
    # Calculation of rise time
    rt = t[np.array(pv) >= (pv[0]+abs(pv[0]-sp) * 0.9)][0]
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
    
"""     # add data to tensorboard
    tb.add_scalar("Reward",score,episode)
    tb.add_scalar("End_State",new_state,episode)
    tb.add_scalar("ITAE",itae,episode)
    tb.add_scalar("OverShoot",mos,episode)
    tb.add_scalar("RiseTime",rt,episode)
    tb.add_scalar("SteadyStateError",ess,episode)

    # Display episode information
    score_history.append(score)
    print('episode ', episode, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

tb.close() """

