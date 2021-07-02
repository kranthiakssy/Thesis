# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:54:13 2021

@author: kranthi
"""
# Closed Loop System Environment

# Import Packages
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Custom Environment creation
class PIDEnv(Env):
    def __init__(self):
        # Actions we can take are incremental changes in Kp, Ti, Td
        self.action_space = Box(low=np.float32(np.array([-1,-1,-1])),\
                                     high=np.float32(np.array([1,1,1]))) #Discrete(3)
        # Observed Process Value 
        self.observation_space = Box(low=np.float32(np.array([0])),\
                                     high=np.float32(np.array([200])))

        # Initialization Parameters
        self.init_state = np.array([0]) # initial state of the process value
        self.Gp = np.array([2]) # Process Gain
        self.taup = np.array([5]) # Process time constant
        self.thetap = np.array([1]) # Process delay

        # Parameters for reward functions
        self.k1 = np.array([1])
        self.k2 = np.array([1])
        self.c1 = np.array([1])
        self.c2 = np.array([1])
        self.epsilon = np.array([0.01])
        self.epsilon1 = np.array([0.01])


    def process(self, y,t,u,dummy):
        dydt = -y/self.taup + self.Gp/self.taup * u
        return dydt

    # defining environment step function
    def step(self, actionvector, statevector):
        #Action arguments
        self.Kp = np.max([0,actionvector[0]]) # Proportional Gain
        self.Ti = np.max([0.000001,actionvector[1]]) # Integral Time
        self.Td = np.max([0,actionvector[2]]) # Derivative time
        # Statevector arguments
        self.e = statevector[0][0] # error: e(t)
        self.delta_e = statevector[1][0] # Delta error: e(t-1)-e(t)
        self.ie = statevector[2][0] # Integral Error: ie(t-1) + e(t) * dt
        self.dpv =  statevector[3][0] # Process value rate of change: (PV(t)-PV(t-1))/dt
        self.dt = statevector[4] # time step duration: (t)-(t-1)
        self.pv = statevector[5][0] # Process Value: pv
       # PID terms
        P = self.Kp * self.e # P-Term
        I = self.Kp / self.Ti * self.ie # I-Term
        D = - self.Kp * self.Td * self.dpv 
        # Controller Output
        cout = P + I + D
        cout = np.max([0, np.min([100, cout])])
        # Running odeint solver for ODE
        y = odeint(self.process,self.pv,[0,self.dt],args=(cout,"dummy"))
        self.state = y[-1]
 
        # Defining Reward Function
        # r1 function
        if abs(self.e) <= self.epsilon:
            r1 = self.c1
        else:
            r1 = -abs(self.e)
        # r2 function
        r2 = - self.k1 * abs(self.delta_e)
        # r3 function
        if self.delta_e <= self.epsilon1:
            r3 = - self.k2 * abs(self.delta_e)
        else:
            r3 = 0
        # r4 function
        if ((self.delta_e+self.e)<=self.epsilon) and (self.e<=self.epsilon):
            r4 = self.c2
        else:
            r4 = 0
        # total reward         
        reward = r1+r2+r3+r4

        # Checking end of simulation
        done = False

        # Info function
        info = {}

        return np.array(self.state), reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        self.init_state = np.array([0]) # initial state of the process value
        self.state = np.array([0]) # Reset observed process value
        return np.array(self.state)
    

# Running locally
if __name__ == '__main__':
    env = PIDEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    print("State Dim: {0}\n Action Dim: {1}\n Action Bound: {2}"\
          .format(state_dim, action_dim, action_bound))
    #print("Observation")
    #print(env.observation_space.shape, type(env.observation_space))
    
    episodes = 10 # no of episodes
    ns = 300 # no of steps to run in each episode    
    t = np.linspace(0,ns/10,ns+1) # define time points
    dt = t[1]-t[0] # time step duration

    # initial controller parameters
    def initialize():
        global tune_param, pv, sp, e, delat_e, ie, dpv
        tune_param = [0.1, 2, 0.1]  # Kp, Ti, Td respectively
        pv = [0] # process value list
        sp = 20 # setpoint
        e = [0] # error list
        delat_e = [0] # change in error list
        ie = [0] # integral error list
        dpv = [0] # change in pv list


    # action space function
    def statevectorfunc(state, sp, dt):
        pv.append(state)
        e.append(sp-pv[-1])
        delat_e.append(e[-2]-e[-1])
        ie.append(ie[-1]+e[-1]*dt)
        dpv.append((pv[-1]-pv[-2])/dt)
        statevec = [e[-1], delat_e[-1],ie[-1],dpv[-1],dt,pv[-1]]
        return statevec

    # running for specific no of episodes
    for episode in range(1, episodes+1):
        state = env.reset()
        initialize()
        done = False
        score = 0        
        for k in range(0,ns):
            #env.render()
            action = env.action_space.sample()
            tune_param += action
            statevec = statevectorfunc(state, sp, dt)
            state, reward, done, info  = env.step(tune_param, statevec)
            score += reward[0]
        print('Episode:{} Score:{}'.format(episode, score))
        #print("Process Value: {}".format(pv))