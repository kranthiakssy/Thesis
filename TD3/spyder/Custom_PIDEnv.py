# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:54:13 2021

@author: kranthi
"""
# Closed Loop System Environment for first order system

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
        self.action_space = Box(low=np.float32(np.array([-0.2,-0.2,-0.2])),\
                                     high=np.float32(np.array([0.2,0.2,0.2]))) #Discrete(3)
        # Observed Process Value 
        self.observation_space = Box(low=np.float32(np.array([-100,-100,-100,-100])),\
                                     high=np.float32(np.array([100,100,100,100])))

        # Initialization Parameters
        self.init_state = 0 # initial state of the process value
        self.Gp = 2 # Process Gain
        self.taup = 5 # Process time constant
        self.thetap = 1 # Process delay

        # Parameters for reward functions
        self.k1 = 1
        self.k2 = 1
        self.c1 = 1
        self.c2 = 1
        self.epsilon = 0.01
        self.epsilon1 = 0.01


    def process(self, y,t,u,dummy):
        # try:
        if (len(u)*0.1-self.thetap) <= 0:
            um = u[0]
        else:
            um = u[len(u)-int(self.thetap/0.1)-1]
        # except:
            #print('Error with time extrapolation: ' + str(t))
            # um = 0
        # print(u)
        # calculate derivative
        dydt = -y/self.taup + self.Gp/self.taup * um
        return dydt

    # defining environment step function
    def step(self, actionvector, statevector, dt, pv, p_in):
        #Action arguments
        self.Kp = actionvector[0] #np.max([0,actionvector[0]]) # Proportional Gain
        self.Ti = actionvector[1] #np.max([0.000001,actionvector[1]]) # Integral Time
        self.Td = actionvector[2] * 0 #np.max([0,actionvector[2]]) # Derivative time
        # Statevector arguments
        self.e = statevector[0] # error: e(t)
        self.delta_e = statevector[1] # Delta error: e(t-1)-e(t)
        self.ie = statevector[2] # Integral Error: ie(t-1) + e(t) * dt
        self.dpv =  statevector[3] # Process value rate of change: (PV(t)-PV(t-1))/dt
        self.dt = dt # time step duration: (t)-(t-1)
        self.pv = pv # Process Value: pv
       # PID terms
        P = self.Kp * self.e # P-Term
        I = self.Kp / self.Ti * self.ie # I-Term
        D = - self.Kp * self.Td * self.dpv # D-Term
        # Controller Output
        cout = P + I + D
        csat = False # Controller output saturated?
        if cout > 100 or cout < 0:
            csat = True  # Controller output saturated?
        cout_clip = np.max([0, np.min([100, cout])])
        p_in.append(cout_clip)
        # Running odeint solver for ODE
        y = odeint(self.process,self.pv,[0,self.dt],args=(p_in,"dummy"))
        self.state = y[-1][0]
 
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
        # r5 function
        if cout > 90:
            r5 = 90 - cout
        elif cout < 0:
            r5 = cout
        else:
            r5 = 0
        # total reward         
        reward = r1+r2+r3+r4+r5

        # Checking end of simulation
        done = False

        # Info function
        info = {}

        return self.state, reward, done, info, cout, csat, p_in

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        self.init_state = 0 # initial state of the process value
        self.state = 0 # Reset observed process value
        return self.state
    

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
        global tune_param, pv, sp, e, delat_e, ie, dpv, statevec
        tune_param = [0.1, 2, 0.1]  # Kp, Ti, Td respectively
        pv = [0] # process value list
        sp = 20 # setpoint
        e = [0] # error list
        delat_e = [0] # change in error list
        ie = [0] # integral error list
        dpv = [0] # change in pv list
        statevec = [0,0,0,0] # e, delta_e, ie, dpv


    # action space function
    def statevectorfunc(pv, sp, dt):
        e.append(sp-pv[-1])
        delat_e.append(e[-2]-e[-1])
        ie.append(ie[-1]+e[-1]*dt)
        dpv.append((pv[-1]-pv[-2])/dt)
        statevec = [e[-1], delat_e[-1],ie[-1],dpv[-1]]
        return statevec

    # running for specific no of episodes
    for episode in range(1, episodes+1):
        state = env.reset()
        initialize()
        done = False
        score = 0
        p_in = []      
        for k in range(0,ns):
            #env.render()
            action = env.action_space.sample()
            tune_param += action
            state, reward, done, info, cout, csat,  p_in = env.step(tune_param, statevec, dt, pv[-1], p_in)
            pv.append(state)
            statevec = statevectorfunc(pv, sp, dt)
            if csat == True:
                ie[-1] = ie[-2] # anti-reset windup
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
        #print("Process Value: {}".format(pv))