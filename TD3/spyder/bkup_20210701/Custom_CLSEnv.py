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
class CLSEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Box(low=np.float32(np.array([-1])),\
                                     high=np.float32(np.array([1]))) #Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.float32(np.array([0])),\
                                     high=np.float32(np.array([200])))
        # Set start temp
        self.state = np.array([60 + random.uniform(-3,3)])
        # Set no of steps
        self.ns = 300
        # Set output
        self.pv = []
        # Set Set point
        self.sp = 48
        # Set error
        self.error = np.zeros(self.ns, dtype=float)
        # Set epsilon
        self.epsilon = 0.05
        # Set current position
        self.cpos = np.array([24 + random.uniform(-0.1,0.1)])
        
    def process(self, y,t,u,Kp,taup):
        dydt = -y/taup + Kp/taup * u
        return dydt
        
    def step(self, action, delta_t, Kp, taup, i):
        # Solve ODE equation
        self.cpos += action
        self.cpos = np.max([0,np.min([self.cpos,100])])
        y = odeint(self.process,self.state,[0,delta_t],args=(self.cpos,Kp,taup))
        self.state = y[-1]
        self.pv.append(self.state)
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature
        #print("temp:{0} \n const:{1}".format(temp,const))
        #print(self.state.shape, type(self.state), self.state)
        #print(action, type(action), action)
        #self.state += action #-1
        #print("temp:{}".format(temp))
        # Reduce shower length by 1 second
        
        #print("state:{}".format(self.state))
        self.ns -= 1 
        
        if i > 1:
            self.error[i] = abs(self.pv[i] - self.sp)
            delta_err = self.error[i-1] - self.error[i]
        else:
            self.error[i] = 0
            delta_err = 0
        
        # Calculate reward-1
        if self.error[i] <= self.epsilon:
            r1 = 2
        else:
            r1 = -self.error[i]
        # Calculate reward-2
        r2 = -2*abs(delta_err)
        # Calclate reward-3
        if delta_err <= 0.001:
            r3 = -2*abs(delta_err)
        else:
            r3 = -1
        #Calculate reward-4
        if self.error[i-1]<=self.epsilon and self.error[i]<=self.epsilon:
            r4 = 2
        else:
            r4 = -1
                 
        reward = r1+r2+r3+r4
        #if self.state >=47 and self.state <=49: 
        #    reward =1 
        #else: 
        #    reward = -1
        
        # Check if shower is done
        if self.ns <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        #print("Action:{0}, State:{1}, Reward:{2}, Error: {3}"\
        #      .format(self.cpos, self.state, reward, self.error[i]))
        # Return step information
        return np.array(self.state), reward, done, info, np.array(self.pv)

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([60 + random.uniform(-3,3)])
        # Set no of steps
        self.ns = 300 
        # Set output
        self.pv = []
        # Set Set point
        self.sp = 48
        # Set error
        self.error = np.zeros(self.ns, dtype=float)
        # Set current position
        self.cpos = np.array([24 + random.uniform(-0.1,0.1)])
        return np.array(self.state)
    

# Running locally
if __name__ == '__main__':
    env = CLSEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    print("State Dim: {0}\n Action Dim: {1}\n Action Bound: {2}"\
          .format(state_dim, action_dim, action_bound))
    #print("Observation")
    #print(env.observation_space.shape, type(env.observation_space))
    
    episodes = 10
    # specify number of steps
    ns = 300
    # define time points
    t = np.linspace(0,ns/10,ns+1)
    delta_t = t[1]-t[0]
    # process model
    Kp = 2.0
    taup = 5.0
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0
        k = 0
        
        while not done:
            #env.render()
            action = env.action_space.sample()
            n_state, reward, done, info, pv = env.step(action, delta_t, Kp, taup, k)
            score+=reward
            k += 1
        print('Episode:{} Score:{}'.format(episode, score))
        #print("Process Value: {}".format(pv))
    