# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:54:13 2021

@author: kranthi
"""
# Import Packages
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random


# Custom Environment creation
class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Box(low=np.float32(np.array([-1])),\
                                     high=np.float32(np.array([1]))) #Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.float32(np.array([0])),\
                                     high=np.float32(np.array([100])))
        # Set start temp
        self.state = np.array([48 + random.uniform(-3,3)])
        # Set shower length
        self.shower_length = 60
        
    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature
        #print("temp:{0} \n const:{1}".format(temp,const))
        #print(self.state.shape, type(self.state), self.state)
        #print(action, type(action), action)
        self.state += action #-1
        #print("temp:{}".format(temp))
        # Reduce shower length by 1 second
        
        #print("state:{}".format(self.state))
        self.shower_length -= 1 
        
        # Calculate reward
        if self.state >=47 and self.state <=49: 
            reward =1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return np.array(self.state), reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([48 + random.uniform(-3,3)])
        # Reset shower time
        self.shower_length = 60 
        return np.array(self.state)
    

# Running locally
if __name__ == '__main__':
    env = ShowerEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    print("State Dim: {0}\n Action Dim: {1}\n Action Bound: {2}"\
          .format(state_dim, action_dim, action_bound))
    #print("Observation")
    #print(env.observation_space.shape, type(env.observation_space))
    
    episodes = 10
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            #env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
    