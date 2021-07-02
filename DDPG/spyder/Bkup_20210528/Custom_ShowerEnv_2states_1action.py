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
        self.observation_space = Box(low=np.float32(np.array([0,-1])),\
                                     high=np.float32(np.array([100,1])))
        # Set start temp
        #self.state = np.array([38 + random.randint(-3,3),  1]) #(38 + random.randint(-3,3),  1)
        # Set shower length
        self.shower_length = 60
        
    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature
        temp, const = self.state
        #print("temp:{0} \n const:{1}".format(temp,const))
        temp += action
        #print("temp:{}".format(temp))
        self.state = (temp[0], const) #action #-1 
        # Reduce shower length by 1 second
        
        #print("state:{}".format(self.state))
        self.shower_length -= 1 
        
        # Calculate reward
        if self.state[0] >=37 and self.state[0] <=39: 
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
        self.state = np.array([38 + random.randint(-3,3),  1])
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
    