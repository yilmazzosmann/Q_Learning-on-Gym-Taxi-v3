#!/usr/bin/env python
# coding: utf-8

import gym
import random
import numpy as np
import time
# Environment
env = gym.make("Taxi-v3")


# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode

# Q tables for rewards
Q_reward = -1000*np.ones((500,6))



# Training w/ random sampling of actions
for episodes in range(num_of_episodes):
    
    state = env.reset() 
    done = False
    step = 0
    
    for step in range(num_of_steps):
        
        action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        
        if done == True:
            Q_reward[state, action] = Q_reward[state, action]+ alpha*(reward+ gamma*0-Q_reward[state,action])
            break
        else:
            Q_reward[state, action] = Q_reward[state,action]+ alpha*(reward+ gamma*np.max(Q_reward[new_state,:])-Q_reward[state,action])
        
        state = new_state
    #print("Episode :{}".format(episodes))
    episodes +=1




state = env.reset()
tot_reward = 0
for t in range(50):
    action = np.argmax(Q_reward[state,:])
    state, reward, done, info = env.step(action)
    tot_reward += reward
    env.render()
    time.sleep(1)
    if done:
        print("Total reward %d" %tot_reward)
        break

