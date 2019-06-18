from MountainCarEnv import MountainCarEnv
import math
from numpy.random import choice
import numpy as np
from functools import reduce
from operator import add
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def h(s, a, theta):
    return theta[0]*s[0] + theta[1]*s[1] + theta[2]*a

def pi_softmax (s, current_action, theta, action_space):
    d = 0
    for a in range(action_space.n):
        d += math.e**h(s,a,theta)
    return math.e**h(s,current_action,theta) / d

# implementation of the algorithm
def REINFORCE(n_episodes, pi, env, gamma=1, alpha=0.1, max_steps=1000):
    theta = (0,0,0)
    possible_actions = np.array(range(env.action_space.n))
    for _ in range(n_episodes):
        env.reset()
        episode = []
        for _ in range(max_steps):
            env.render()
            prev_state = env.state
            probabilities = pi(env.state, possible_actions, theta, env.action_space)
            action = choice(possible_actions, p=probabilities)
            new_state, reward, done, _ = env.step(action)
            episode.append((prev_state, action, reward, new_state))
            if done:
                break
        # for t, step in enumerate(episode):
        #     G = reduce(lambda acc, t: gamma*(acc + t[1]), episode, 0)
        #     theta += alpha*(gamma**t)*G*
        print (episode)
    env.close()

env = MountainCarEnv()
REINFORCE(1, pi_softmax, env)


# print (env.action_space.n)
# env.reset()
# a = np.array(range(env.action_space.n))
# print(a)
# probs = pi_softmax(env.state, a, (0,0,1), env.action_space)
# print(probs)
# print(choice(a, p=probs))
# print (env.step(1))

