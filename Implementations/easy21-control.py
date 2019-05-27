#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'easy21'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Easy 21 Control Assignment
# ### Exercise instructions:
# In this assignment, we want to learn the state-value function for a given policy \pi
# Consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits,
# plus other player’s policies of your choice.  For each of the 2 policies, and for each
# algorithm, plot the optimal value function v_\pi using similar axes to the Figure 5.2 (right)
# from Sutton and Barto’s book. Note that now the dealer can show either a black or a red card.
#
# ### Possible actions:
# - stick - Don't draw any new cards
# - hit - draw new card
#
# ### State definition:
# - Values of the player’s cards (added (black cards) or subtracted (red cards))
# - Value of dealer's cards
#
# ### State-Action transitions:
# - stick -> draw new card
# - hit -> The dealer always sticks on any sum of 17 or greater, and hits otherwise.
#
# ### Draw card:
# - number 1-10 uniform distribution
# - Color: 1/3 red 2/3 black
#
#%% [markdown]
# # Part 1 - Implementation of Easy21 simulator

#%%
# %load easy21-environment.py

##################################################################################################
#                                  Environment implementation                                    #
##################################################################################################


import random

# defining constants
CARD_MAX_ABS_VALUE = 10
CARD_MIN_ABS_VALUE = 1
RED = "red"
BLACK = "black"
HIT = 0
STICK = 1
PLAYER = 0
DEALER = 1

class State:
    def __init__(self):
        self.random = random.Random()
        self.playerPoints = random.randint(CARD_MIN_ABS_VALUE, CARD_MAX_ABS_VALUE)
        self.dealerPoints = random.randint(CARD_MIN_ABS_VALUE, CARD_MAX_ABS_VALUE)
        self.isTerminal = False

    def toStateTuple(self):
        return (self.playerPoints, self.dealerPoints)

    def updateState(self, card, agent):
        if agent == PLAYER:
            self.playerPoints += card.value
        else:
            self.dealerPoints += card.value

    def __str__(self):
        return "(Pl, De) = ({0}, {1})".format(self.playerPoints, self.dealerPoints)

class Card(object):
    def __init__ (self):
        self.color = Card.getColor()
        self.absValue = random.randint(CARD_MIN_ABS_VALUE, CARD_MAX_ABS_VALUE)
        self.value = self.absValue if self.color == BLACK else -self.absValue

    @staticmethod
    def getColor():
        colorVariable = random.randint(1,3)
        return RED if colorVariable == 1 else BLACK

class Policy:
    def act(self, state):
        pass

class DefaultDealerPolicy(Policy):
    def act(self, state):
        if state.dealerPoints >= 17:
            return STICK

        return HIT

class DefaultPlayerPolicy(Policy):
    def act(self, state):
        if state.playerPoints >= 20:
            return STICK

        return HIT

class EpisodeStep:
    def __init__(self, state, action, reward, timeStep):
        self.state = state
        self.action = action
        self.reward = reward
        self.timeStep = timeStep
    def __str__(self):
        return "(S, A, R, t) = ({0}, {1}, {2}, {3})".format(str(self.state), "hit" if self.action == 0 else "stick", str(self.reward), self.timeStep)


class Game:
    def __init__(self, playerPolicy=None, debug=False):
        self.currentState = State()
        self.playerPolicy = playerPolicy
        self.dealerPolicy = DefaultDealerPolicy()
        self.debug = debug
        self.RandomReset()


    def rewardFunction(self, state):
        if state.playerPoints > 21 or state.playerPoints < 1:
            return -1
        if state.dealerPoints > 21 or state.dealerPoints < 1:
            return 1
        if not state.isTerminal:
            return 0
        if state.dealerPoints == state.playerPoints:
            return 0
        if state.playerPoints - state.dealerPoints > 0:
            return 1
        else:
            return -1

    def step (self, state, playerAction):
        if playerAction == HIT and (state.playerPoints <= 21 or state.playerPoints > 0):
            card = Card()
            if self.debug: print ("Player Hit:", card.value, card.color)
            state.updateState(card, PLAYER)
            if self.debug: print("Current state:", state.playerPoints, state.dealerPoints)
            if state.playerPoints > 21 or state.playerPoints < 1:
                state.isTerminal = True
        elif state.dealerPoints <= 21 or state.dealerPoints > 0:
            if self.debug: print ("Player stick", str(state))
            dealerAction = self.dealerPolicy.act(state)
            while dealerAction == HIT:
                card = Card()
                if self.debug: print ("Dealer Hit:", card.value, card.color)
                state.updateState(card, DEALER)
                if self.debug: print("Current state:", state.playerPoints, state.dealerPoints)
                dealerAction = self.dealerPolicy.act(state)
            state.isTerminal = True

        return self.rewardFunction(state), state

    def SimulateEpisode(self):
        episodes = []
        t = 0
        self.currentState = State()
        if self.debug: print("Initial state:", self.currentState.playerPoints, self.currentState.dealerPoints)
        while not self.currentState.isTerminal:
            stateTuple = (self.currentState.playerPoints, self.currentState.dealerPoints)
            playerAction = self.playerPolicy.act(self.currentState)
            reward, _ = self.step(self.currentState, playerAction)
            t += 1
            episodes.append(EpisodeStep(stateTuple, playerAction, reward, t))

        if self.debug: print("End state:", self.currentState.playerPoints, self.currentState.dealerPoints)

        return episodes

    def SimulateMultipleEpisodes(self, n):
        sample = []
        self.RandomReset()
        for i in range(n):
            sample.append(self.SimulateEpisode())

        return sample

    def RandomReset(self):
        random.seed(10)

#%%
##################################################################################################
#                                         Environment test                                       #
##################################################################################################

game = Game(DefaultPlayerPolicy())

for i in range (10):
    episodes = game.SimulateEpisode()
    for e in episodes:
        print (e)
    print ("-------------------------------------")

#%% [markdown]
# # Auxiliary functions and imports for the tests

#%%
##################################################################################################
#                                             Imports                                            #
##################################################################################################

get_ipython().run_line_magic('matplotlib', 'inline')
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sys import argv
from itertools import product
import sys
from time import time


#%%
##################################################################################################
#                                         Auxiliary methods                                      #
##################################################################################################

def extractValueFunction(q):
    v = []
    for k in q:
        v.append(((k[0], k[1]), q[k]))
    return v

def argmaxA(q, s):
    if (s[0], s[1], HIT) in q:
        return HIT if q[(s[0], s[1], HIT)] > q[(s[0], s[1], STICK)] else STICK
    return 0

def getStateActionVisits(episode):
    firstStateActionVisits = {}
    everyStateVisitsCount = dict.fromkeys(product(range(1, 22), range(1, 11)), 0)
    for t in range(len(episode)):
        step = episode[t]
        everyStateVisitsCount[step.state] += 1
        if not step.state in firstStateActionVisits:
            firstStateActionVisits[(step.state[0], step.state[1], step.action)] = t+1

    return firstStateActionVisits, everyStateVisitsCount

def plotMutipleValueFunction(vPis, sizes, rows=2, cols=3, message='episode', width=19, height=9.5):
    fig = plt.figure(figsize=(width, height))
    for i in range(len(vPis)):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        ax.set_title("{} {}{}".format(sizes[i], message,'s' if i > 0 else ''),
                     fontsize=12)
#         fig.colorbar(surf, shrink=0.5, aspect=5)
        plotSurface(vPis[i], ax)

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()

def plotSurface(vPi, ax):
    x = list(map(lambda x: x[0][1], vPi))
    y = list(map(lambda y: y[0][0], vPi))
    z = list(map(lambda x: x[1], vPi))
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})


    ax.set_xlabel('Dealer initial card')
    ax.set_ylabel('Player card sum')
    ax.set_zlabel('State value')

    ax.set_xticks(range(1,11))
    ax.set_yticks(range(1,22,2))
    ax.set_zlim([-1, 1])

    ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.coolwarm, linewidth=0.1)

def plotValueFunction(vPi):
    x = list(map(lambda x: x[0][1], vPi))
    y = list(map(lambda y: y[0][0], vPi))
    z = list(map(lambda x: x[1], vPi))
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('Dealer initial card')
    ax.set_ylabel('Player card sum')
    ax.set_zlabel('State value')

    ax.set_xticks(range(1,11))
    ax.set_yticks(range(1,22,2))
    ax.set_zlim([-1, 1])

    surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.coolwarm, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

#%% [markdown]
# # Part 2 - Monte Carlo Control
#
# $V(S_t) \leftarrow V(S_t) + [G_t - V(S_t)]$.
#
# Where:
#
# $G_t = R_{t+1}+\gamma R_{t+2}+ \gamma^2 R_{t+3}...+\gamma^{T-t}R_{T}$

#%%
##################################################################################################
#                      Definition of  Monte Carlo method for policy control                      #
##################################################################################################

def GetReturn(episode, t, gamma):
    G = 0
    t = t-1
    for i in range(t, len(episode)):
        G += pow(gamma, i-t) * episode[i].reward

    return G

def MonteCarloControl(episodes, gamma=1, n0 = 100, everyVisit=True):
    # Generic Monte Carlo initialization
    q = dict.fromkeys(product(range(1, 22), range(1, 11), range(0,2)), 0)
    nStateAction = dict.fromkeys(product(range(1, 22), range(1, 11), range(0,2)), 0)
    nState = dict.fromkeys(product(range(1, 22), range(1, 11)), 0)
    game = Game()
    game.RandomReset()
    epsilon = 1
    visiteStateAction = set()
    visiteStates = set()
    for _ in range(episodes):
        episode = []
        visiteStateAction.clear()
        visiteStates.clear()
        t = 0
        S = State()
        while not S.isTerminal:
            stateTuple = S.toStateTuple()
            if random.random() < epsilon:
                playerAction = random.randint(0, 1)
            else:
                playerAction = argmaxA(q, stateTuple)
            reward, _ = game.step(S, playerAction)
            t += 1
            episode.append(EpisodeStep(stateTuple, playerAction, reward, t))
        for e in episode:
            SA = (e.state[0], e.state[1], e.action)
            if not SA in visiteStateAction or everyVisit:
                visiteStateAction.add(SA)
                nStateAction[SA] += 1
                q[SA] += 1/nStateAction[SA] * (GetReturn(episode, e.timeStep, gamma) - q[SA])
                if not e.state in visiteStates or everyVisit:
                    nState[e.state] += 1
                    visiteStates.add(e.state)
                epsilon = n0/(n0+nState[e.state])
    return q

#%%
numberOfEpisodes = 100000
q1,q2  = MonteCarloControl(numberOfEpisodes), MonteCarloControl(numberOfEpisodes, False)

# plotValueFunction(extractValueFunction(q))
plotMutipleValueFunction([extractValueFunction(q1), extractValueFunction(q2)], ['every visit', 'first visit'], rows=1, cols=2,message='')
#%% [markdown]
# # Part 3 - TD Control
#
# $V(S_t) \leftarrow V(S_t) + [G_t - V(S_t)]$.
#
# Where:
#
# $G_t = R_{t+1}+\gamma R_{t+2}+ \gamma^2 R_{t+3}...+\gamma^{T-t}R_{T}$
#%%
def getNextAction(stateTuple, q, epsilon):
    if random.random() < epsilon:
        playerAction = random.randint(0, 1)
    else:
        playerAction = argmaxA(q, stateTuple)

    return playerAction

def SARSA(episodes, gamma=1, n0 = 100, everyVisit=True):
    # Generic Monte Carlo initialization
    q = dict.fromkeys(product(range(1, 22), range(1, 11), range(0,2)), 0)
    nStateAction = dict.fromkeys(product(range(1, 22), range(1, 11), range(0,2)), 0)
    nState = dict.fromkeys(product(range(1, 22), range(1, 11)), 0)
    game = Game()
    game.RandomReset()
    epsilon = 1
    for _ in range(episodes):
        S = State()
        A = getNextAction(S.toStateTuple(), q, epsilon)
        while not S.isTerminal:
            SA = (S.playerPoints, S.dealerPoints, A)
            # the step method already updates the current state S
            # with the resulting state
            R, SPrime = game.step(S, A)
            APrime = getNextAction(SPrime.toStateTuple(), q, epsilon)
            SAPrime = (SPrime.playerPoints, SPrime.dealerPoints, APrime)
            nStateAction[SA] += 1
            QSAPrime = 0 if SPrime.isTerminal else q[SAPrime]
            q[SA] += 1/nStateAction[SA] * (R + gamma*QSAPrime - q[SA])
            nState[e.state] += 1
            epsilon = n0/(n0+nState[e.state])
            A = APrime
            S = SPrime
    return q
#%%
sizes = [1, 10, 100, 1000, 10000, 100000]
vStars = []
for numberOfEpisodes in sizes:
    q = SARSA(numberOfEpisodes)
    vStars.append(extractValueFunction(q))

plotMutipleValueFunction(vStars, sizes)
#%%
def QLearning(episodes, gamma=1, n0 = 100, everyVisit=True):
    # Generic Monte Carlo initialization
    q = dict.fromkeys(product(range(1, 22), range(1, 11), range(0,2)), 0)
    nStateAction = dict.fromkeys(product(range(1, 22), range(1, 11), range(0,2)), 0)
    nState = dict.fromkeys(product(range(1, 22), range(1, 11)), 0)
    game = Game()
    game.RandomReset()
    epsilon = 1
    for _ in range(episodes):
        S = State()
        while not S.isTerminal:
            A = getNextAction(S.toStateTuple(), q, epsilon)
            SA = (S.playerPoints, S.dealerPoints, A)
            # the step method already updates the current state S
            # with the resulting state
            R, SPrime = game.step(S, A)
            APrime = getNextAction(SPrime.toStateTuple(), q, 0)
            SAPrime = (SPrime.playerPoints, SPrime.dealerPoints, APrime)
            nStateAction[SA] += 1
            QSAPrime = 0 if SPrime.isTerminal else q[SAPrime]
            q[SA] += 1/nStateAction[SA] * (R + gamma*QSAPrime - q[SA])
            nState[e.state] += 1
            epsilon = n0/(n0+nState[e.state])

    return q
#%%
sizes = [1, 10, 100, 1000, 10000, 100000]
vStars = []
for numberOfEpisodes in sizes:
    q = QLearning(numberOfEpisodes)
    vStars.append(extractValueFunction(q))

plotMutipleValueFunction(vStars, sizes)
