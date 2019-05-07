##################################################################################################
#                                          Easy 21 assignment                                    #
##################################################################################################

# Exercise instructions:
# In this assignment, we want to learn the state-value function for a given policy \pi
# Consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits,
# plus other player’s policies of your choice.  For each of the 2 policies, and for each
# algorithm, plot the optimal value function v_\pi using similar axes to the Figure 5.2 (right)
# from Sutton and Barto’s book. Note that now the dealer can show either a black or a red card.

# Possible actions:
#  stick - Don't draw any new cards
#  hit - draw new card

# State definition:
# Values of the player’s cards (added (black cards) or subtracted (red cards))
# Value of dealer's cards

# State-Action transitions:
# stick -> draw new card
# hit -> The dealer always sticks on any sum of 17 or greater, and hits otherwise.

# Draw card:
# number 1-10 uniform distribution
# Color: 1/3 red 2/3 black


##################################################################################################
#                                  Environment implementation                                    #
##################################################################################################


import random

CARD_MAX_ABS_VALUE = 10
CARD_MIN_ABS_VALUE = 1
RED = "red"
BLACK = "black"
HIT = 0
STICK = 1
PLAYER = 0
DEALER = 1

random.seed(10)

class State:
    def __init__(self):
        self.random = random.Random()
        self.playerPoints = random.randint(CARD_MIN_ABS_VALUE, CARD_MAX_ABS_VALUE)
        self.dealerPoints = random.randint(CARD_MIN_ABS_VALUE, CARD_MAX_ABS_VALUE)

    def updateState(self, card, agent):
        if agent == PLAYER:
            self.playerPoints += card.value
        else:
            self.dealerPoints += card.value

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

class DefaultPlayerPolicy(Policy):
    def act(self, state):
        if state.playerPoints >= 20:
            return STICK

        return HIT

class DefaultDealerPolicy(Policy):
    def act(self, state):
        if state.dealerPoints >= 17:
            return STICK

        return HIT


class EpisodeStep:
    def __init__(self, state, action, reward, timeStep):
        self.state = state
        self.action = action
        self.reward = reward
        self.timeStep = timeStep
    def __str__(self):
        # return "S: {0}; A: {1}; R: {2}; t: {3}".format(str(self.state), "hit" if self.action == 0 else "stick" , str(self.reward), self.timeStep)
        return "(S, A, R, t) = ({0}, {1}, {2}, {3})".format(str(self.state), "hit" if self.action == 0 else "stick" , str(self.reward), self.timeStep)


class Game:
    def __init__(self, playerPolicy, debug = False):
        self.currentState = State()
        self.playerPolicy = playerPolicy
        self.dealerPolicy = DefaultDealerPolicy()
        self.debug = debug


    def rewardFunction(self, state):
        if state.playerPoints > 21 or state.playerPoints < 1:
            return -1
        if state.dealerPoints > 21 or state.dealerPoints < 1:
            return 1
        if not self.stateIsTerminal(state):
            return 0
        if state.dealerPoints == state.playerPoints:
            return 0
        if state.playerPoints - state.dealerPoints > 0:
            return 1
        else:
            return -1

    def stateIsTerminal(self, state):
        if state.playerPoints > 21 or state.playerPoints < 1:
            return True
        if state.dealerPoints > 21 or state.dealerPoints < 1:
            return True
        return False

    def step (self, agent, action, state):
        if self.debug: print(state.playerPoints, state.dealerPoints)
        card = Card()
        if self.debug: print ("{}:".format("player" if agent == 0 else "dealer"), card.value, card.color)
        state.updateState(card, agent)
        return self.rewardFunction(state)

    def SimulateDealerPlay(self, state):
        dealerAction = self.dealerPolicy.act(state)
        while dealerAction == HIT and not self.stateIsTerminal(state):
            self.step(DEALER, dealerAction, state)
            dealerAction = self.dealerPolicy.act(state)

    def SimulateEpisode(self):
        episodes = []
        t = 0
        self.currentState = State()
        playerAction = self.playerPolicy.act(self.currentState)
        while playerAction == HIT and not self.stateIsTerminal(self.currentState):
            stateTuple = (self.currentState.playerPoints, self.currentState.dealerPoints)
            reward = self.step(PLAYER, playerAction, self.currentState)
            episodes.append(EpisodeStep(stateTuple, playerAction, reward, t))
            t += 1
            playerAction = self.playerPolicy.act(self.currentState)


        self.SimulateDealerPlay(self.currentState)

        if self.debug: print("End state:", self.currentState.playerPoints, self.currentState.dealerPoints)

        episodes[-1].reward = self.rewardFunction(self.currentState)
        return episodes


# if __name__ == "__main__":
#     game = Game(DefaultPlayerPolicy(), debug=True)
#     episodes = game.SimulateEpisode()
#     for e in episodes:
#         print (e)