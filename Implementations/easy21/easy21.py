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
    def __init__(self, playerPolicy, debug = False):
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

    def step (self, state):
        playerAction = self.playerPolicy.act(state)
        if playerAction == HIT and (state.playerPoints <= 21 or state.playerPoints > 0):
            card = Card()
            if self.debug: print ("Player Hit:", card.value, card.color)
            state.updateState(card, PLAYER)
            if self.debug: print (str(state), "hit" if playerAction == 0 else "stick")
        elif state.dealerPoints <= 21 or state.dealerPoints > 0:
            dealerAction = self.dealerPolicy.act(state)
            while dealerAction == HIT and (state.dealerPoints <= 21 or state.dealerPoints > 0):
                card = Card()
                if self.debug: print ("Dealer Hit:", card.value, card.color)
                state.updateState(card, DEALER)
                dealerAction = self.dealerPolicy.act(state)

        return self.rewardFunction(state), playerAction

    def SimulateEpisode(self):
        episodes = []
        t = 0
        self.currentState = State()
        if self.debug: print("Initial state:", self.currentState.playerPoints, self.currentState.dealerPoints)
        while True:
            stateTuple = (self.currentState.playerPoints, self.currentState.dealerPoints)
            reward, playerAction = self.step(self.currentState)
            t += 1
            episodes.append(EpisodeStep(stateTuple, playerAction, reward, t))
            if not reward == 0:
                break

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

    # def step (self, agent, action, state):
    #     if self.debug: print(state.playerPoints, state.dealerPoints)
    #     if action == HIT:
    #         card = Card()
    #         if self.debug: print ("{} Hit:".format("player" if agent == 0 else "dealer"), card.value, card.color)
    #         state.updateState(card, agent)
    #     else:
    #         if self.debug: print ("{} Stick".format("player" if agent == 0 else "dealer"))

    #     return self.rewardFunction(state)

    # def SimulateDealerPlay(self, state):
    #     dealerAction = self.dealerPolicy.act(state)
    #     while dealerAction == HIT and not self.stateIsTerminal(state):
    #         self.step(DEALER, dealerAction, state)
    #         dealerAction = self.dealerPolicy.act(state)