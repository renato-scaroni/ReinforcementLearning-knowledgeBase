#imports
from random import randint
import sys
import math
import json
import matplotlib.pyplot as plt
import math
from math import exp, factorial
import numpy as np


MAX_CARS = 20

MAX_MOVE_OF_CARS = 5

GAMMA = 0.9

SHOP1 = 0
SHOP2 = 1
RENTAL = 0
RETURN = 1

VALUE_THRESHOLD = 1e-4

POISSON_UPPER_BOUND = 11

values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

poisson_cache = dict()
def poisson(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = exp(-lam) * pow(lam, n) / factorial(n)
    return poisson_cache[key]


poissonParameters = [
    [3, 3], # [rental, return]
    [4, 2]  # [rental, return]
]

def probabilityOfTransition(rentals_shop1, rentals_shop2, returns_shop1, returns_shop2):
    return poisson (poissonParameters[SHOP1][RENTAL], rentals_shop1) *\
           poisson (poissonParameters[SHOP2][RENTAL], rentals_shop2) *\
           poisson (poissonParameters[SHOP1][RETURN], returns_shop1) *\
           poisson (poissonParameters[SHOP2][RETURN], returns_shop2)


def stateValue(s, a, values):
    v = 0
    # this adds the new car moving restriction
    v -= max(0, abs(a)-1)*2
    for rentals_1 in range(POISSON_UPPER_BOUND):
        for rentals_2 in range(POISSON_UPPER_BOUND):
            cars_shop1 = s[SHOP1] - a
            cars_shop2 = s[SHOP2] + a

            rentals_1 = min(rentals_1, cars_shop1)
            rentals_2 = min(rentals_2, cars_shop1)

            earnings = (rentals_1 + rentals_2) * 10
            cars_shop1 -= rentals_1
            cars_shop2 -= rentals_2

            # the following two lines implement the exceding overnight cars punishment
            earnings -= max(cars_shop1 -  10, 0) * 4
            earnings -= max(cars_shop2 -  10, 0) * 4

            for returns_1 in range(POISSON_UPPER_BOUND):
                for returns_2 in range(POISSON_UPPER_BOUND):
                    cars_shop1 = min(returns_1 + cars_shop1, MAX_CARS)
                    cars_shop2 = min(returns_2 + cars_shop2, MAX_CARS)
                    p = probabilityOfTransition(rentals_1, rentals_2, returns_1, returns_2)
                    v += p * (earnings + values[cars_shop1, cars_shop2])

    return v

def updateValues(values, policy):
    for i in range(MAX_CARS+1):
        for j in range(MAX_CARS+1):
            a = policy[i, j]
            s1 = [i, j]
            values[i, j] = stateValue([i, j], policy[i, j], values)

    return values

policy = np.zeros(values.shape, dtype=np.int)

actions = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

while True:
    value_change = 1

    while value_change > VALUE_THRESHOLD:
        new_values = updateValues(np.copy(values), policy)
        value_change = np.abs((new_values - values)).sum()
        print('value change %f' % (value_change))
        values = new_values

    # policy improvement
    new_policy = np.copy(policy)
    for i in range(MAX_CARS + 1):
        for j in range(MAX_CARS + 1):
            action_returns = []
            for action in actions:
                if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
                    action_returns.append(stateValue([i, j], policy[i, j], values))
                else:
                    action_returns.append(-float('inf'))
            new_policy[i, j] = actions[np.argmax(action_returns)]

    policy_change = (new_policy != policy).sum()
    print('policy changed in %d states' % (policy_change))
    policy = new_policy
    print(policy)
    if policy_change == 0:
        break
