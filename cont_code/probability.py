# Code for drawing an arm given the probabilities of the arms
# Gets as input the list of the probabilities for each arm
# Returns the arm to be drawn
import random
import numpy as np

# For discrete probability distributions
def draw(probs_lst, gamma, calA_vol):
    np.random.seed()
    t           = np.random.uniform(0,1)
    cumulative  = 0.0
    for i in range(len(probs_lst)):
        cumulative += (1.0 - gamma)*probs_lst[i] + 1.0*gamma/calA_vol
        if cumulative > t:
            return i
    return (len(probs_lst)-1)

