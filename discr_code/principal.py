import numpy as np
import random 
import math
from params import * 
from probability import *
from oracle import *


class Principal(object): 

    # Only one principal in the setting
    def __init__(self, T, calA, num_repetitions, p, a_G):
        self.calA           = calA #array of actions
        self.calA_size      = len(calA)
        self.pi             = [1.0/self.calA_size for i in range(self.calA_size)]
        self.weights        = [1 for i in range(self.calA_size)]
        self.eta_exp3       = math.sqrt((math.log(self.calA_size,2))/(T*self.calA_size))
        self.eta_dgrind     = math.sqrt((math.log(self.calA_size,2))/(T*(a_G)))
        self.eta_dgrind_regress     = math.sqrt((math.log(self.calA_size,2))/(T*(self.calA_size)))
        self.eta_finfo      = math.sqrt((math.log(self.calA_size,2))/(T))       
        self.est_loss       = [0]*self.calA_size
        self.loss_func      = [[] for _ in range(T)]

    def choose_action(self):
        a_index = draw(self.pi, 0) #index of the chosen action 
        return (self.calA[a_index], a_index)



