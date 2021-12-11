''' Class of principal that uses EXP3 algorithm. '''

import numpy as np
import random 
import math
from params import * 
from probability import *
from oracle_cont import *


class Principal_Exp3(object): 

    # Only one principal in the setting
    def __init__(self, T, calA_exp3, num_repetitions, p):
        self.calA_exp3      = calA_exp3 #array of actions, diff for EXP3 and grinding
        self.calA_size_exp3 = len(calA_exp3)
        self.pi_exp3        = [1.0/self.calA_size_exp3 for i in range(self.calA_size_exp3)]
        self.weights_exp3   = [1 for i in range(self.calA_size_exp3)]
        self.eta_exp3       = math.sqrt(1.0*(math.log(self.calA_size_exp3,2))/(T*self.calA_size_exp3))
        self.est_loss_exp3  = [0]*self.calA_size_exp3
        self.loss_func_exp3 = [[] for _ in range(T)]

    def choose_action_exp3(self):
        a_index = draw(self.pi_exp3, 0.0, 1.0) #index of the chosen action 
        return (self.calA_exp3[a_index], a_index)



