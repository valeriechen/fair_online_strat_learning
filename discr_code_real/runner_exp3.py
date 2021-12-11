''' Benchmark No1: 
    EXP3 Algo treating each action as an arm.
    The loss func is of the form: f(a) = ell(a, vr_t(a)). 
'''
import sys
import numpy as np
''' Benchmark No1: 
    EXP3 Algo treating each action as an arm.
    The loss func is of the form: f(a) = ell(a, vr_t(a), y_t). 
'''

import random
import math
from copy import deepcopy
from principal import *
from agent import *
from regret import *
from oracle import *

def main_exp3(principal, agents, oracle, resp_lst, curr_rep, T, num_agents, d): 
    temp_regr = []
    algo_loss = []
    # resp_lst: |calA| x T
    for t in range(T):
        (a_t,arm_chosen)    = principal.choose_action()
        resp                = agents[t].response(a_t, d)  
        principal.loss_func[t]   = oracle.compute_loss(resp_lst,t)
        estimated_loss      = 1.0*principal.loss_func[t][arm_chosen]/principal.pi[arm_chosen]
        principal.est_loss[arm_chosen] += estimated_loss
        arr = np.array([(-principal.eta_exp3)*principal.est_loss[i] for i in range(principal.calA_size)], dtype=np.float128)
        principal.weights = np.exp(arr)
        principal.pi = [principal.weights[i]/sum(principal.weights) for i in range(principal.calA_size)]
        # prevent division by almost 0
        for j in range(principal.calA_size):
            if (principal.pi[j] < 0.00000001): 
                principal.pi[j] = 0.00000001
    
        algo_loss.append(principal.loss_func[t][arm_chosen])
        temp_regr.append(regret(principal.loss_func, principal.calA, algo_loss, t))

    return temp_regr 
