''' Benchmark No1: 
    EXP3 Algo treating each action as an arm.
    The loss func is of the form: f(a) = ell(a, vr_t(a), y_t). 
'''
import sys
import numpy as np
import random
import math
from copy import deepcopy
from principal_exp3 import *
from agent import *
from regret import *
from oracle_cont import *

def main_exp3(principal, agents, oracle, resp_lst, curr_rep, T, num_agents, d): 
    temp_regr       = []
    temp_best_fixed = [] 
    algo_loss       = []
    # resp_lst: |calA| x T
    for t in range(T):
        # set eta to be different at each t
        principal.eta_exp3 = math.sqrt(1.0*(math.log(principal.calA_size_exp3,2))/((t+1.0)*principal.calA_size_exp3))
        (a_t,arm_chosen)                    = principal.choose_action_exp3()
        resp                                = agents[t].response(a_t, d)  
        principal.loss_func_exp3[t]         = oracle.compute_loss_exp3(resp_lst, t, principal.calA_exp3)
        estimated_loss                      = 1.0*principal.loss_func_exp3[t][arm_chosen]/principal.pi_exp3[arm_chosen]
        principal.est_loss_exp3[arm_chosen] += estimated_loss
        
        arr                     = np.array([(-principal.eta_exp3)*principal.est_loss_exp3[i] for i in range(principal.calA_size_exp3)], dtype=np.float128)
        principal.weights_exp3  = np.exp(arr)
        principal.pi_exp3       = [principal.weights_exp3[i]/sum(principal.weights_exp3) for i in range(principal.calA_size_exp3)]
        # prevent division by almost 0
        for j in range(principal.calA_size_exp3):
            if (principal.pi_exp3[j] < 0.0000000001): 
                principal.pi_exp3[j] = 0.0000000001
    
        algo_loss.append(principal.loss_func_exp3[t][arm_chosen])
        (regr, best_fixed) = compute_regret_exp3(principal.loss_func_exp3, principal.calA_exp3, algo_loss, t)
        temp_regr.append(regr)
        temp_best_fixed.append(best_fixed)

    return (temp_regr, temp_best_fixed) 
