'''
    Implementation of the GRINDER algorithm.
'''
import os 
import numpy as np
import random
import math
from copy import deepcopy
from principal_grind import *
from agent import *
from regret import *
from oracle_cont import *
from grinding_polytopes import *
from params import *
import polytope as pc

def compute_eta(pr, t, prob):
    min_vol = pr.smallest_polytope_vol()
    C2      = math.log(1.0*pr.calA_vol/min_vol,2)
    print ("estimated polytopes")
    print (1.0*pr.calA_vol/min_vol)
    vol_m   = sum(p.pol_repr.volume for p in pr.calP_m_gt)
    size_ul = len(pr.calP_u_gt) + len(pr.calP_l_gt)
    if min_vol != pr.calA_vol:
        if size_ul != 0:    
            C1  = math.log(1.0*(pr.calA_vol*size_ul*t/min_vol),2) + vol_m
        else: 
            C1  = math.log(1.0*(pr.calA_vol*t/min_vol),2) + vol_m
        
        if pr.C1 < C1:
            pr.C1 = C1


        eta = math.sqrt(1.0*C2/(t*C1))
    else: 
        eta = 0.5
    

    return eta

def main_grind(regress, principal, agents, oracle, curr_rep, T, num_agents, d, best_fixed, prob):   
    temp_regr = []
    algo_loss = []
    actions_taken   = []
    updated         = [] # list holding the set of updated polytopes at each timestep
    improvement = []

    print ("runner grind repetition: %d"%curr_rep)
    for t in range(T):
        print ("Timestep t=%d"%t)
        principal.eta_grind = compute_eta(principal, t+1, prob)
        # calP is the list with the currently active polytopes at timestep t
        # i.e., does not contain all polytopes from beginning of time
        weights_lst = [p.weight for p in principal.calP]
        pi_lst      = [p.pi for p in principal.calP]
        cp_probs    = deepcopy(pi_lst)
        gamma       = principal.eta_grind
        # pol_chosen is the index of the chosen polytope
        (a_t, pol_chosen) = principal.choose_action_grind(cp_probs, gamma)
        old = np.dot(a_t,agents[t].x_real)>=0
        resp    = agents[t].response(a_t, d)
        principal.algo_loss[t] = 1.0 if np.sign(1.0*np.dot(a_t, resp)*agents[t].label) == -1 else 0.0
        #principal.algo_loss[t] = 1.0 if (np.dot(a_t,resp) >= 0) > (np.dot(a_t,agents[t].x_real) >= 0) else 0.0
        new = np.dot(a_t,resp)>=0
        improvement.append(new>=old) # if new is at least as good as old?

        # build the three polytope sets for ground truth
        calP_gt = principal.calP_u_gt + principal.calP_m_gt + principal.calP_l_gt
        
        # compute the three polytope sets for the conservative, safe ball
        cp_resp1 = deepcopy(resp)
        cp_resp2 = deepcopy(resp)
        
        actions_taken.append(a_t)
        principal.calP[pol_chosen].updated[t] = 0
        (calP_u, calP_m, calP_l) = oracle.compute_calP_in_probs(4.0*np.sqrt(2), 2, deepcopy(principal.calP), t, cp_resp1, cp_resp2, actions_taken) 
            
        # concatenation of the above lists corresponds to the new polytopes list
        calP_new = calP_u + calP_m + calP_l
        principal.calP_u_gt = calP_u
        principal.calP_l_gt = calP_l
        principal.calP_m_gt = calP_m

        # create weights + probs
        arr                     = np.array([(-principal.eta_grind)*pol.est_loss for pol in calP_new], dtype=np.float128)
        weights_grind           = np.exp(arr)
        pi_grind                = [weights_grind[i]/sum(weights_grind) for i in range(len(calP_new))]
        principal.calP          = calP_new #update list including new polytopes
        for i in range(len(principal.calP)): 
            p           = principal.calP[i]
            p.weight    = weights_grind[i]
        for i in range(len(principal.calP)): 
            p           = principal.calP[i]
            p.pi        = pi_grind[i]
        
        
        algo_loss.append(principal.algo_loss[t])
        # instead, we will use the sequence of best fixed actions defined by EXP3 algo
        temp_regr.append(compute_regret_grind(algo_loss, best_fixed[curr_rep][t]))

    return temp_regr, improvement

