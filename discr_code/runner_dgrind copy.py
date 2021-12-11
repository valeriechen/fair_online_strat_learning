'''
    Implementation of the GRINDER algorithm for discrete action set.
'''

import sys
import numpy as np
import random
import math
from copy import deepcopy
from principal import *
from agent import *
from regret import *
from oracle import *
from fairness_metric import *
import matplotlib.pyplot as plt 

def main_dgrind(regress, principal, agents, oracle, resp_lst, curr_rep, T, num_agents, d, subgroup):   
    temp_regr = []
    algo_loss = []
    actions_taken   = []
    updated         = []
    updated_curr    = [0]*principal.calA_size
    improvement_0 = []
    improvement_1 =[]
    fair_regret = []
    stack_regret = []
    total_fair_regret = 0
    total_stack_regret = 0
    lmbda = 0.5
    metric = 'improvement_gap'# 'social_gap'#  


    min_fair_loss, fairest_arm, fair_arm_index, ranking = find_optimal_rule(principal.calA, metric, d, agents[0].delta)


    print ("runner dgrind repetition: %d"%curr_rep)
    for t in range(T):
        print ("Timestep t=%d"%t)
        cp_probs  = deepcopy(principal.pi)
        a_t = fairest_arm
        arm_chosen = fair_arm_index
        #(a_t, arm_chosen) = principal.choose_action() 
        
        resp, original = agents[t].response(a_t, d)
        if subgroup[t]:
            improvement_1.append((np.dot(a_t,resp)>=0) > (np.dot(a_t,original)>=0)) # if new is at least as good as old?
        else:
            improvement_0.append((np.dot(a_t,resp)>=0) > (np.dot(a_t,original)>=0))

        
        #Pick which loss to do here:
        #principal.loss_func[t] = fairness_regret_t(principal.calA, principal.pi, metric, d, agents[0].delta)
        principal.loss_func[t] = oracle.compute_loss(resp_lst,t)

        #this needs to compute for each 
        #new_fair_regret = (fairness_regret_t(principal.calA, principal.pi, metric, d, agents[0].delta) - min_fair_loss)
        #total_fair_regret += new_fair_regret
        #print(total_fair_regret)
        #fair_regret.append(total_fair_regret)

        # new_stack_regret = oracle.compute_loss(resp_lst,t)
        # total_stack_regret += new_stack_regret
        # stack_regret.append(total_stack_regret)


        if (not regress):
            in_probs = oracle.compute_in_probs(cp_probs, d, resp_lst,t)
        else: # no omnipotent oracle, only a regression one is available
            actions_taken.append(principal.calA[arm_chosen]) # list containing all actions taken by principal
            updated_curr[arm_chosen] = 1 # flag for the current arm
            tot  = 0.0
            incl = [0]*principal.calA_size # unclear what it does at the moment
            for i in range(principal.calA_size): # iterate over all of principal's actions
                a = principal.calA[i]
                dist = 1.0*np.dot(a, resp)/np.linalg.norm(a[:d]) # compute distance of current point from action a
                if (i != arm_chosen): 
                    if (np.abs(dist) >= 2*agents[t].delta):
                        updated_curr[i] = 1
                        tot += principal.pi[i] # builds the dataset for the logistic regression
                        incl[i] = 1
                    else: 
                        updated_curr[i] = 0
                else: 
                    if np.abs(dist) >= 2*agents[t].delta:
                        tot += principal.pi[i]
                        incl[i] = 1
                    
            updated.append(updated_curr) 
            in_probs = oracle.compute_in_probs_regr(cp_probs, d, t, updated, actions_taken, tot, incl)

        estimated_loss    = (1.0*principal.loss_func[t][arm_chosen])/in_probs[arm_chosen]
        principal.est_loss[arm_chosen] += estimated_loss
        

        for i in range(principal.calA_size):
            a = principal.calA[i]
            if (i != arm_chosen):
                dist    = 1.0*np.dot(a, resp)/np.linalg.norm(a[:d])
                if np.abs(dist) >= 2*agents[t].delta: 
                    if (np.sign(dist*agents[t].label) == -1):
                        principal.est_loss[i] += (1.0)/in_probs[i] 
                    else: 
                        principal.est_loss[i] += (0.0)/in_probs[i] 
                        
        # according to whether you have access to omnipotent oracle or just a regression one, eta changes accordingly
        if (not regress):
            eta = principal.eta_dgrind
        else: 
            eta = principal.eta_dgrind_regress
        
        arr = np.array([(-eta)*principal.est_loss[i] for i in range(principal.calA_size)], dtype=np.float128)
        principal.weights = np.exp(arr) 
        #principal.weights = [(1-lmbda)*principal.weights[i] + lmbda*(1/ranking[i]) for i in range(principal.calA_size)]
        principal.pi = [principal.weights[i]/sum(principal.weights) for i in range(principal.calA_size)]

        # prevent division by almost 0
        for j in range(principal.calA_size):
            if (principal.pi[j] < 0.00000001): 
                principal.pi[j] = 0.00000001
        
        # cumulative utility for the actions playedin

        algo_loss.append(principal.loss_func[t][arm_chosen])
        temp_regr.append(regret(principal.loss_func, principal.calA, algo_loss, t))


    print("fairness", fair_regret)
    print("stackelberg", stack_regret)


    iters = [i+1 for i in range(T)]
    plt.plot(iters, temp_regr, "*-")
    plt.title("Stackelberg regret")
    plt.show()
    #plt.plot(iters, fair_regret, "*-")
    #plt.title("Fairness regret")
    #plt.show()



    #analyze improvement across groups
    
    #print(np.mean(np.array(improvement_0)), np.mean(np.array(improvement_1)))

    return temp_regr 
