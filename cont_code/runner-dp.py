''' Driver file '''

from principal_exp3 import *
from principal_grind import *
from agent import *
from oracle_cont import *
from copy import deepcopy
from master_file import regret_grind, regret_exp3
from params import set_params
import math

import numpy as np
import random
import math
from copy import deepcopy
from runner_grind import *
from runner_exp3 import *
from oracle_cont import *
from multiprocessing import Pool

grind_regr         = []
exp3_regr          = []
best_fixed         = []
grind_regr_regress = []

def log_grind_results(res):
    grind_regr.append(res)

def log_grind_regress_results(res):
    grind_regr_regress.append(res)

def log_exp3_results(res):
    exp3_regr.append(res[0])
    best_fixed.append(res[1])

num_repetitions = 1#30
dgrind  = [] 
exp3    = []
min_num_rounds = 0
max_num_rounds = 1000
step = 5
#rounds = [T for T in range(min_num_rounds,max_num_rounds)] 

T = max_num_rounds
(num_agents, dim, x_real, calA_exp3, calA_grind, agent_types, true_labels, delta, noise, prob, subgroups) = set_params(T, 0.2)

cp_xreal       = deepcopy(x_real)
#for delta in [0.05]: #[0.05,0.1, 0.15, 0.3, 0.5]:

if __name__ == '__main__':

    delta = 0.05
    print ("Current delta = %.5f"%delta)
    agents_grind   = [Agent(t, agent_types, cp_xreal, delta, subgroups) for t in range(T)]
    oracle_grind   = Oracle(deepcopy(agents_grind), T) 

    principal_grind = [Principal_Grind(T, calA_grind, num_repetitions) for _ in range(0, num_repetitions)] 
    principal_exp3  = [Principal_Exp3(T, calA_exp3, num_repetitions, 0) for _ in range(0, num_repetitions)] 

    #agents_exp3     = [Agent(t, agent_types, cp_xreal, delta, subgroups) for t in range(T)]


    #oracle_exp3     = Oracle(deepcopy(agents_exp3), T) 
    #resp_lst_exp3   = oracle_exp3.compute_responses(deepcopy(calA_exp3), dim)

    #(exp3, exp3_regrets, best_fixed) = regret_exp3(principal_exp3, agents_exp3, oracle_exp3, resp_lst_exp3, T, num_repetitions, num_agents, dim)  
    best_fixed = [[0]*T for _ in range(num_repetitions)]


    regress = 1
    principal = principal_grind
    agents = agents_grind
    oracle = oracle_grind
    num_reps = num_repetitions
    p = prob
    
    f1 = "grind_regrets_regress%.2f.txt"%agents[0].delta
    grind_regrets = open(f1, "w")
    pool = Pool(processes = num_reps)
    if not regress: 
        results = [pool.apply_async(main_grind, args = (regress, principal[rep], agents, oracle, rep, T, num_agents, dim, best_fixed, p), callback = log_grind_results) for rep in range(num_reps)]
    else:
        results = [pool.apply_async(main_grind, args = (1, principal[rep], agents, oracle, rep, T, num_agents, dim, best_fixed, p), callback = log_grind_regress_results) for rep in range(num_reps)]
        
    
    pool.close()
    pool.join()   

    regrs = grind_regr_regress

    grind_expected_regr = []
    for t in range(T):
        grind_expected_regr.append((1.0*sum(z[t] for z in regrs))/num_reps)


    if agents[0].delta == 0.05:
        for r in range(num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            grind_regrets.write(s)
    elif agents[0].delta == 0.1:
        for r in range(num_reps, 2*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            grind_regrets.write(s)
    elif agents[0].delta == 0.15:
        for r in range(2*num_reps, 3*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            grind_regrets.write(s)
    elif agents[0].delta == 0.3:
        for r in range(3*num_reps, 4*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            grind_regrets.write(s)
    else:
        for r in range(4*num_reps, 5*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            grind_regrets.write(s)
    

    final_grind_regr = grind_expected_regr
    grind_regrets.close()

#(grind, grind_regrets) = regret_grind(1, principal_grind, agents_grind, oracle_grind, T, num_repetitions, num_agents, dim, best_fixed, prob)  
