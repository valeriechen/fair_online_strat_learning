''' Driver file '''

from principal import *
from agent import *
from oracle import *
from copy import deepcopy
from master_file import  regret_dgrind, regret_exp3
from params import set_params, set_params_new
import math

import numpy as np
import random
import math
from copy import deepcopy
from runner_dgrind import *
from runner_exp3 import *
from oracle import *
from multiprocessing import Pool

dgrind_regr         = []
exp3_regr           = []
dgrind_regr_regress = []

def log_dgrind_results(res):
    dgrind_regr.append(res)

def log_dgrind_regress_results(res):
    dgrind_regr_regress.append(res)

def log_exp3_results(res):
    exp3_regr.append(res)


num_repetitions = 1#30
dgrind  = [] 
exp3    = []
min_num_rounds = 0
max_num_rounds = 100#1000
step = 5
rounds = [T for T in range(min_num_rounds,max_num_rounds)] #size: 5000 bytes

T = max_num_rounds
(num_agents, dim, x_real, calA, agent_types, true_labels, delta, noise, p, subgroup, pop_X, pop_Y) = set_params_new(T)

cp_xreal  = deepcopy(x_real)
#for delta in [0.05, 0.10, 0.15, 0.3, 0.5]:
if __name__ == '__main__':


    delta = 0.05
    print ("Current delta = %.5f"%delta)
    agents_dgrind   = [Agent(t, agent_types, cp_xreal, delta) for t in range(T)]
    oracle_dgrind   = Oracle(deepcopy(agents_dgrind), calA, T) 
    resp_lst_dgrind = oracle_dgrind.compute_responses(dim)
    print ("Independence Number")
    # computes approximate independence number in order to tune \eta for GRINDER correctly
    #a_G             = oracle_dgrind.compute_independence_number(dim, resp_lst_dgrind)
    #print(a_G)
    a_G = 1 #sub this back later

    principal_dgrind        = [Principal(T, calA, num_repetitions, p, a_G) for _ in range(0, num_repetitions)] 
    #principal_dgrind_regr  = [Principal(T, calA, num_repetitions, p, a_G) for _ in range(0, num_repetitions)] 
    #principal_exp3          = [Principal(T, calA, num_repetitions, p, a_G) for _ in range(0, num_repetitions)]

    #agents_exp3         = [Agent(t, agent_types, cp_xreal, delta) for t in range(T)]
    #agents_dgrind_regr  = [Agent(t, agent_types, cp_xreal, delta) for t in range(T)]

    #oracle_exp3         = Oracle(deepcopy(agents_exp3), calA, T) 
    #resp_lst_exp3       = deepcopy(resp_lst_dgrind)
    #oracle_dgrind_regr  = Oracle(deepcopy(agents_dgrind_regr), calA, T) 
    #resp_lst_dgrind_regr= deepcopy(resp_lst_dgrind)


    #(dgrind, dgrind_regrets) = regret_dgrind(0, principal_dgrind, agents_dgrind, oracle_dgrind, resp_lst_dgrind, T, num_repetitions, num_agents, dim)  
    #(exp3, exp3_regrets)     = regret_exp3(principal_exp3, agents_exp3, oracle_exp3, resp_lst_exp3, T, num_repetitions, num_agents, dim)  
    #(dgrind_regress, dgrind_regrets_regress) = regret_dgrind(1, principal_dgrind_regr, agents_dgrind_regr, oracle_dgrind_regr, resp_lst_dgrind_regr, T, num_repetitions, num_agents, dim)  


    #copy regret_dgrind here:

    regress = 0
    principal = principal_dgrind
    agents = agents_dgrind
    oracle = oracle_dgrind
    resp_lst = resp_lst_dgrind
    num_reps = num_repetitions
    #p = prob

    #all_regrets = []
    #prob = [1, 1, 1, 1, 1]
    #for p in prob:
    p = 1
    regr = main_dgrind(regress, principal[0], agents, oracle, resp_lst, 0, T, num_agents, dim, subgroup, pop_X, pop_Y, p)
    all_regrets.append(regr)

    # iters = [i+1 for i in range(T)]
    # plt.plot(iters, all_regrets[0], "-", label='p=0')
    # plt.plot(iters, all_regrets[1], "-", label='p=0.25')
    # plt.plot(iters, all_regrets[2], "-", label='p=0.5')
    # plt.plot(iters, all_regrets[3], "-", label='p=0.75')
    # plt.plot(iters, all_regrets[4], "-", label='p=1')
    # plt.legend()
    # plt.title("Stackelberg regret")
    # plt.show()

    # if not regress: 
    #     f1 = "dgrind_regrets%.2f.txt"%agents[0].delta
    # else: 
    #     f1 = "dgrind_regrets_regress%.2f.txt"%agents[0].delta
    # dgrind_regrets = open(f1, "w")
    # pool = Pool(processes = num_reps)
    # if not regress: 
    #     results = [pool.apply_async(main_dgrind, args = (regress, principal[rep], agents, oracle, resp_lst, rep, T, num_agents, dim, subgroup), callback = log_dgrind_results) for rep in range(num_reps)]
    # else:
    #     results = [pool.apply_async(main_dgrind, args = (regress, principal[rep], agents, oracle, resp_lst, rep, T, num_agents, dim, subgroup), callback = log_dgrind_regress_results) for rep in range(num_reps)]
    
    # pool.close()
    # pool.join()   

    # if regress == 1:
    #     regrs = dgrind_regr_regress
    # else: 
    #     regrs = dgrind_regr

    # #print ("DGrind Regrets from results")
    # #print dgrind_regr
    # dgrind_expected_regr = []
    # for t in range(T):
    #     dgrind_expected_regr.append((1.0*sum(z[t] for z in regrs))/num_reps)

    # #print ("DGrind Regrets returned to master_file:")
    # #print dgrind_expected_regr
    # if agents[0].delta == 0.05:
    #     for r in range(num_reps):
    #         s = ""
    #         for t in range(T):
    #             s += ("%.5f "%regrs[r][t])
    #         s += "\n"
    #         dgrind_regrets.write(s)
    # elif agents[0].delta == 0.1:
    #     for r in range(num_reps, 2*num_reps):
    #         s = ""
    #         for t in range(T):
    #             s += ("%.5f "%regrs[r][t])
    #         s += "\n"
    #         dgrind_regrets.write(s)
    # elif agents[0].delta == 0.15:
    #     for r in range(2*num_reps, 3*num_reps):
    #         s = ""
    #         for t in range(T):
    #             s += ("%.5f "%regrs[r][t])
    #         s += "\n"
    #         dgrind_regrets.write(s)
    # elif agents[0].delta == 0.3:
    #     for r in range(3*num_reps, 4*num_reps):
    #         s = ""
    #         for t in range(T):
    #             s += ("%.5f "%regrs[r][t])
    #         s += "\n"
    #         dgrind_regrets.write(s)
    # else:
    #     for r in range(4*num_reps, 5*num_reps):
    #         s = ""
    #         for t in range(T):
    #             s += ("%.5f "%regrs[r][t])
    #         s += "\n"
    #         dgrind_regrets.write(s)
    

    # final_dgrind_regr = dgrind_expected_regr
    # dgrind_regrets.close()



