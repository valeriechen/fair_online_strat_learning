''' File that handles the parallel running of each instance.
    In the end, collects the results of each instance and 
    outputs them to separate txt files.
'''
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

def regret_dgrind(regress, principal, agents, oracle, resp_lst, T, num_reps, num_agents, dim):
    if not regress: 
        f1 = "dgrind_regrets%.2f.txt"%agents[0].delta
    else: 
        f1 = "dgrind_regrets_regress%.2f.txt"%agents[0].delta
    dgrind_regrets = open(f1, "w")
    pool = Pool(processes = num_reps)
    if not regress: 
        results = [pool.apply_async(main_dgrind, args = (regress, principal[rep], agents, oracle, resp_lst, rep, T, num_agents, dim), callback = log_dgrind_results) for rep in range(num_reps)]
    else:
        results = [pool.apply_async(main_dgrind, args = (regress, principal[rep], agents, oracle, resp_lst, rep, T, num_agents, dim), callback = log_dgrind_regress_results) for rep in range(num_reps)]
    
    pool.close()
    pool.join()   

    if regress == 1:
        regrs = dgrind_regr_regress
    else: 
        regrs = dgrind_regr

    #print ("DGrind Regrets from results")
    #print dgrind_regr
    dgrind_expected_regr = []
    for t in range(T):
        dgrind_expected_regr.append((1.0*sum(z[t] for z in regrs))/num_reps)

    #print ("DGrind Regrets returned to master_file:")
    #print dgrind_expected_regr
    if agents[0].delta == 0.05:
        for r in range(num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            dgrind_regrets.write(s)
    elif agents[0].delta == 0.1:
        for r in range(num_reps, 2*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            dgrind_regrets.write(s)
    elif agents[0].delta == 0.15:
        for r in range(2*num_reps, 3*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            dgrind_regrets.write(s)
    elif agents[0].delta == 0.3:
        for r in range(3*num_reps, 4*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            dgrind_regrets.write(s)
    else:
        for r in range(4*num_reps, 5*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%regrs[r][t])
            s += "\n"
            dgrind_regrets.write(s)
    

    final_dgrind_regr = dgrind_expected_regr
    dgrind_regrets.close()
    return (final_dgrind_regr, regrs)


def regret_exp3(principal, agents, oracle, resp_lst, T, num_reps, num_agents, dim):
    f1 = "exp3_regrets%.2f.txt"%agents[0].delta
    exp3_regrets = open(f1, "w")
    
    pool = Pool(processes = num_reps)
    results = [pool.apply_async(main_exp3, args = (principal[rep], agents, oracle, resp_lst, rep, T, num_agents, dim), callback = log_exp3_results) for rep in range(num_reps)]
    

    pool.close()
    pool.join()   

    #print ("EXP3 Regrets from results")
    #print exp3_regr
    exp3_expected_regr = []
    for t in range(T):
        exp3_expected_regr.append((1.0*sum(z[t] for z in exp3_regr))/num_reps)

    #print ("EXP3 Regrets returned to master_file:")
    #print exp3_expected_regr

    if agents[0].delta == 0.05:
        for r in range(num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%exp3_regr[r][t])
            s += "\n"
            exp3_regrets.write(s)
    elif agents[0].delta == 0.1:
        for r in range(num_reps, 2*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%exp3_regr[r][t])
            s += "\n"
            exp3_regrets.write(s)
    elif agents[0].delta == 0.15:
        for r in range(2*num_reps, 3*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%exp3_regr[r][t])
            s += "\n"
            exp3_regrets.write(s)
    elif agents[0].delta == 0.3:
        for r in range(3*num_reps, 4*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%exp3_regr[r][t])
            s += "\n"
            exp3_regrets.write(s)
    else:
        for r in range(4*num_reps, 5*num_reps):
            s = ""
            for t in range(T):
                s += ("%.5f "%exp3_regr[r][t])
            s += "\n"
            exp3_regrets.write(s)
    

    final_exp3_regr = exp3_expected_regr
    exp3_regrets.close()
    return (final_exp3_regr, exp3_regr)

