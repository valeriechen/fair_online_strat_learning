''' Driver file '''

from principal import *
from agent import *
from oracle import *
from copy import deepcopy
from master_file import  regret_dgrind, regret_exp3
from params import set_params
import math

num_repetitions = 30
dgrind  = [] 
exp3    = []
min_num_rounds = 0
max_num_rounds = 1000
step = 5
rounds = [T for T in range(min_num_rounds,max_num_rounds)] #size: 5000 bytes

T = max_num_rounds
(num_agents, dim, x_real, calA, agent_types, true_labels, delta, noise, p) = set_params(T)

cp_xreal  = deepcopy(x_real)
for delta in [0.05, 0.10, 0.15, 0.3, 0.5]:
    print ("Current delta = %.5f"%delta)
    agents_dgrind   = [Agent(t, agent_types, cp_xreal, delta) for t in range(T)]
    oracle_dgrind   = Oracle(deepcopy(agents_dgrind), calA, T) 
    resp_lst_dgrind = oracle_dgrind.compute_responses(dim)
    print ("Independence Number")
    # computes approximate independence number in order to tune \eta for GRINDER correctly
    a_G             = oracle_dgrind.compute_independence_number(dim, resp_lst_dgrind)
    print a_G

    principal_dgrind        = [Principal(T, calA, num_repetitions, p, a_G) for _ in range(0, num_repetitions)] 
    principal_dgrind_regr  = [Principal(T, calA, num_repetitions, p, a_G) for _ in range(0, num_repetitions)] 
    principal_exp3          = [Principal(T, calA, num_repetitions, p, a_G) for _ in range(0, num_repetitions)]

    agents_exp3         = [Agent(t, agent_types, cp_xreal, delta) for t in range(T)]
    agents_dgrind_regr  = [Agent(t, agent_types, cp_xreal, delta) for t in range(T)]

    oracle_exp3         = Oracle(deepcopy(agents_exp3), calA, T) 
    resp_lst_exp3       = deepcopy(resp_lst_dgrind)
    oracle_dgrind_regr  = Oracle(deepcopy(agents_dgrind_regr), calA, T) 
    resp_lst_dgrind_regr= deepcopy(resp_lst_dgrind)


    (dgrind, dgrind_regrets) = regret_dgrind(0, principal_dgrind, agents_dgrind, oracle_dgrind, resp_lst_dgrind, T, num_repetitions, num_agents, dim)  
    (exp3, exp3_regrets)     = regret_exp3(principal_exp3, agents_exp3, oracle_exp3, resp_lst_exp3, T, num_repetitions, num_agents, dim)  
    (dgrind_regress, dgrind_regrets_regress) = regret_dgrind(1, principal_dgrind_regr, agents_dgrind_regr, oracle_dgrind_regr, resp_lst_dgrind_regr, T, num_repetitions, num_agents, dim)  


