import numpy as np
import random
import math
from copy import deepcopy

def set_params(T): 
    np.random.seed()
    d = 2 # dimension of space
    p = 0.7 # probability of being truthful agent (aka non-spammer) 
    delta = 0.1 #radius of best-responses -- implicitly affects regret

    sub_group = []
    for i in range(T):
        choice = random.choices([0,1], k=1)
        sub_group.append(choice[0])

    true_labels = []
    agent_type = []
    #original feature vectors for agents
    x_real = []
    for i in range(T):
        
        if sub_group[i]:
            temp = np.array([np.random.normal(0.4, 0.2), np.random.normal(0.4,0.2), 1])
        else:
            temp = np.array([np.random.normal(0.2, 0.1), np.random.normal(0.2,0.1), 1])

        if temp[0] > 0.35:
            agent_type.append(1)
        else:
            agent_type.append(0)
        x_real.append(temp)

    # principal's action space -- for now, discrete
    calA_size   = 100
    eps = 1.0/5     

    initial = []
    zero = np.array([0, 0, 1])
    one  = np.array([1, 1, 1])
    curr_size = 0

    while curr_size < calA_size:
        temp  = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)])
        dist0 = np.abs(1.0*np.dot(temp,zero)/np.linalg.norm(temp[:d]))  
        dist1 = np.abs(1.0*np.dot(temp,one)/np.linalg.norm(temp[:d]))  
        if dist0 <= np.sqrt(2) and dist1 <= np.sqrt(2):
            initial.append(temp)
            curr_size += 1

    calA_size = len(initial)

    print ("Actions")
    print (initial)
    print ("Number of actions")
    print (calA_size)
    
    calA        = [init/np.linalg.norm(init[:d]) for init in initial]
    noise = []
    return (T, d, x_real, calA, agent_type, true_labels, delta, noise, p, sub_group)
    
