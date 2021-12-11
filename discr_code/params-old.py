import numpy as np
import random
import math
from copy import deepcopy

def set_params(T): 
    np.random.seed()
    d = 2 # dimension of space
    p = 0.7 # probability of being truthful agent (aka non-spammer)
    
    agent_type = [0]*T

    for i in range(T):
        if i <= T/2:
            agent_type[i] = 0
        else:
            agent_type[i] = 1    

    true_labels = [1 if agent_type[i] else -1 for i in range(T)] 
    delta = 0.7 #radius of best-responses -- implicitly affects regret

    #original feature vectors for agents
    #2d case for now
    x_real = []
    for i in range(T):
        if agent_type[i]:
            x_real.append(np.array([np.random.normal(0.6, 0.4), np.random.normal(0.4,0.6), 1]))
        else:
            x_real.append(np.array([np.random.normal(0.4, 0.6), np.random.normal(0.6,0.4), 1]))

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
    return (T, d, x_real, calA, agent_type, true_labels, delta, noise, p)
    
