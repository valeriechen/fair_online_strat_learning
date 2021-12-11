''' Class implementing the agents' behavior '''

import numpy as np
import random 
import math
import cvxpy as cp

class Agent(object):
    def __init__(self, agent_id, agent_type, x_real, delta, subg):
        self.id             = agent_id
        self.type           = agent_type[agent_id]
        self.delta          = delta
        self.x_real         = x_real[agent_id] 
        self.label          = 1 if self.type else -1
        self.subgroup       = subg[agent_id]

    def response(self, a, d): 
        inn_prod  = np.dot(a,self.x_real)
        dist = 1.0*inn_prod/np.linalg.norm(a[:d])  #d = dimension

        if inn_prod >= 0: #classify 1, irrespective of ground truth
            return self.x_real #no need to manipulate
        else: 
            if np.abs(dist) <= self.delta: #manipulation possible 
                x1 = cp.Variable(1)
                x2 = cp.Variable(1)

                objective = cp.Minimize((self.x_real[0] - x1)**2 + (self.x_real[1] - x2)**2) #impose a more complex cost function between groups
                constraints = [a[0]*x1 + a[1]*x2 + a[2] >= 0.0001] 
                prob = cp.Problem(objective, constraints)   
                result = prob.solve()
                resp = [float(x1.value), float(x2.value), 1]
                
                # Code below corresponds to second utility function
                #resp = [self.x_real[i] +1.0*(np.abs(dist)+0.00001)*a[i]/np.linalg.norm(a[:d]) if a[i]!=0 else self.x_real[i] for i in range(d)]
                #resp = np.append(resp, [1])
                return resp
            else: 
                return self.x_real

