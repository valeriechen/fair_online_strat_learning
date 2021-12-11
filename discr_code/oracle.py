'''
Code includes both the omnipotent oracle and a regression one.
For now, the regression one hasnt been fixed.
'''

import numpy as np
from agent import *
import cvxpy as cp
import networkx as nx
from networkx.algorithms import approximation
import sys
sys.setrecursionlimit(10**6)

class Oracle(object):
    def __init__(self, agents_lst, calA, T):
        self.agents_lst = agents_lst
        self.calA       = calA
        self.calA_size  = len(calA)
        self.T          = T 
        self.G          = nx.Graph()

    # Returns a 2d matrix: |A|xT with the responses of all agents
    # for all potential principal's actions
    def compute_responses(self,d):
        resp_lst = [[np.array(7) for _ in range(self.T)] for _ in range(self.calA_size)]
        for i in range(self.calA_size):
            a = self.calA[i]
            for t in range(self.T): 
                curr_ag     = self.agents_lst[t]
                dist        = 1.0*np.dot(a,curr_ag.x_real)/np.linalg.norm(a[:d])  
                inn_prod    = np.dot(a,curr_ag.x_real) #sign of inner product gives estimated label
                if inn_prod >= 0 and curr_ag.type == 0: #classify 1, while should be -1
                    resp_lst[i][t] = curr_ag.x_real #no need to manipulate
               
                if inn_prod >= 0 and curr_ag.type == 1: #classify 1, should be 1
                    resp_lst[i][t] = curr_ag.x_real #no need to manipulate
                     
                if inn_prod <= 0: #classify -1, should be -1
                    if np.abs(dist) <= curr_ag.delta: 
                        x1 = cp.Variable(1)
                        x2 = cp.Variable(1)

                        objective = cp.Minimize((curr_ag.x_real[0] - x1)**2 + (curr_ag.x_real[1] - x2)**2)
                        constraints = [a[0]*x1 + a[1]*x2 + a[2] >= 0.0001]
                        prob = cp.Problem(objective, constraints)   
                        result = prob.solve()
                        resp_lst[i][t] = [x1.value, x2.value, 1]
                        # Different utility func
                        #resp = [curr_ag.x_real[j] +1.0*(np.abs(dist)+0.00001)*a[j]/np.linalg.norm(a[:d]) if a[j]!=0 else curr_ag.x_real[j] for j in range(d)]
                        #resp_lst[i][t] = np.append(resp, [1])
                    else: 
                        resp_lst[i][t] = curr_ag.x_real
        return resp_lst

    # total probability of action chosen to see a clean report
    def compute_in_probs(self,pi,d,resp_lst,t):
        in_pr       = [0 for i in range(self.calA_size)]
        for i in range(self.calA_size):
            a = self.calA[i]
            for j in range(self.calA_size):
                if (i == j):
                    in_pr[i] += pi[j]
                else: 
                    # compute distance of action a from best-response of the agent
                    # to action a_played
                    dist2 = 1.0*np.dot(a,resp_lst[j][t])/np.linalg.norm(a[:d])
                    if (np.abs(dist2) >= 2*self.agents_lst[0].delta):
                        in_pr[i]   += pi[j]
                        self.G.add_edge(i,j)
        return in_pr
    
    def compute_in_probs_regr(self, pi, d, t, updated, actions_taken, lower_bound, incl):
        beta        = 0.95
        time        = len(updated) # size of training sample
        # samples that are more recent are getting exponentially more weight
        weights     = np.array([np.power(beta, time - i - 1) for i in range(time)])
        #X           = np.array([np.concatenate((a, np.power(a,2), np.power(a,3)), axis=None) for a in actions_taken])
        X           = np.array(actions_taken)
        all_actions = np.array(self.calA)
        in_probs    = []
        
        # need a different logistic regression for every action
        for i in range(self.calA_size): 
            labels  = np.array([1 if updated[j][i] else 0 for j in range(time)])
            if (1 in labels and 0 in labels):
                logistic    = linear_model.LogisticRegression(solver='lbfgs')
                logistic.fit(X, labels, weights)
            
                #act     = np.array([np.concatenate((a, np.power(a, 2), np.power(a,3)), axis=None) for a in all_actions]) 
                act    = np.array(all_actions)

                output      = logistic.predict_proba(act)
                one         = output[:,1]
                one         = [1.0 if j == i else one[j] for j in range(self.calA_size)]
                #est         = sum([pi[j]*one[j] for j in range(self.calA_size)])
                est         = 0 
                for j in range(self.calA_size):
                    if one[j] >= 0.5:
                        est += pi[j]
                
                if incl[i] == 0:
                    #action i is not already in 2delta region
                    lb  = pi[i]
                else: 
                    lb  = lower_bound 
               
                in_probs.append(est if est>=lb else lb) 
            else: 
                in_probs.append(pi[i])
        return in_probs
    

    # it can only be an approximation algorithm
    def compute_independence_number(self, d, resp_lst):
        for t in range(self.T):
            indicator_vec   = [1]*self.calA_size
            fg              = self.compute_in_probs(indicator_vec, d, resp_lst, t)

        ind_set = approximation.maximum_independent_set(self.G)
        a_G = len(ind_set)
        return a_G
 
    def compute_loss(self,resp_lst,t):
        est_label   = [1.0*np.dot(self.calA[i], resp_lst[i][t]) for i in range(self.calA_size)]
        loss        = [1 if np.sign(est_label[i]*self.agents_lst[t].label) == -1 else 0 for i in range(self.calA_size)]
        return loss     

