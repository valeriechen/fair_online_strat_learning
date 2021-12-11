''' Code for the oracles in the continuous case. 
    Includes both the way in which we update the polytopes
    and also the implementation of the regression oracles.
'''

import numpy as np
from agent import *
import polytope as pc
from grinding_polytopes import *
from sklearn import linear_model
import cvxpy as cp

class Oracle(object):
    def __init__(self, agents_lst, T):
        self.agents_lst     = agents_lst
        self.T              = T 
        self.spammers_lst   = [1 if self.agents_lst[t].type == 0 else 0 for t in range(T)]
    
    # used only for computing the responses used in exp3
    # Returns a 2d matrix: |A|xT with the responses of all agents
    # for all potential principal's actions
    def compute_responses(self, calA, d):
        resp_lst = [[np.array(7) for _ in range(self.T)] for _ in range(len(calA))]
        for i in range(len(calA)):
            a = calA[i]
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
                    else: 
                        resp_lst[i][t] = curr_ag.x_real
        return resp_lst
    
    def compute_loss_exp3(self, resp_lst, t, calA):
        est_label   = [1.0*np.dot(calA[i], resp_lst[i][t]) for i in range(len(calA))]
        loss        = [1 if np.sign(est_label[i]*self.agents_lst[t].label) == -1 else 0 for i in range(len(calA))]
        return loss    

    def is_empty(self, polytope):
        return (pc.is_empty(polytope) or (polytope.volume == 0.0))

    def has_tiny_vol(self, polytope):
        return (polytope.volume < 0.01)
    
    def compute_in_probs_regr(self, pi, t, updated, actions_taken, actions_set, lower_bound, incl):
        beta        = 0.95
        time        = len(actions_taken) # timesteps for which we have data
        # samples that are more recent are getting exponentially more weight
        weights     = np.array([np.power(beta, time - i - 1) for i in range(time)])
        X           = np.array([np.concatenate((a, np.power(a,2), np.power(a,3)), axis=None) for a in actions_taken])
        #X           = np.array(actions_taken)
        all_actions = np.array(actions_set)
        in_probs    = []
        
        # need a different logistic regression for every action
        for i in range(len(all_actions)): 
            labels = []
            for j in range(t + 1):
                labels.append(1 if updated[i][j] else 0)
            if len(labels) != time:
                print("problem")
                print(len(labels) < time)
            if (1 in labels and 0 in labels):
                logistic    = linear_model.LogisticRegression(solver='lbfgs')
                logistic.fit(X, labels, weights)
            
                act         = np.array([np.concatenate((a, np.power(a, 2), np.power(a,3)), axis=None) for a in all_actions]) 
                #act         = all_actions

                output      = logistic.predict_proba(act)
                one         = output[:,1]
                one         = [1.0 if j == i else one[j] for j in range(len(updated))]
                est         = 0 
                for j in range(len(updated)):
                    if one[j] >= 0.5:
                        est += pi[j]
                if incl[i] == 0:
                    lb = pi[i]
                else: 
                    lb = lower_bound
                in_probs.append(est if est >= lb else lb)
            else: 
                in_probs.append(pi[i])
        return in_probs

    def compute_calP_in_probs(self, c, d, calP, t, cp1, cp2, actions_taken):
        A_upper  = np.array([
                [-1.0,  0.0,  0.0],
                [ 1.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0],
                [ 0.0,  1.0,  0.0],
                [ 0.0,  0.0, -1.0],
                [ 0.0,  0.0,  1.0], #surrounding box
                [-cp1[0], -cp1[1], -cp1[2]]])

        b_upper  = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -c*self.agents_lst[t].delta])
        p_upper  = pc.Polytope(A_upper, b_upper) #large upper polytope  

        A_lower  = np.array([
                [-1.0,  0.0,  0.0],
                [ 1.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0],
                [ 0.0,  1.0,  0.0],
                [ 0.0,  0.0, -1.0],
                [ 0.0,  0.0,  1.0], #surrounding box
                [cp2[0], cp2[1], cp2[2]]])

        b_lower  = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -c*self.agents_lst[t].delta])
        p_lower  = pc.Polytope(A_lower, b_lower) #large lower polytope

        calP_u   = []
        calP_l   = []
        calP_m   = []

        print ("actual number of polytopes:")
        print (len(calP))

        for pol in calP:
            p               = pol.pol_repr
            if p.volume > 0.01:
                # intersections of the polytope with upper and lower of curr round
                intersect_up  = pc.intersect(p, p_upper)
                intersect_lo  = pc.intersect(p, p_lower)
    
                # check whether the intersection is empty-ish (very small vol)
                emptyish_up   = self.is_empty(intersect_up) or self.has_tiny_vol(intersect_up)
                emptyish_lo   = self.is_empty(intersect_lo) or self.has_tiny_vol(intersect_lo)

                if (emptyish_up and emptyish_lo):
                    calP_m.append(pol)
                elif (emptyish_up):
                    diff_lo  = p.diff(p_lower) # what is left in the diff should go in the middle space unless emptyish
                    if (self.is_empty(diff_lo) or self.has_tiny_vol(diff_lo)): # intersection is exactly the polytope
                        # if emptyish, no need to create new polytope
                        pol.updated[t] = 1
                        calP_l.append(pol)
                    else: 
                        # otherwise we need to create new polytope
                        upd_lst     = pol.updated #store the history of updates for curr pol
                        upd_lst[t]  = 0 # by default assume that you won't update it 
                        ratio       = 1.0*diff_lo.volume/p.volume # how big is the prob that you keep for new pol
                        g_pol1      = Grind_Polytope(diff_lo, ratio*pol.pi, ratio*pol.weight, d, self.T, pol.est_loss, pol.loss, upd_lst)
                        calP_m.append(g_pol1)

                        upd_lst2     = pol.updated 
                        upd_lst2[t]  = 1 # this part of the pol will be part of the calP_lower
                        g_pol2       = Grind_Polytope(intersect_lo, (1.0-ratio)*pol.pi, (1.0 - ratio)*pol.weight, d, self.T, pol.est_loss, pol.loss, upd_lst2)
                        calP_l.append(g_pol2)
                elif (emptyish_lo):
                    diff_up  = p.diff(p_upper)
                    if (self.is_empty(diff_up) or self.has_tiny_vol(diff_up)):
                        pol.updated[t] = 1
                        calP_u.append(pol)
                    else: 
                        upd_lst     = pol.updated
                        upd_lst[t]  = 0
                        ratio       = 1.0*diff_up.volume/p.volume
                        g_pol1      = Grind_Polytope(diff_up, ratio*pol.pi, ratio*pol.weight, d, self.T, pol.est_loss, pol.loss, upd_lst)
                        calP_m.append(g_pol1)
                        
                        upd_lst2     = pol.updated
                        upd_lst2[t]  = 1
                        g_pol2      = Grind_Polytope(intersect_up, (1.0-ratio)*pol.pi, (1.0 - ratio)*pol.weight, d, self.T, pol.est_loss, pol.loss, upd_lst2)
                        calP_u.append(g_pol2)
                else: 
                    diff_up  = p.diff(p_upper)
                    diff_lo  = p.diff(p_lower)
                    ratio1   = 1.0*intersect_up.volume/p.volume
                    upd_lst  = pol.updated
                    upd_lst[t] = 1
                    g_pol1   = Grind_Polytope(intersect_up, ratio1*pol.pi, ratio1*pol.weight, d, self.T, pol.est_loss, pol.loss, upd_lst)  
                    calP_u.append(g_pol1)
                    
                    ratio2   = 1.0*intersect_lo.volume/p.volume
                    g_pol2   = Grind_Polytope(intersect_lo, ratio2*pol.pi, ratio2*pol.weight, d, self.T, pol.est_loss, pol.loss, upd_lst)  
                    calP_l.append(g_pol2)
                    
                    #if ratio1 > 0 or ratio2 > 0:
                    diff_uplo = pc.intersect(diff_up, diff_lo)
                    if (not self.is_empty(diff_uplo) and not self.has_tiny_vol(diff_uplo)):
                        upd_lst[t] = 0
                        g_pol3   = Grind_Polytope(diff_uplo, (1.0 - ratio1 - ratio2)*pol.pi, (1.0 -ratio1 - ratio2)*pol.weight, d, self.T, pol.est_loss, pol.loss, upd_lst)  
                        calP_m.append(g_pol3)
            elif (p.volume == 0):
                # discard point-polytopes
                print ("Timestep t=%d polytope w zero vol"%t)
                print (p)
                print ("Was it really empty?")      
                print(pc.is_empty(p))
            else: 
                intersect_up    = pc.intersect(p, p_upper)
                intersect_lo    = pc.intersect(p, p_lower)
                if (self.is_empty(intersect_up) and self.is_empty(intersect_lo)):
                    pol.updated[t] = 0
                    calP_m.append(pol)
                elif (self.is_empty(intersect_lo) or intersect_lo.volume < intersect_up.volume):
                    # should go with upper polytopes set
                    pol.updated[t] = 1
                    calP_u.append(pol) 
                elif (self.is_empty(intersect_up) or intersect_up.volume < intersect_lo.volume):
                    # should go with lower polytopes set
                    pol.updated[t] = 1
                    calP_l.append(pol)
        
        for pol in calP_u:
            if pol.pi <= 0.000001:
                pol.pi = 0.000001
        for pol in calP_m:
            if pol.pi <= 0.000001:
                pol.pi = 0.000001
        for pol in calP_l:
            if pol.pi <= 0.000001:
                pol.pi = 0.000001
                
        
        actions_set = []
        upd         = [] 
        pi_lst      = []
        tot_up      = 0.0
        tot_lo      = 0.0
        incl        = [] # indicator function whether the current action belongs in an upper or lower polytope set
        for pol in calP_u:
            actions_set.append(pol.action)
            upd.append(pol.updated) #size |\calA| x T
            pi_lst.append(pol.pi)
            incl.append(1)
            tot_up += pol.pi

        for pol in calP_m:
            actions_set.append(pol.action)
            upd.append(pol.updated)
            incl.append(0)
            pi_lst.append(pol.pi)
            
        for pol in calP_l:
            actions_set.append(pol.action)
            upd.append(pol.updated)
            pi_lst.append(pol.pi)
            incl.append(1)
            tot_lo  += pol.pi
        
        # a lower bound in the in probabilities of the actions that are to be updated is
        # the tot prob of the upper and the lower polytopes sets
        in_probs_est = self.compute_in_probs_regr(pi_lst, t, upd, actions_taken, actions_set, tot_up + tot_lo, incl)

        spammer = 1 if self.agents_lst[t].type == 0 else 0
        j = 0
        for pol in calP_u:
            pol.est_loss += (1.0*spammer)/in_probs_est[j]
            j += 1

        for pol in calP_m:
            j += 1

        for pol in calP_l:
            pol.est_loss += (1.0 - spammer)/in_probs_est[j]
            j += 1
    
        return (calP_u, calP_m, calP_l)

