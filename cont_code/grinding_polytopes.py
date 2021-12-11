''' Class implementing the split of one polytope to smaller ones.
    Takes care of information passing from parent polytope to child polytopes.
    For each newly created polytope it assigns a representative action through
    rejection sampling. 
'''

import numpy as np
import random
import polytope as pc

def gen_id():
    np.random.seed()
    return int(np.random.uniform(-100000000,100000000))



class Grind_Polytope(object):
    # start_lst: contains the start of the polytope 
    # for each dimension (so, size = d+1)
    # end_lst: contains the end of the polytope
    # for each dimension (so, size = d+1)
    def __init__(self, polytope, pi, weight, dim, T, est_loss, loss, updated):
        self.id         = gen_id()
        self.pol_repr   = polytope
        self.d          = dim
        self.pi         = pi
        self.weight     = weight
        self.updated    = updated #whether polytope got updated at timestep t or not
        self.est_loss   = est_loss
        self.loss       = loss
        self.action     = self.sample_action_within_pol()

    ''' 
        Returns an action within the polytope uniformly at random
        using Rejection Sampling.
    '''
    def sample_action_within_pol(self):
        pol         = self.pol_repr
        bbox        = pol.bounding_box 
         
        np.random.seed()
        action      = np.array([np.random.uniform(bbox[0][0], np.random.uniform(bbox[1][0])), 
                                np.random.uniform(bbox[0][1], np.random.uniform(bbox[1][1])), 
                                np.random.uniform(bbox[0][2], np.random.uniform(bbox[1][2])) 
                               ]).flatten()

        while action not in pol: 
            np.random.seed()
            action  = np.array([np.random.uniform(bbox[0][0], np.random.uniform(bbox[1][0])), 
                                np.random.uniform(bbox[0][1], np.random.uniform(bbox[1][1])), 
                                np.random.uniform(bbox[0][2], np.random.uniform(bbox[1][2])) 
                                ]).flatten()

        return action
