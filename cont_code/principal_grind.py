''' Class for principal that uses GRINDER algorithm.'''
import numpy as np
import random 
import math
from params import * 
from probability import *
from oracle_cont import *
import polytope
from grinding_polytopes import *


class Principal_Grind(object): 
    # Only one principal in the setting
    def __init__(self, T, calP, num_repetitions):
        self.calP           = calP #list of polytopes 
        self.eta_grind      = 1.0 # just an initialization for now
        self.est_loss       = [] #size == tot_num_pols
        self.algo_loss      = [0.0]*T
        self.calA_vol       = 8.0
        self.calP_u_gt      = [] 
        self.calP_l_gt      = [] 
        self.calP_m_gt      = calP
        self.C1             = 0.0
        self.C2             = 1.0
        
    
    def smallest_polytope_vol(self):
        pol_vols = [pol.pol_repr.volume for pol in self.calP if pol.pol_repr.volume != 0]
        return np.min(pol_vols)
    
    def largest_polytope_vol(self):
        pol_vols = [pol.pol_repr.volume for pol in self.calP if pol.pol_repr.volume != 0]
        return np.max(pol_vols)

    # Implementa 2-stage action selection
    def choose_action_grind(self, pi_lst, gamma):
        # First choose polytope according to  
        # induced distr by grinding
        pol_index   = draw(pi_lst, gamma, self.calA_vol) #index of the chosen polytope
        pol_chosen  = self.calP[pol_index]
        action      = pol_chosen.action
   
        return (action, pol_index)
