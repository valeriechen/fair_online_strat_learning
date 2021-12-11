import numpy as np
import random
import math
from copy import deepcopy
from grinding_polytopes import * 
import polytope as pc
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


d = 2 # dimension of space
p = 0.7 
# variable is controlled by the outside file
delta = 0.05 #radius of best-responses -- implicitly affects regret
T = 1000
agent_type = [0]*T


agent_type = np.random.binomial(1,p,T)

true_labels = [1 if agent_type[i] else -1 for i in range(T)] 

#original feature vectors for agents
x_real = []
y = []
for i in range(T):
    if agent_type[i]:
        x_real.append(np.array([np.random.normal(0.7, 0.2), np.random.normal(0.7,0.2), 1]))
        y.append(0)
    else:
        x_real.append(np.array([np.random.normal(0.4, 0.2), np.random.normal(0.4,0.2), 1]))
        y.append(1)


x_real = np.array(x_real)

plt.scatter(x_real[:,0], x_real[:,1], c=y)
plt.show()

