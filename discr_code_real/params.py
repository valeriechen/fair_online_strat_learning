import numpy as np
import random
import math
from copy import deepcopy
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def read_dataset():

    df = pd.read_csv('UCI_Credit_Card.csv', delimiter=',')
    group1 = df[(df['EDUCATION']==1)]
    group2 = df[(df['EDUCATION']==3)]

    group1_data = group1.values
    group2_data = group2.values

    group_label = [0]*group1_data.shape[0] + [1]*group2_data.shape[0]

    new_data = np.concatenate((group1_data, group2_data), axis=0)
    X = new_data[:,:24]
    y = new_data[:,24]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_new = SelectKBest(chi2, k=5).fit_transform(X_scaled, y)

    temp = np.ones((X_new.shape[0],1))

    X_new = np.concatenate((X_new, temp), axis=1)

    # for i in range(5):
    #     print(np.mean(X_new[:group1_data.shape[0],i]), np.mean(X_new[group1_data.shape[0]:,i]))

    return X_new, y, group_label

def set_params_new(T):

    np.random.seed()
    d = 5 # dimension of space
    p = 0.7 # probability of being truthful agent (aka non-spammer) 
    delta = 0.1 #radius of best-responses -- implicitly affects regret

    X_new, y_new, group_label = read_dataset()

    true_labels = []
    agent_type = []
    #original feature vectors for agents
    x_real = []
    for i in range(T):
        
        ind = random.randint(0,len(group_label)-1)
        x_real.append(X_new[ind])

        #if sub_group[i]:
        #    temp = np.array([np.random.normal(0.4, 0.2), np.random.normal(0.4,0.2), 1])
        #else:
        #    temp = np.array([np.random.normal(0.2, 0.1), np.random.normal(0.2,0.1), 1])

        agent_type.append(y_new[ind])

        # if temp[0] > 0.35:
        #     agent_type.append(1)
        # else:
        #     agent_type.append(0)
        #x_real.append(temp)

    # principal's action space -- for now, discrete
    calA_size   = 100
    eps = 1.0/5     

    initial = []
    zero = np.array([0, 0, 0, 0, 0, 1])
    one  = np.array([1, 1, 1, 1, 1, 1])
    curr_size = 0

    while curr_size < calA_size:
        temp  = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)])
        dist0 = np.abs(1.0*np.dot(temp,zero)/np.linalg.norm(temp[:d]))  
        dist1 = np.abs(1.0*np.dot(temp,one)/np.linalg.norm(temp[:d]))  
        if dist0 <= np.sqrt(2) and dist1 <= np.sqrt(2): #unsure if need to change this...
            initial.append(temp)
            curr_size += 1

    calA_size = len(initial)

    print ("Actions")
    print (initial)
    print ("Number of actions")
    print (calA_size)
    
    calA        = [init/np.linalg.norm(init[:d]) for init in initial]
    noise = []
    return (T, d, x_real, calA, agent_type, true_labels, delta, noise, p, group_label, X_new, y_new)



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


