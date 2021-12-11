import numpy as np
import random 
import math
import cvxpy as cp
from probability import *
import matplotlib.pyplot as plt 

def plot_decision_boundary():

	xs1 = []
	xs2 = []

	xs1neg = []
	xs1pos = []
	xs2neg = []
	xs2pos = []

	pred1 = []
	pred2 = []
	ys = []
	groups = []
	# groups2 = []

	#rule = [0.62093175, -0.78386463, -0.1841558] #(for social gap)
	#rule = [ 0.71898738,  0.69502313, -0.32273545] # (for improvement gap)
	#rule = [ 0.92182761, -0.38760014, -0.43144894] # normal
	#rule = [ 0.77103397, -0.63679401, -0.18191191] #normal
	#rule = [ 0.8582664,  -0.51320442, -0.01424547]
	rule = [ 0.9157181,   0.40182129, -0.50501726]

	for i in range(100):
		groups.append(0)
		x,y = sample_from_group(0)
		xs1.append([x[0], x[1],1])

		if np.sign(np.dot(rule, np.array([x[0], x[1],1]))) == 1.0:
			xs1pos.append([x[0], x[1],1])
		else:
			xs1neg.append([x[0], x[1],1])

		ys.append(y)

	for i in range(100):
		groups.append(1)
		x,y = sample_from_group(1)
		xs2.append([x[0], x[1],1])
		
		if np.sign(np.dot(rule, np.array([x[0], x[1],1]))) == 1.0:
			xs2pos.append([x[0], x[1],1])
		else:
			xs2neg.append([x[0], x[1],1])

		#pred2.append(np.sign(np.dot(rule, np.array([x[0], x[1],1]))) == 1.0)
		ys.append(y)

	xs1pos = np.array(xs1pos)
	xs2pos = np.array(xs2pos)

	xs1neg = np.array(xs1neg)
	xs2neg = np.array(xs2neg)


	if xs1neg.shape[0] > 0:
		plt.scatter(xs1neg[:,0], xs1neg[:,1], color = 'red', marker='.', label='Group A')
	if xs1pos.shape[0] > 0:
		plt.scatter(xs1pos[:,0], xs1pos[:,1], color = 'red', marker='+', label='Group B')
	if xs2neg.shape[0] > 0:
		plt.scatter(xs2neg[:,0], xs2neg[:,1], color = 'blue', marker='.', label='Group A')
	if xs2pos.shape[0] > 0:
		plt.scatter(xs2pos[:,0], xs2pos[:,1], color = 'blue', marker='+', label='Group B')
	
	#plt.axvline(x=0.35, color='black')
	plt.legend()
	plt.show()

def plot_groups():

	xs1 = []
	xs2 = []
	ys = []
	groups = []
	# groups2 = []

	for i in range(100):
		groups.append(0)
		x,y = sample_from_group(0)
		xs1.append([x[0], x[1]])
		ys.append(y)

	for i in range(100):
		groups.append(1)
		x,y = sample_from_group(1)
		xs2.append([x[0], x[1]])
		ys.append(y)

	xs1 = np.array(xs1)
	xs2 = np.array(xs2)
	plt.scatter(xs1[:,0], xs1[:,1], label='Group A')
	plt.scatter(xs2[:,0], xs2[:,1], label='Group B')
	plt.axvline(x=0.35, color='black')
	plt.legend()
	plt.show()

def sample_from_group(group_num):
	if group_num:
		temp = np.array([np.random.normal(0.4, 0.1), np.random.normal(0.4,0.1), 1])
	else:
		temp = np.array([np.random.normal(0.2, 0.1), np.random.normal(0.2,0.1), 1])

	if temp[0] > 0.35:
		y = 1
	else:
		y = 0
	
	return temp, y


def find_optimal_rule(all_rules, metric, d, delta):

	results = []

	new_results = []

	for i in range(len(all_rules)):

		if metric == 'social_gap':
			res = calculate_socialgap(all_rules[i],d, delta)
			results.append(res)
			new_results.append([i, res])

		elif metric == 'improvement_gap':
			res = calculate_improvementgap(all_rules[i], d, delta)
			results.append(res)
			new_results.append([i, res])

		else:
			print("metric not defined")
			return None

	
	new_results.sort(key = lambda x: x[1]) #, reverse=True (DO NOT REVERSE!) 
	rankings = [0]*len(all_rules)
	for i in range(len(all_rules)):
		rankings[new_results[i][0]] = i+1

	print(rankings)

	min_rule = results.index(min(results))
	#print("HERE", all_rules[min_rule])

	return min(results), all_rules[min_rule], min_rule, rankings # don't need index?


def best_response(x_real, a, d, delta): 
    inn_prod  = np.dot(a, x_real)
    dist = 1.0*inn_prod/np.linalg.norm(a[:d])  

    if inn_prod >= 0: #classify 1, irrespective of ground truth
        return x_real, x_real #no need to manipulate
    else: 
        if np.abs(dist) <= delta: #manipulation possible 
            x1 = cp.Variable(1)
            x2 = cp.Variable(1)

            objective = cp.Minimize((x_real[0] - x1)**2 + (x_real[1] - x2)**2)
            constraints = [a[0]*x1 + a[1]*x2 + a[2] >= 0.0001]
            prob = cp.Problem(objective, constraints)   
            result = prob.solve()
            resp = [x1.value[0], x2.value[0], 1]
            return resp, x_real
        else: 
            return x_real, x_real

def fairness_regret_t(all_rules, prob_over_rules, metric, d, delta):
	
	results = []
	for i in range(100):
		
		#sample from all rules using prob over rules
		#ind = np.random.choice(len(all_rules), 1, p=prob_over_rules)
		ind = draw(prob_over_rules, 0)
		rule = all_rules[ind]

		#calculate metric

		if metric == 'social_gap':
			results.append(calculate_socialgap(rule,d, delta))

		elif metric == 'improvement_gap':
			results.append(calculate_improvementgap(rule, d, delta))
		else:
			print("metric not defined")
			return None

	return np.mean(np.array(results))
	

def calculate_socialgap(rule, d, delta):

	group_1_costs = []

	for i in range(100):

		x,y = sample_from_group(0)
		x_prime, x_orig = best_response(x, rule, d, delta)

		if y == 1:
			cost = (x[0] - x_prime[0])**2 + (x[1] - x_prime[1])**2
			group_1_costs.append(cost)

	avg1 = np.mean(np.array(group_1_costs))


	group_2_costs = []

	for i in range(100):
		x,y = sample_from_group(1)
		x_prime, x_orig = best_response(x, rule, d, delta)

		if y == 1:
			cost = (x[0] - x_prime[0])**2 + (x[1] - x_prime[1])**2
			group_2_costs.append(cost)

	avg2 = np.mean(np.array(group_2_costs))

	if math.isnan(avg1):
		avg1 = 0
	if math.isnan(avg2):
		avg2 = 0
	
	return avg2 - avg1 #add epsilon tolerance?



def calculate_improvementgap(rule, d, delta):

	group_1_gain = []

	for i in range(100):
		x,y = sample_from_group(0)
		x_prime, x_orig = best_response(x, rule, d, delta)

		group_1_gain.append((np.dot(rule,x_prime)>=0) > (np.dot(rule,x)>=0))
		#group_1_gain.append(np.dot(rule,x_prime) - np.dot(rule,x))

	avg1 = np.mean(np.array(group_1_gain))

	group_2_gain = []

	for i in range(100):
		x,y = sample_from_group(1)
		x_prime, x_orig = best_response(x, rule, d, delta)

		group_2_gain.append((np.dot(rule,x_prime)>=0) > (np.dot(rule,x)>=0))
		#group_2_gain.append(np.dot(rule,x_prime) - np.dot(rule,x))

	avg2 = np.mean(np.array(group_2_gain))


	return avg2 - avg1 #> 0 #turning this into indicator, but why?


#plot_decision_boundary()
# plot_groups()
