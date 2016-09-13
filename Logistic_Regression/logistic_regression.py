import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt 
from numpy.linalg import inv
from functools import reduce
from numpy import linalg as LA

data = pd.read_csv('data.csv')

# Features
gre   = np.array(data['gre'])
gpa   = np.array(data['gpa'])
rank  = np.array(data['rank'])

# Outcome
admit = np.array(data['admit'])

# Create Train Set
Admit  = admit[0:len(gre)-5] 
X      = np.zeros(shape = (len(gre)-5, 4))
X[:,0] = np.ones(len(gre) - 5)
X[:,1] = gre[0:len(gre) - 5]
X[:,2] = gpa[0:len(gre) - 5]
X[:,3] = rank[0:len(gre) - 5]

# Create Test Set
X_test      = np.zeros(shape = (5, 4))
X_test[:,0] = np.ones(5)
X_test[:,1] = gre[len(gre)-5:]
X_test[:,2] = gpa[len(gre)-5:]
X_test[:,3] = rank[len(gre)-5:]
admit_test  = admit[len(gre)-5:] 

def classification(values):
	returnArray = np.zeros(len(values))
	for i in range(len(values)):
		if values[i]<=0.5:
			returnArray[i] = int(0)
		else: returnArray[i] = int(1)
	return returnArray

def sigmoid(x, theta):
	return 1./(1. + np.exp(-1. * np.dot(x, theta)))

# iteratively reweighted least squares
def irls(x, y):

	theta  = np.zeros(X.shape[1])	
	theta_ = np.inf

	while max(np.abs(theta - theta_))>1E-15:

		sig          = sigmoid(x, theta)
		S            = np.diag(sig * (1. - sig))
		hessian      = reduce(np.dot, [X.T, S, X])
		gradient     = np.dot(X.T, (sig - y))
		theta_       = theta
		theta        = theta - np.dot(inv(hessian), np.dot(X.T, (sig - y)))

	return theta 

# Train
theta_final = irls(X, Admit)

# Test

print classification(sigmoid(X_test, theta_final))
print admit_test








