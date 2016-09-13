import numpy as np 
from scipy import optimize 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# Our Data
trainX = np.array(([3, 5], [5, 1], [10, 2], [6, 1.5]), dtype = float)
trainy = np.array(([75], [82], [93], [70]), dtype = float)

testX = np.array(([4, 5.5], [4.5, 1], [9, 2.5], [6, 2]), dtype = float)
testy = np.array(([70], [89], [85], [75]), dtype = float)

# Normalisation
trainX = trainX/np.amax(trainX, axis = 0)
trainy = trainy/100.

testX = testX/np.amax(trainX, axis = 0)
testy = testy/100.

class Neural_Network(object):

	def __init__(self, Lambda=0):

		# Define Hyperparameters
		self.inputLayerSize  = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		# Weights (Parameters)
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

		# Regularization Parameter
		self.Lambda = Lambda

	def forward(self, X):
		# Propagate inputs through network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat    = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		return 1./(1. + np.exp(-z))

	def sigmoidPrime(self, z):
		# Derivative of the Sigmoid Function
		return np.exp(-z)/(1+np.exp(-z))**2

	def costFunction(self, X, y):
		self.yHat = self.forward(X)
		J = 0.5*sum((y-self.yHat)**2)/float(X.shape[0]) + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
		return J

	def costFunctionPrime(self, X, y):
		# Compute derivative with respect to W1 and W2
		self.yHat = self.forward(X)

		delta3    = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2     = np.dot(self.a2.T, delta3)/float(X.shape[0]) + self.Lambda*self.W2

		delta2    = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1     = np.dot(X.T, delta2)/float(X.shape[0]) + self.Lambda*self.W1

		return dJdW1, dJdW2

	def getParams(self):
		# Get W1 and W2 rolled into vector
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params

	def setParams(self, params):
		# Set W1 and W2 using single parameter vector

		W1_start = 0
		W1_end   = self.hiddenLayerSize * self.inputLayerSize
		self.W1  = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))

		W2_end   = W1_end + self.hiddenLayerSize * self.outputLayerSize
		self.W2  = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize,self.outputLayerSize))

	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

	def computeNumericalGradient(self, N, X, y):

		paramsInitial = N.getParams()
		numgrad       = np.zeros(paramsInitial.shape)
		perturb       = np.zeros(paramsInitial.shape)
		e = 1E-4

		for p in range(len(paramsInitial)):
			# Set perturbation vector
			perturb[p] = e
			N.setParams(paramsInitial + perturb)
			loss2 = N.costFunction(X, y)

			N.setParams(paramsInitial - perturb)
			loss1 = N.costFunction(X, y)

			# Compute Numerical Gradient
			numgrad[p] = (loss2 - loss1)/(2*e)

			# Return the value we changed back to zero
			perturb[p] = 0

		# Return Params to original value    
		N.setParams(paramsInitial)

		return numgrad

class trainer(object):

	def __init__(self, N):
		# Make local reference to Neural Networks
		self.N = N

	def costFunctionWrapper(self, params, X, y):
		self.N.setParams(params)
		cost = self.N.costFunction(X, y)
		grad = self.N.computeGradients(X, y)
		return cost, grad

	def callBackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.X, self.y))
		self.testJ.append(self.N.costFunction(self.testX, self.testy))

	def train(self, trainX, trainy, testX, testy):
		self.X = trainX
		self.y = trainy

		self.testX = testX
		self.testy = testy

		# Make empty list to store costs
		self.J     = []
		self.testJ = []

		params0 = self.N.getParams()
		options = {'maxiter':200, 'disp':True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = 'BFGS', args = (trainX,trainy), options = options, callback = self.callBackF)

		self.N.setParams(_res.x)
		self.optimizationResults = _res



NN   = Neural_Network(Lambda = 0.001)

T = trainer(NN)
T.train(trainX, trainy, testX, testy)

#print NN.forward(X)

hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)

# Normalise Data (same way data was normalised)
hoursSleepNorm = hoursSleep/10.0
hoursStudyNorm = hoursStudy/5.0

# Create a 2D version of input for plotting
a, b = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

# Join into a single input matrix
allInputs = np.zeros((a.size, 2))
allInputs[:,0] = a.ravel()
allInputs[:,1] = b.ravel()

allOutputs = NN.forward(allInputs)

yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

plt.figure()
CS = plt.contour(xx, yy, 100*allOutputs.reshape(100,100))
plt.clabel(CS, inline = 1, fontsize = 10)
plt.xlabel('Hours Sleep')
plt.ylabel('Hours Study')
plt.show()

fig = plt.figure()
ax = plt.gca(projection = '3d')
ax.scatter(10*trainX[:,0], 5*trainX[:,1], 100*trainy, c='k', alpha = 1, s=30)
surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100), cmap = plt.get_cmap('rainbow'), linewidth=0, alpha = 0.5)
ax.set_xlabel('Hours Sleep')
ax.set_ylabel('Hours Study')
ax.set_zlabel('Test Score')
plt.show()

plt.figure()
plt.plot(T.J, label = 'Train')
plt.plot(T.testJ, label = 'Test')
plt.legend()
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

'''
numgrad = NN.computeNumericalGradient(NN, X, y)
grad   = NN.computeGradients(X,y)
print grad
print numgrad
print '-'*20

print np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

cost1 = NN.costFunction(X,y)
dJdW1, dJdW2 = NN.costFunctionPrime(X,y)

scalar = 3.0

NN.W1 = NN.W1 - scalar*dJdW1
NN.W2 = NN.W2 - scalar*dJdW2
cost2 = NN.costFunction(X,y)

print cost1
print '-'*10
print cost2 
'''


