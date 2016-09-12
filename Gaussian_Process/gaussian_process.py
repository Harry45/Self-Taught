import numpy as np 
import matplotlib.pylab as plt 
from numpy import linalg as LA
from numpy.linalg import inv
from numpy.linalg import det
from functools import reduce

# Plotting conditions
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Palatino']})
fontSize = 20
figSize  = (10, 6)

'''
Explanation on kernel types:

squared exponential : two parameters - width and sigma
periodic random     : one parameter  - width
exponential         : one parameter  - width
gamma exponential   : two parameters - width and gamma
rational quadratic  : two parameters - width and alpha
mattern             : one parameter  - width
'''

class kernels:

	def __init__(self):
		pass

	def SquaredExponential(self, dist, s, w):
		return s**2 * np.exp((-0.5/w**2)*(dist)**2)

	def periodicRandom(self, dist, w):
		return np.exp(-(2./w**2) * (np.sin(0.5*dist))**2)

	def exponential(self, dist, w):
		return np.exp(-(dist/w))

	def gammaExponential(self, dist, w, gamma):
		return np.exp(-((dist/w)**gamma))

	def rationalQuadratic(self, dist, w, alpha):
		return (1. + (dist**2)/(2. * alpha * w**2))**(-alpha)

def gaussianProcess(x, f, newPoints, sigmaNoise, kernelType, **kwargs):

	kernel = kernels()

	# Kernel Matrix for the Test Points
	Matrix1 = np.zeros(shape = (len(x), len(x)))
	for i in range(len(x)): Matrix1[:,i] = x - x[i]
	
	# Working with the Train Points
	Matrix2 = np.zeros(shape = (len(x), len(newPoints)))
	for i in range(len(newPoints)): Matrix2[:,i] = x - newPoints[i]

	if kernelType == 'periodicRandom':
		width       = float(kwargs['width'])
		K           = kernel.periodicRandom(Matrix1, width) + 1E-6
		eigenValues = LA.eigvals(K)
		if np.all(eigenValues > 0) == True:
			print 'The Matrix is positive-definite'
		else:
			print 'The Matrix is NOT positive-definite'
		K_star      = kernel.periodicRandom(Matrix2, width)
		K_star_star = np.identity(len(newPoints))*kernel.periodicRandom((newPoints - newPoints),width)

	if kernelType == 'SquaredExponential':
		sigma       = float(kwargs['sigma'])
		width       = float(kwargs['width'])
		K           = kernel.SquaredExponential(Matrix1, sigma, width) + 1E-6
		eigenValues = LA.eigvals(K)
		if np.all(eigenValues > 0) == True:
			print 'The Matrix is positive-definite'
		else:
			print 'The Matrix is NOT positive-definite'
		K_star      = kernel.SquaredExponential(Matrix2, sigma, width)
		K_star_star = np.identity(len(newPoints))*kernel.SquaredExponential((newPoints - newPoints),sigma,width)

	if kernelType == 'exponential':
		width       = float(kwargs['width'])
		K           = kernel.exponential(Matrix1, width) + 1E-6
		eigenValues = LA.eigvals(K)
		if np.all(eigenValues > 0) == True:
			print 'The Matrix is positive-definite'
		else:
			print 'The Matrix is NOT positive-definite'
		K_star      = kernel.exponential(Matrix2, width)
		K_star_star = np.identity(len(newPoints))*kernel.exponential((newPoints - newPoints),width)

	if kernelType == 'gammaExponential':
		width       = float(kwargs['width'])
		gamma       = float(kwargs['gamma'])
		K           = kernel.gammaExponential(Matrix1, width, gamma) + 1E-6
		eigenValues = LA.eigvals(K)
		if np.all(eigenValues > 0) == True:
			print 'The Matrix is positive-definite'
		else:
			print 'The Matrix is NOT positive-definite'
		K_star      = kernel.gammaExponential(Matrix2, width, gamma)
		K_star_star = np.identity(len(newPoints))*kernel.gammaExponential((newPoints - newPoints),width, gamma)

	if kernelType == 'rationalQuadratic':
		width       = float(kwargs['width'])
		alpha       = float(kwargs['alpha'])
		K           = kernel.rationalQuadratic(Matrix1, width, alpha) + 1E-6
		eigenValues = LA.eigvals(K)
		if np.all(eigenValues > 0) == True:
			print 'The Matrix is positive-definite'
		else:
			print 'The Matrix is NOT positive-definite'
		K_star      = kernel.rationalQuadratic(Matrix2, width, alpha)
		K_star_star = np.identity(len(newPoints))*kernel.rationalQuadratic((newPoints - newPoints),width, alpha)

	L        = np.linalg.cholesky(K + sigmaNoise * np.identity(len(x)))
	alpha    = reduce(np.dot, [inv(L).T, inv(L), f])
	E_mean   = np.dot(K_star.T, alpha)
	v        = np.dot(inv(L), K_star)
	variance = K_star_star - np.dot(v.T, v)

	return E_mean, variance

def trueFunction(x):
	return np.sin(x)

numberTestPoint = 10
xmin = 0.0
xmax = 2.0*np.pi

# True Signal
xTrue  = np.linspace(xmin, xmax, 1E4)
yTrue  = trueFunction(xTrue)
xTest  = np.linspace(xmin, xmax, numberTestPoint)#np.random.uniform(xmin, xmax, numberTestPoint)##np.array([0.1, 0.9, 1.6, 3.5, 5.2, 6.0])#
xTrain = np.linspace(xmin, xmax, 1E3)
sigma_noise = 0.2

# Noiseless Gaussian Process
yTest       = trueFunction(xTest)
yTrain, var = gaussianProcess(xTest, yTest, xTrain, 0.0, 'SquaredExponential', sigma = 1.0, width = 1.0)
Sigma       = np.sqrt(np.diag(var))

fill1 = np.concatenate([xTrain, xTrain[::-1]])
fill2 = np.concatenate([yTrain - 1.9600 * Sigma, (yTrain + 1.9600 * Sigma)[::-1]])

# Noisy Gaussian Process
yTest_noisy             = trueFunction(xTest) + sigma_noise * np.random.randn(numberTestPoint)
yTrain_noisy, var_noisy = gaussianProcess(xTest, yTest_noisy, xTrain, sigma_noise, 'SquaredExponential', sigma = 1.0, width = 1.0)
Sigma_noisy             =  np.sqrt(np.diag(var_noisy))

fill1_noisy = np.concatenate([xTrain, xTrain[::-1]])
fill2_noisy = np.concatenate([yTrain_noisy - 1.9600 * Sigma_noisy, (yTrain_noisy+ 1.9600 * Sigma_noisy)[::-1]])

left   = 0.125  # the left side of the subplots of the figure
right  = 0.9    # the right side of the subplots of the figure
bottom = 0.1    # the bottom of the subplots of the figure
top    = 0.9    # the top of the subplots of the figure
wspace = None   # the amount of width reserved for blank space between subplots
hspace = 0.40   # the amount of height reserved for white space between subplots

plt.figure(figsize = figSize)

plt.subplot(211)
plt.scatter(xTest, yTest)
plt.plot(xTrue, yTrue, c = 'k', label = 'True Function')
plt.plot(xTrain, yTrain, '--', c = 'r', label = 'Gaussian Process')
plt.fill(fill1, fill2, alpha=0.4, fc='c', ec='None')
plt.legend(loc = 'best')
plt.xlabel("$x$", fontsize=fontSize)
plt.ylabel("$y$", fontsize=fontSize)
plt.tick_params(axis='both', which='major', labelsize=fontSize)
plt.xlim(xmin-0.1, xmax+0.1)

plt.subplot(212)
plt.errorbar(xTest, yTest_noisy, yerr = sigma_noise, fmt = 'none')
plt.scatter(xTest, yTest_noisy)
plt.plot(xTrue, yTrue, c = 'k', label = 'True Function')
plt.plot(xTrain, yTrain_noisy, '--', c = 'r', label = 'Gaussian Process')
plt.fill(fill1_noisy, fill2_noisy, alpha=0.4, fc='c', ec='None')
plt.legend(loc = 'best')
plt.xlabel("$x$", fontsize=fontSize)
plt.ylabel("$y$", fontsize=fontSize)
plt.tick_params(axis='both', which='major', labelsize=fontSize)
plt.xlim(xmin-0.1, xmax+0.1)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
#plt.savefig('/Users/Harry/Desktop/example_1_uniform.pdf', bbox_inches='tight')
plt.show()



