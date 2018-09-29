import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  load_breast_cancer
from sklearn.model_selection import train_test_split
#initialize parameters(w,b)
def initialize_parameters(layer_dims):
	"""
	:param layer_dims: list,每一层单元的个数（维度）
	:return:dictionary,存储参数w1,w2,...,wL,b1,...,bL
	"""
	np.random.seed(3)
	L = len(layer_dims)#the number of layers in the network
	parameters = {}
	for l in range(1,L):
		# parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1]) # he initialization
		# parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1])) #为了测试初始化为0的后果
		# parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1 / layer_dims[l - 1])  # xavier initialization
		parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
	return parameters
def relu(Z):
	"""
	:param Z: Output of the linear layer
	:return:
	A: output of activation
	"""
	A = np.maximum(0,Z)
	return A
#implement the activation function(ReLU and sigmoid)
def sigmoid(Z):
	"""
	:param Z: Output of the linear layer
	:return:
	"""
	A = 1 / (1 + np.exp(-Z))
	return A

def forward_propagation(X, parameters):
	"""
	X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2",...,"WL", "bL"
                    W -- weight matrix of shape (size of current layer, size of previous layer)
                    b -- bias vector of shape (size of current layer,1)
    :return:
	AL: the output of the last Layer(y_predict)
	caches: list, every element is a tuple:(W,b,z,A_pre)
	"""
	L = len(parameters) // 2  # number of layer
	A = X
	caches = [(None,None,None,X)]  # 第0层(None,None,None,A0) w,b,z用none填充,下标与层数一致，用于存储每一层的，w,b,z,A
	# calculate from 1 to L-1 layer
	for l in range(1,L):
		A_pre = A
		W = parameters["W" + str(l)]
		b = parameters["b" + str(l)]
		z = np.dot(W,A_pre) + b #计算z = wx + b
		A = relu(z) #relu activation function
		caches.append((W,b,z,A))
	# calculate Lth layer
	WL = parameters["W" + str(L)]
	bL = parameters["b" + str(L)]
	zL = np.dot(WL,A) + bL
	AL = sigmoid(zL)
	caches.append((WL,bL,zL,AL))
	return AL, caches
#calculate cost function
def compute_cost(AL,Y):
	"""
	:param AL: 最后一层的激活值，即预测值，shape:(1,number of examples)
	:param Y:真实值,shape:(1, number of examples)
	:return:
	"""
	m = Y.shape[1]
	# cost = -1.0/m * np.sum(Y*np.log(AL)+(1-Y)*np.log(1.0 - AL))#py中*是点乘
	# cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T)) #推荐用这个，上面那个容易出错
	cost = 1. / m * np.nansum(np.multiply(-np.log(AL), Y) +
	                          np.multiply(-np.log(1 - AL), 1 - Y))
	#从数组的形状中删除单维条目，即把shape中为1的维度去掉，比如把[[[2]]]变成2
	cost = np.squeeze(cost)
	# print('=====================cost===================')
	# print(cost)
	return cost

# derivation of relu
def relu_backward(Z):
	"""
	:param Z: the input of activation
	:return:
	"""
	dA = np.int64(Z > 0)
	return dA

def backward_propagation(AL, Y, caches):
	"""
	Implement the backward propagation presented in figure 2.
	Arguments:
	X -- input dataset, of shape (input size, number of examples)
	Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
	caches -- caches output from forward_propagation(),(W,b,z,pre_A)

	Returns:
	gradients -- A dictionary with the gradients with respect to dW,db
	"""
	m = Y.shape[1]
	L = len(caches) - 1
	# print("L:   " + str(L))
	#calculate the Lth layer gradients
	prev_AL = caches[L-1][3]
	dzL = 1./m * (AL - Y)
	# print(dzL.shape)
	# print(prev_AL.T.shape)
	dWL = np.dot(dzL, prev_AL.T)
	dbL = np.sum(dzL, axis=1, keepdims=True)
	gradients = {"dW"+str(L):dWL, "db"+str(L):dbL}
	#calculate from L-1 to 1 layer gradients
	for l in reversed(range(1,L)): # L-1,L-3,....,1
		post_W= caches[l+1][0] #要用后一层的W
		dz = dzL #用后一层的dz

		dal = np.dot(post_W.T, dz)
		z = caches[l][2]#当前层的z
		dzl = np.multiply(dal, relu_backward(z))
		prev_A = caches[l-1][3]#前一层的A
		dWl = np.dot(dzl, prev_A.T)
		dbl = np.sum(dzl, axis=1, keepdims=True)

		gradients["dW" + str(l)] = dWl
		gradients["db" + str(l)] = dbl
		dzL = dzl #更新dz
	return gradients

def update_parameters_with_gd(parameters, grads, learning_rate):
	"""
	:param parameters: dictionary,  W,b
	:param grads: dW,db
	:param learning_rate: alpha
	:return:
	"""
	L = len(parameters) // 2
	for l in range(L):
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l+1)]
	return parameters


def random_mini_batches(X, Y, mini_batch_size = 64, seed=1):
	"""
	Creates a list of random minibatches from (X, Y)
	Arguments:
	X -- input data, of shape (input size, number of examples)
	Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
	mini_batch_size -- size of the mini-batches, integer

	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""
	np.random.seed(seed)
	m = X.shape[1]  # number of training examples
	mini_batches = []

	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((1, m))

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = m // mini_batch_size  # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches


def initialize_velocity(parameters):
	"""
	Initializes the velocity as a python dictionary with:
				- keys: "dW1", "db1", ..., "dWL", "dbL"
				- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
	Arguments:
	parameters -- python dictionary containing your parameters.
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl

	Returns:
	v -- python dictionary containing the current velocity.
					v['dW' + str(l)] = velocity of dWl
					v['db' + str(l)] = velocity of dbl
	"""
	L = len(parameters) // 2  # number of layers in the neural networks
	v = {}
	# Initialize velocity
	for l in range(L):
		v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
		v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
	return v

#momentum
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
	"""
	Update parameters using Momentum
	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	v -- python dictionary containing the current velocity:
					v['dW' + str(l)] = ...
					v['db' + str(l)] = ...
	beta -- the momentum hyperparameter, scalar
	learning_rate -- the learning rate, scalar

	Returns:
	parameters -- python dictionary containing your updated parameters

	'''
	VdW = beta * VdW + (1-beta) * dW
	Vdb = beta * Vdb + (1-beta) * db
	W = W - learning_rate * VdW
	b = b - learning_rate * Vdb
	'''
	"""


	L = len(parameters) // 2  # number of layers in the neural networks

	# Momentum update for each parameter
	for l in range(L):
		# compute velocities
		v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
		v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
		# update parameters
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

	return parameters

#nesterov momentum
def update_parameters_with_nesterov_momentum(parameters, grads, v, beta, learning_rate):
	"""
	Update parameters using Momentum
	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	v -- python dictionary containing the current velocity:
					v['dW' + str(l)] = ...
					v['db' + str(l)] = ...
	beta -- the momentum hyperparameter, scalar
	learning_rate -- the learning rate, scalar

	Returns:
	parameters -- python dictionary containing your updated parameters
	v -- python dictionary containing your updated velocities

	'''
	VdW = beta * VdW - learning_rate * dW
	Vdb = beta * Vdb - learning_rate * db
	W = W + beta * VdW - learning_rate * dW
	b = b + beta * Vdb - learning_rate * db
	'''
	"""

	L = len(parameters) // 2  # number of layers in the neural networks

	# Momentum update for each parameter
	for l in range(L):
		# compute velocities
		v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
		v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
		# update parameters
		parameters["W" + str(l + 1)] += beta * v["dW" + str(l + 1)]- learning_rate * grads['dW' + str(l + 1)]
		parameters["b" + str(l + 1)] += beta * v["db" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

	return parameters


#AdaGrad initialization
def initialize_adagrad(parameters):
	"""
	Initializes the velocity as a python dictionary with:
				- keys: "dW1", "db1", ..., "dWL", "dbL"
				- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
	Arguments:
	parameters -- python dictionary containing your parameters.
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl

	Returns:
	Gt -- python dictionary containing sum of the squares of the gradients up to step t.
					G['dW' + str(l)] = sum of the squares of the gradients up to dwl
					G['db' + str(l)] = sum of the squares of the gradients up to db1
	"""
	L = len(parameters) // 2  # number of layers in the neural networks
	G = {}
	# Initialize velocity
	for l in range(L):
		G["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
		G["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
	return G

#AdaGrad
def update_parameters_with_adagrad(parameters, grads, G, learning_rate, epsilon = 1e-7):
	"""
	Update parameters using Momentum
	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	G -- python dictionary containing the current velocity:
					G['dW' + str(l)] = ...
					G['db' + str(l)] = ...
	learning_rate -- the learning rate, scalar
	epsilon -- hyperparameter preventing division by zero in adagrad updates

	Returns:
	parameters -- python dictionary containing your updated parameters

	'''
	GW += (dW)^2
	W -= learning_rate/sqrt(GW + epsilon)*dW
	Gb += (db)^2
	b -= learning_rate/sqrt(Gb + epsilon)*db
	'''
	"""

	L = len(parameters) // 2  # number of layers in the neural networks

	# Momentum update for each parameter
	for l in range(L):
		# compute velocities
		G["dW" + str(l + 1)] += grads['dW' + str(l + 1)]**2
		G["db" + str(l + 1)] += grads['db' + str(l + 1)]**2
		# update parameters
		parameters["W" + str(l + 1)] -= learning_rate / (np.sqrt(G["dW" + str(l + 1)]) + epsilon) * grads['dW' + str(l + 1)]
		parameters["b" + str(l + 1)] -= learning_rate / (np.sqrt(G["db" + str(l + 1)]) + epsilon) * grads['db' + str(l + 1)]

	return parameters


#initialize_adadelta
def initialize_adadelta(parameters):
	"""
	Initializes s and delta as two python dictionaries with:
				- keys: "dW1", "db1", ..., "dWL", "dbL"
				- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

	Arguments:
	parameters -- python dictionary containing your parameters.
					parameters["W" + str(l)] = Wl
					parameters["b" + str(l)] = bl

	Returns:
	s -- python dictionary that will contain the exponentially weighted average of the squared gradient of dw
					s["dW" + str(l)] = ...
					s["db" + str(l)] = ...
	v -- python dictionary that will contain the RMS
				v["dW" + str(l)] = ...
				v["db" + str(l)] = ...
	delta -- python dictionary that will contain the exponentially weighted average of the squared gradient of delta_w
					delta["dW" + str(l)] = ...
					delta["db" + str(l)] = ...

	"""

	L = len(parameters) // 2  # number of layers in the neural networks
	s = {}
	v = {}
	delta = {}
	# Initialize s, v, delta. Input: "parameters". Outputs: "s, v, delta".
	for l in range(L):
		s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
		s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
		v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
		v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
		delta["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
		delta["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

	return s, v, delta

#adadelta
def update_parameters_with_adadelta(parameters, grads, rho, s, v, delta, epsilon = 1e-6):
	"""
	Update parameters using Momentum
	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	rho -- decay constant similar to that used in the momentum method, scalar
	s -- python dictionary containing the current velocity:
					s['dW' + str(l)] = ...
					s['db' + str(l)] = ...
	delta -- python dictionary containing the current RMS:
					delta['dW' + str(l)] = ...
					delta['db' + str(l)] = ...

	epsilon -- hyperparameter preventing division by zero in adagrad updates

	Returns:
	parameters -- python dictionary containing your updated parameters

	'''
	Sdw = rho*Sdw + (1 - rho)*(dW)^2
	Sdb = rho*Sdb + (1 - rho)*(db)^2
	Vdw = sqrt((delta_w + epsilon) / (Sdw + epsilon))*dW
	Vdb = sqrt((delta_b + epsilon) / (Sdb + epsilon))*dW
	W -= Vdw
	b -= Vdb
	delta_w = rho*delta_w + (1 - rho)*(Vdw)^2
	delta_b = rho*delta_b + (1 - rho)*(Vdb)^2
	'''
	"""

	L = len(parameters) // 2  # number of layers in the neural networks
	# adadelta update for each parameter
	for l in range(L):
		# compute s
		s["dW" + str(l + 1)] = rho * s["dW" + str(l + 1)] + (1 - rho)*grads['dW' + str(l + 1)]**2
		s["db" + str(l + 1)] = rho * s["db" + str(l + 1)] + (1 - rho)*grads['db' + str(l + 1)]**2
		#compute RMS
		v["dW" + str(l + 1)] = np.sqrt((delta["dW" + str(l + 1)] + epsilon) / (s["dW" + str(l + 1)] + epsilon)) * grads['dW' + str(l + 1)]
		v["db" + str(l + 1)] = np.sqrt((delta["db" + str(l + 1)] + epsilon) / (s["db" + str(l + 1)] + epsilon)) * grads['db' + str(l + 1)]
		# update parameters
		parameters["W" + str(l + 1)] -= v["dW" + str(l + 1)]
		parameters["b" + str(l + 1)] -= v["db" + str(l + 1)]
		#compute delta
		delta["dW" + str(l + 1)] = rho * delta["dW" + str(l + 1)] + (1 - rho) * v["dW" + str(l + 1)] ** 2
		delta["db" + str(l + 1)] = rho * delta["db" + str(l + 1)] + (1 - rho) * v["db" + str(l + 1)] ** 2

	return parameters

#RMSprop
def update_parameters_with_rmsprop(parameters, grads, s, beta = 0.9, learning_rate = 0.01, epsilon = 1e-6):
	"""
	Update parameters using Momentum
	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	s -- python dictionary containing the current velocity:
					v['dW' + str(l)] = ...
					v['db' + str(l)] = ...
	beta -- the momentum hyperparameter, scalar
	learning_rate -- the learning rate, scalar

	Returns:
	parameters -- python dictionary containing your updated parameters
	'''
	SdW = beta * SdW + (1-beta) * (dW)^2
	sdb = beta * Sdb + (1-beta) * (db)^2
	W = W - learning_rate * dW/sqrt(SdW + epsilon)
	b = b - learning_rate * db/sqrt(Sdb + epsilon)
	'''
	"""
	L = len(parameters) // 2  # number of layers in the neural networks
	# rmsprop update for each parameter
	for l in range(L):
		# compute velocities
		s["dW" + str(l + 1)] = beta * s["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]**2
		s["db" + str(l + 1)] = beta * s["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]**2
		# update parameters
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)] / np.sqrt(s["dW" + str(l + 1)] + epsilon)
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads['db' + str(l + 1)] / np.sqrt(s["db" + str(l + 1)] + epsilon)

	return parameters

#initialize adam
def initialize_adam(parameters):
	"""
	Initializes v and s as two python dictionaries with:
				- keys: "dW1", "db1", ..., "dWL", "dbL"
				- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
	Arguments:
	parameters -- python dictionary containing your parameters.
					parameters["W" + str(l)] = Wl
					parameters["b" + str(l)] = bl
	Returns:
	v -- python dictionary that will contain the exponentially weighted average of the gradient.
					v["dW" + str(l)] = ...
					v["db" + str(l)] = ...
	s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
					s["dW" + str(l)] = ...
					s["db" + str(l)] = ...

	"""
	L = len(parameters) // 2  # number of layers in the neural networks
	v = {}
	s = {}
	# Initialize v, s. Input: "parameters". Outputs: "v, s".
	for l in range(L):
		v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
		v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
		s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
		s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

	return v, s

#adam
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
	"""
	Update parameters using Adam

	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	v -- Adam variable, moving average of the first gradient, python dictionary
	s -- Adam variable, moving average of the squared gradient, python dictionary
	learning_rate -- the learning rate, scalar.
	beta1 -- Exponential decay hyperparameter for the first moment estimates
	beta2 -- Exponential decay hyperparameter for the second moment estimates
	epsilon -- hyperparameter preventing division by zero in Adam updates

	Returns:
	parameters -- python dictionary containing your updated parameters
	"""

	L = len(parameters) // 2  # number of layers in the neural networks
	v_corrected = {}  # Initializing first moment estimate, python dictionary
	s_corrected = {}  # Initializing second moment estimate, python dictionary

	# Perform Adam update on all parameters
	for l in range(L):
		# Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
		v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
		v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
		# Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
		v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
		v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
		# Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
		s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
		s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
		# Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
		s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
		s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
		# Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)

	return parameters


def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, optimizer, beta = 0.9, beta2 = 0.999, mini_batch_size = 64, epsilon = 1e-8):
	"""
	:param X:
	:param Y:
	:param layer_dims:list containing the input size and each layer size
	:param learning_rate:
	:param num_iterations:
	:return:
	parameters：final parameters:(W,b)
	"""
	costs = []
	# initialize parameters
	parameters = initialize_parameters(layer_dims)
	if optimizer == "sgd":
		pass  # no initialization required for gradient descent
	elif optimizer == "momentum" or optimizer == "nesterov_momentum" or optimizer == "rmsprop":
		v = initialize_velocity(parameters)
	elif optimizer == "adagrad":
		G = initialize_adagrad(parameters)
	elif optimizer == "adadelta":
		s, v, delta = initialize_adadelta(parameters)
	elif optimizer == "adam":
		v, s = initialize_adam(parameters)
	t = 0 # initializing the counter required for Adam update
	seed = 0
	for i in range(0, num_iterations):
		# Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
		seed = seed + 1
		minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
		for minibatch in minibatches:
			# Select a minibatch
			(minibatch_X, minibatch_Y) = minibatch
			# Forward propagation
			AL, caches = forward_propagation(minibatch_X, parameters)
			# Compute cost
			cost = compute_cost(AL, minibatch_Y)
			# Backward propagation
			grads = backward_propagation(AL, minibatch_Y, caches)
			if optimizer == "sgd":
				parameters = update_parameters_with_gd(parameters, grads, learning_rate)
			elif optimizer == "momentum":
				parameters = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
			elif optimizer == "nesterov_momentum":
				parameters = update_parameters_with_nesterov_momentum(parameters, grads, v, beta, learning_rate)
			elif optimizer == "adagrad":
				parameters = update_parameters_with_adagrad(parameters,grads,G,learning_rate,epsilon)
			elif optimizer == "adadelta":
				parameters = update_parameters_with_adadelta(parameters,grads,beta,s,v,delta,epsilon)
			elif optimizer == "rmsprop":
				parameters = update_parameters_with_rmsprop(parameters, grads, v, beta, learning_rate, epsilon)
			elif optimizer == "adam":
				t += 1
				parameters = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta, beta2, epsilon)

		if i % 100 == 0:
			print("Cost after iteration {}: {}".format(i, cost))
			costs.append(cost)
	print('length of cost')
	print(len(costs))
	plt.clf()
	plt.plot(costs, label = optimizer)
	plt.xlabel("iterations(hundreds)")  # 横坐标名字
	plt.ylabel("cost")  # 纵坐标名字
	plt.legend(loc="best")
	plt.show()
	return parameters

#predict function
def predict(X_test,y_test,parameters):
	"""
	:param X:
	:param y:
	:param parameters:
	:return:
	"""
	m = y_test.shape[1]
	Y_prediction = np.zeros((1, m))
	prob, caches = forward_propagation(X_test,parameters)
	for i in range(prob.shape[1]):
		# Convert probabilities A[0,i] to actual predictions p[0,i]
		if prob[0, i] > 0.5:
			Y_prediction[0, i] = 1
		else:
			Y_prediction[0, i] = 0
	accuracy = 1- np.mean(np.abs(Y_prediction - y_test))
	return accuracy
#DNN model
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate= 0.0005, num_iterations=10000,optimizer = 'sgd', beta = 0.9, beta2 = 0.999, mini_batch_size = 64,epsilon = 1e-8):
	parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations, optimizer, beta, beta2, mini_batch_size, epsilon)
	accuracy = predict(X_test,y_test,parameters)
	return accuracy

if __name__ == "__main__":
	X_data, y_data = load_breast_cancer(return_X_y=True)
	X_train, X_test,y_train,y_test = train_test_split(X_data, y_data, train_size=0.8,random_state=28)
	X_train = X_train.T
	y_train = y_train.reshape(y_train.shape[0], -1).T
	X_test = X_test.T
	y_test = y_test.reshape(y_test.shape[0], -1).T
	# #mini-batch
	# accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1], num_iterations=10000)
	# print(accuracy)
	# # momentum
	# accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1], num_iterations=10000, optimizer='momentum')
	# print(accuracy)
	# nesterov momentum
	# accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1], learning_rate= 0.0001,num_iterations=10000,optimizer='nesterov_momentum')
	# print(accuracy)
	#adagrad
	# accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1], learning_rate= 0.01,num_iterations=10000,optimizer='adagrad')
	# print(accuracy)
	#adadelta
	# accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1],num_iterations=10000, beta= 0.9, epsilon=1e-6, optimizer='adadelta')
	# print(accuracy)
	# #RMSprop
	# accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1], learning_rate=0.001, num_iterations=10000, beta=0.9,epsilon=1e-6, optimizer='rmsprop')
	# print(accuracy)
	#adam
	accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1], learning_rate=0.001, num_iterations=10000, beta=0.9, beta2=0.999, epsilon=1e-8, optimizer='adam')
	print(accuracy)