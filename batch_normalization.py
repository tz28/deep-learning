# implement the batch normalization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  load_breast_cancer
from sklearn.model_selection import train_test_split


#initialize parameters(w,b)
def initialize_parameters(layer_dims):
	"""
	:param layer_dims: list,每一层单元的个数（维度）
			gamma -- scale vector of shape (size of current layer ,1)
            beta -- offset vector of shape (size of current layer ,1)
	:return: parameter: directory store w1,w2,...,wL,b1,...,bL
			 bn_param: directory store moving_mean, moving_var
	"""
	np.random.seed(3)
	L = len(layer_dims)#the number of layers in the network
	parameters = {}
	# initialize the exponential weight average
	bn_param = {}
	for l in range(1,L):
		parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.1
		parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
		parameters["gamma" + str(l)] = np.ones((layer_dims[l],1))
		parameters["beta" + str(l)] = np.zeros((layer_dims[l],1))
		bn_param["moving_mean" + str(l)] = np.zeros((layer_dims[l], 1))
		bn_param["moving_var" + str(l)] = np.zeros((layer_dims[l], 1))

	return parameters, bn_param

def relu_forward(Z):
	"""
	:param Z: Output of the linear layer
	:return:
	A: output of activation
	"""
	A = np.maximum(0,Z)
	return A

#implement the activation function(ReLU and sigmoid)
def sigmoid_forward(Z):
	"""
	:param Z: Output of the linear layer
	:return:
	"""
	A = 1 / (1 + np.exp(-Z))
	return A

def linear_forward(X, W, b):
	z = np.dot(W, X) + b
	return z

def batchnorm_forward(z, gamma, beta, epsilon = 1e-12):
	"""
	:param z: the input of activation (z = np.dot(W,A_pre) + b)
	:param epsilon: is a constant for denominator is 0
	:return: z_out, mean, variance
	"""
	mu = np.mean(z, axis=1, keepdims=True)#axis=1按行求均值
	var = np.var(z, axis=1, keepdims=True)
	sqrt_var = np.sqrt(var + epsilon)
	z_norm = (z - mu) / sqrt_var
	z_out = np.multiply(gamma,z_norm) + beta #对应元素点乘
	return z_out, mu, var, z_norm, sqrt_var


def forward_propagation(X, parameters, bn_param, decay = 0.9):
	"""
	X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "gamma1","beta1",W2", "b2","gamma2","beta2",...,"WL", "bL","gammaL","betaL"
                    W -- weight matrix of shape (size of current layer, size of previous layer)
                    b -- bias vector of shape (size of current layer,1)
                    gamma -- scale vector of shape (size of current layer ,1)
                    beta -- offset vector of shape (size of current layer ,1)
                    decay -- the parameter of exponential weight average
                    moving_mean = decay * moving_mean + (1 - decay) * current_mean
                    moving_var = decay * moving_var + (1 - decay) * moving_var
                    the moving_mean and moving_var are used for test
    :return:
	AL: the output of the last Layer(y_predict)
	caches: list, every element is a tuple:(A, W,b,gamma,sqrt_var,z_out,Z_norm)
	"""
	L = len(parameters) // 4  # number of layer
	A = X
	caches = []
	# calculate from 1 to L-1 layer
	for l in range(1,L):
		W = parameters["W" + str(l)]
		b = parameters["b" + str(l)]
		gamma = parameters["gamma" + str(l)]
		beta = parameters["beta" + str(l)]
		z = linear_forward(A, W, b)
		z_out, mu, var, z_norm, sqrt_var = batchnorm_forward(z, gamma, beta) #batch normalization
		caches.append((A, W, b, gamma, sqrt_var, z_out, z_norm)) #以激活单元为分界线，把做激活前的变量放在一起，激活后可以认为是下一层的x了
		A = relu_forward(z_out) #relu activation function
		#exponential weight average for test
		bn_param["moving_mean" + str(l)] = decay * bn_param["moving_mean" + str(l)] + (1 - decay) * mu
		bn_param["moving_var" + str(l)] = decay * bn_param["moving_var" + str(l)] + (1 - decay) * var
	# calculate Lth layer(last layer)
	WL = parameters["W" + str(L)]
	bL = parameters["b" + str(L)]
	zL = linear_forward(A, WL, bL)
	AL = sigmoid_forward(zL)
	caches.append((AL, WL, bL, None, None, None, None))
	return AL, caches, bn_param

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

#derivation of relu
def relu_backward(dA, Z):
	"""
	:param Z: the input of activation function
	:return:
	"""
	dout = np.multiply(dA, np.int64(Z > 0))
	return dout

def batchnorm_backward(dout, cache):
	"""
	:param dout: Upstream derivatives
	:param cache:
	:return:
	"""
	_, _, _, gamma, sqrt_var, _, Z_norm = cache
	m = dout.shape[1]
	dgamma = np.sum(dout*Z_norm, axis=1, keepdims=True) #*作用于矩阵时为点乘
	dbeta = np.sum(dout, axis=1, keepdims=True)
	dy = 1./m * gamma * sqrt_var * (m * dout - np.sum(dout, axis=1, keepdims=True) - Z_norm*np.sum(dout*Z_norm, axis=1, keepdims=True))
	return dgamma, dbeta, dy

def linear_backward(dZ, cache):
	"""
	:param dZ: Upstream derivative, the shape (n^[l+1],m)
	:param A: input of this layer
	:return:
	"""
	A, W, _, _, _, _, _ = cache
	dW = np.dot(dZ, A.T)
	db = np.sum(dZ, axis=1, keepdims=True)
	da = np.dot(W.T, dZ)
	return da, dW, db

def backward_propagation(AL, Y, caches):
	"""
	Implement the backward propagation presented in figure 2.
	Arguments:
	Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
	caches -- caches output from forward_propagation(),(w,b,gamma,sqrt_var,z_out,Z_norm,A)

	Returns:
	gradients -- A dictionary with the gradients with respect to dW,db
	"""
	m = Y.shape[1]
	L = len(caches)-1
	# print("L:   " + str(L))
	#calculate the Lth layer gradients
	dz = 1./m * (AL - Y)
	da, dWL, dbL = linear_backward(dz, caches[L])
	gradients = {"dW"+str(L+1): dWL, "db"+str(L+1): dbL}
	#calculate from L-1 to 1 layer gradients
	for l in reversed(range(0,L)): # L-1,L-3,....,1
		#relu_backward->batchnorm_backward->linear backward
		A, w, b, gamma, sqrt_var, z_out, z_norm = caches[l]
		#relu backward
		dout = relu_backward(da,z_out)
		#batch normalization
		dgamma, dbeta, dz = batchnorm_backward(dout,caches[l])
		# print("===============dz" + str(l+1) + "===================")
		# print(dz.shape)
		#linear backward
		da, dW, db = linear_backward(dz,caches[l])
		# print("===============dw"+ str(l+1) +"=============")
		# print(dW.shape)
		#gradient
		gradients["dW" + str(l+1)] = dW
		gradients["db" + str(l+1)] = db
		gradients["dgamma" + str(l+1)] = dgamma
		gradients["dbeta" + str(l+1)] = dbeta
	return gradients

def update_parameters(parameters, grads, learning_rate):
	"""
	:param parameters: dictionary, W, b
	:param grads: dW,db,dgamma,dbeta
	:param learning_rate: alpha
	:return:
	"""
	L = len(parameters) // 4
	for l in range(L):
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l+1)]
		if l < L-1:
			parameters["gamma" + str(l + 1)] = parameters["gamma" + str(l + 1)] - learning_rate * grads["dgamma" + str(l + 1)]
			parameters["beta" + str(l + 1)] = parameters["beta" + str(l + 1)] - learning_rate * grads["dbeta" + str(l + 1)]
	return parameters

def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations):
	"""
	:param X:
	:param Y:
	:param layer_dims: list containing the input size and each layer size
	:param learning_rate:
	:param num_iterations:
	:return:
	parameters：final parameters:(W,b,gamma,beta)
	bn_param: moving_mean, moving_var
	"""
	costs = []
	# initialize parameters
	parameters, bn_param = initialize_parameters(layer_dims)
	for i in range(0, num_iterations):
		#foward propagation
		AL,caches,bn_param = forward_propagation(X, parameters,bn_param)
		# calculate the cost
		cost = compute_cost(AL, Y)
		if i % 1000 == 0:
			print("Cost after iteration {}: {}".format(i, cost))
			costs.append(cost)
		#backward propagation
		grads = backward_propagation(AL, Y, caches)
		#update parameters
		parameters = update_parameters(parameters, grads, learning_rate)
	print('length of cost')
	print(len(costs))
	plt.clf()
	plt.plot(costs)  # o-:圆形
	plt.xlabel("iterations(thousand)")  # 横坐标名字
	plt.ylabel("cost")  # 纵坐标名字
	plt.show()
	return parameters,bn_param

#fp for test
def forward_propagation_for_test(X, parameters, bn_param, epsilon = 1e-12):
	"""
	X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "gamma1","beta1",W2", "b2","gamma2","beta2",...,"WL", "bL","gammaL","betaL"
                    W -- weight matrix of shape (size of current layer, size of previous layer)
                    b -- bias vector of shape (size of current layer,1)
                    gamma -- scale vector of shape (size of current layer ,1)
                    beta -- offset vector of shape (size of current layer ,1)
                    decay -- the parameter of exponential weight average
                    moving_mean = decay * moving_mean + (1 - decay) * current_mean
                    moving_var = decay * moving_var + (1 - decay) * moving_var
                    the moving_mean and moving_var are used for test
    :return:
	AL: the output of the last Layer(y_predict)
	caches: list, every element is a tuple:(A, W,b,gamma,sqrt_var,z,Z_norm)
	"""
	L = len(parameters) // 4  # number of layer
	A = X
	# calculate from 1 to L-1 layer
	for l in range(1,L):
		W = parameters["W" + str(l)]
		b = parameters["b" + str(l)]
		gamma = parameters["gamma" + str(l)]
		beta = parameters["beta" + str(l)]
		z = linear_forward(A, W, b)
		#batch normalization
		# exponential weight average
		moving_mean = bn_param["moving_mean" + str(l)]
		moving_var = bn_param["moving_var" + str(l)]
		sqrt_var = np.sqrt(moving_var + epsilon)
		z_norm = (z - moving_mean) / sqrt_var
		z_out = np.multiply(gamma, z_norm) + beta  # 对应元素点乘
		#relu forward
		A = relu_forward(z_out) #relu activation function

	# calculate Lth layer(last layer)
	WL = parameters["W" + str(L)]
	bL = parameters["b" + str(L)]
	zL = linear_forward(A, WL, bL)
	AL = sigmoid_forward(zL)
	return AL



#predict function
def predict(X_test, y_test, parameters, bn_param):
	"""
	:param X:
	:param y:
	:param parameters:
	:return:
	"""
	m = y_test.shape[1]
	Y_prediction = np.zeros((1, m))
	prob = forward_propagation_for_test(X_test, parameters, bn_param)
	for i in range(prob.shape[1]):
		# Convert probabilities A[0,i] to actual predictions p[0,i]
		if prob[0, i] > 0.5:
			Y_prediction[0, i] = 1
		else:
			Y_prediction[0, i] = 0
	accuracy = 1- np.mean(np.abs(Y_prediction - y_test))
	return accuracy

#DNN model
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate= 0.001, num_iterations=30000):
	parameters, bn_param = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations)
	accuracy = predict(X_test,y_test,parameters,bn_param)
	return accuracy


if __name__ == "__main__":
	X_data, y_data = load_breast_cancer(return_X_y=True)
	X_train, X_test,y_train,y_test = train_test_split(X_data, y_data, train_size=0.8,test_size=0.2,random_state=28)
	X_train = X_train.T
	y_train = y_train.reshape(y_train.shape[0], -1).T
	X_test = X_test.T
	y_test = y_test.reshape(y_test.shape[0], -1).T
	accuracy = DNN(X_train,y_train,X_test,y_test,[X_train.shape[0],10,5,1])
	print(accuracy)