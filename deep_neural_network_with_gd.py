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

def update_parameters(parameters, grads, learning_rate):
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

def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, gradient_descent = 'bgd',mini_batch_size = 64):
	"""
	:param X:
	:param Y:
	:param layer_dims:list containing the input size and each layer size
	:param learning_rate:
	:param num_iterations:
	:return:
	parameters：final parameters:(W,b)
	"""
	m = Y.shape[1]
	costs = []
	# initialize parameters
	parameters = initialize_parameters(layer_dims)
	if gradient_descent =='bgd':
		for i in range(0, num_iterations):
			#foward propagation
			AL,caches = forward_propagation(X, parameters)
			# calculate the cost
			cost = compute_cost(AL, Y)
			if i % 1000 == 0:
				print("Cost after iteration {}: {}".format(i, cost))
				costs.append(cost)
			#backward propagation
			grads = backward_propagation(AL, Y, caches)
			#update parameters
			parameters = update_parameters(parameters, grads, learning_rate)
	elif gradient_descent == 'sgd':
		np.random.seed(3)
		# 把数据集打乱，这个很重要
		permutation = list(np.random.permutation(m))
		shuffled_X = X[:, permutation]
		shuffled_Y = Y[:, permutation].reshape((1, m))
		for i in range(0, num_iterations):
			for j in range(0, m):  # 每次训练一个样本
				# Forward propagation
				AL,caches = forward_propagation(shuffled_X[:, j].reshape(-1,1), parameters)
				# Compute cost
				cost = compute_cost(AL, shuffled_Y[:, j].reshape(1,1))
				# Backward propagation
				grads = backward_propagation(AL, shuffled_Y[:,j].reshape(1,1), caches)
				# Update parameters.
				parameters = update_parameters(parameters, grads, learning_rate)
				if j % 20 == 0:
					print("example size {}: {}".format(j, cost))
					costs.append(cost)
	elif gradient_descent == 'mini-batch':
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
				parameters = update_parameters(parameters, grads, learning_rate)
			if i % 100 == 0:
				print("Cost after iteration {}: {}".format(i, cost))
				costs.append(cost)
	print('length of cost')
	print(len(costs))
	plt.clf()
	plt.plot(costs)
	plt.xlabel("iterations(hundred)")  # 横坐标名字
	plt.ylabel("cost")  # 纵坐标名字
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
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate= 0.0006, num_iterations=30000, gradient_descent = 'bgd',mini_batch_size = 64):
	parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations,gradient_descent,mini_batch_size)
	accuracy = predict(X_test,y_test,parameters)
	return accuracy

if __name__ == "__main__":
	X_data, y_data = load_breast_cancer(return_X_y=True)
	X_train, X_test,y_train,y_test = train_test_split(X_data, y_data, train_size=0.8,random_state=28)
	X_train = X_train.T
	y_train = y_train.reshape(y_train.shape[0], -1).T
	X_test = X_test.T
	y_test = y_test.reshape(y_test.shape[0], -1).T
	#use bgd
	accuracy = DNN(X_train,y_train,X_test,y_test,[X_train.shape[0],10,5,1])
	print(accuracy)
	#use sgd
	accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1],num_iterations=5, gradient_descent = 'sgd')
	print(accuracy)
	#mini-batch
	accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1], num_iterations=10000,gradient_descent='mini-batch')
	print(accuracy)