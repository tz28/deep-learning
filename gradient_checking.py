import numpy as np
from sklearn.datasets import  load_breast_cancer
from sklearn.model_selection import train_test_split

#initialize parameters(w,b)
def initialize_parameters(layer_dims):
	"""
	:param layer_dims: list,每一层单元的个数（维度）
	:return:dictionary,存储参数w1,w2,...,wL,b1,...,bL
	"""
	np.random.seed(1)
	L = len(layer_dims)#the number of layers in the network
	parameters = {}
	for l in range(1,L):
		# parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		# parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1]) # he initialization
		# parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1])) #为了测试初始化为0的后果
		parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1 / layer_dims[l - 1])  # xavier initialization
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
	cost = 1. / m * np.nansum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y))
	#从数组的形状中删除单维条目，即把shape中为1的维度去掉，比如把[[[2]]]变成2
	cost = np.squeeze(cost)
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
	caches -- caches output from forward_propagation(),(W,b,z,A)

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
		dzl = np.multiply(dal, relu_backward(z))#可以直接用dzl = np.multiply(dal, np.int64(Al > 0))来实现
		prev_A = caches[l-1][3]#前一层的A
		dWl = np.dot(dzl, prev_A.T)
		dbl = np.sum(dzl, axis=1, keepdims=True)

		gradients["dW" + str(l)] = dWl
		gradients["db" + str(l)] = dbl
		dzL = dzl #更新dz
	return gradients

#convert parameter into vector
def dictionary_to_vector(parameters):
	"""
	Roll all our parameters dictionary into a single vector satisfying our specific required shape.
	"""
	count = 0
	for key in parameters:
		# flatten parameter
		new_vector = np.reshape(parameters[key], (-1, 1))#convert matrix into vector
		if count == 0:#刚开始时新建一个向量
			theta = new_vector
		else:
			theta = np.concatenate((theta, new_vector), axis=0)#和已有的向量合并成新向量
		count = count + 1

	return theta

#convert gradients into vector
def gradients_to_vector(gradients):
	"""
	Roll all our parameters dictionary into a single vector satisfying our specific required shape.
	"""
	# 因为gradient的存储顺序是{dWL,dbL,....dW2,db2,dW1,db1}，为了统一采用[dW1,db1,...dWL,dbL]方面后面求欧式距离（对应元素）
	L = len(gradients) // 2
	keys = []
	for l in range(L):
		keys.append("dW" + str(l + 1))
		keys.append("db" + str(l + 1))
	count = 0
	for key in keys:
		# flatten parameter
		new_vector = np.reshape(gradients[key], (-1, 1))#convert matrix into vector
		if count == 0:#刚开始时新建一个向量
			theta = new_vector
		else:
			theta = np.concatenate((theta, new_vector), axis=0)#和已有的向量合并成新向量
		count = count + 1

	return theta

#convert vector into dictionary
def vector_to_dictionary(theta, layer_dims):
	"""
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
	parameters = {}
	L = len(layer_dims)  # the number of layers in the network
	start = 0
	end = 0
	for l in range(1, L):
		end += layer_dims[l]*layer_dims[l-1]
		parameters["W" + str(l)] = theta[start:end].reshape((layer_dims[l],layer_dims[l-1]))
		start = end
		end += layer_dims[l]*1
		parameters["b" + str(l)] = theta[start:end].reshape((layer_dims[l],1))
		start = end
	return parameters


def gradient_check(parameters, gradients, X, Y, layer_dims, epsilon=1e-7):
	"""
	Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

	Arguments:
	parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
	grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
	x -- input datapoint, of shape (input size, 1)
	y -- true "label"
	epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
	layer_dims -- the layer dimension of nn
	Returns:
	difference -- difference (2) between the approximated gradient and the backward propagation gradient
	"""

	parameters_vector = dictionary_to_vector(parameters)  # parameters_values
	grad = gradients_to_vector(gradients)
	num_parameters = parameters_vector.shape[0]
	J_plus = np.zeros((num_parameters, 1))
	J_minus = np.zeros((num_parameters, 1))
	gradapprox = np.zeros((num_parameters, 1))

	# Compute gradapprox
	for i in range(num_parameters):
		thetaplus = np.copy(parameters_vector)
		thetaplus[i] = thetaplus[i] + epsilon
		AL, _ = forward_propagation(X, vector_to_dictionary(thetaplus,layer_dims))
		J_plus[i] = compute_cost(AL,Y)

		thetaminus = np.copy(parameters_vector)
		thetaminus[i] = thetaminus[i] - epsilon
		AL, _ = forward_propagation(X, vector_to_dictionary(thetaminus, layer_dims))
		J_minus[i] = compute_cost(AL,Y)
		gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

	numerator = np.linalg.norm(grad - gradapprox)
	denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
	difference = numerator / denominator

	if difference > 2e-7:
		print(
			"\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
	else:
		print(
			"\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

	return difference


if __name__ == "__main__":
	X_data, y_data = load_breast_cancer(return_X_y=True)
	X_train, X_test,y_train,y_test = train_test_split(X_data, y_data, train_size=0.8,test_size=0.2,random_state=28)
	X_train = X_train.T
	y_train = y_train.reshape(y_train.shape[0], -1).T
	X_test = X_test.T
	y_test = y_test.reshape(y_test.shape[0], -1).T

	#根据自己实现的bp计算梯度
	parameters = initialize_parameters([X_train.shape[0],5,3,1])
	AL, caches = forward_propagation(X_train,parameters)
	cost = compute_cost(AL,y_train)
	gradients = backward_propagation(AL,y_train,caches)
	#gradient checking
	# # print(X_train.shape[0])
	difference = gradient_check(parameters, gradients, X_train, y_train,[X_train.shape[0],5,3,1])
