# deep-learning
personal practice
---------------
深度学习个人练习，该项目内容包括：<br>

1.deep_neural_network.py: 自己实现的最简单的深度神经网络（多层感知机）,关于具体的推导实现讲解，请移步本人的CSDN博客：https://blog.csdn.net/u012328159/article/details/79485767<br><br>
2.deep_neural_network_ng.py ---改正版ng在Coursera上的深度神经网络<br>
<font color='red'>具体主要改正的是对relu激活函数的求导，具体内容为:</font><br>
def relu_backward(dA, cache):<br>
	"""
	Implement the backward propagation for a single RELU unit.
	
	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently
	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	Z = cache
	A = np.maximum(0, Z)
	dZ = dA * np.int64(A > 0) # np.int64(A > 0)是A对Z求导
	return dZ
<font color='red'>ng在作业中写的relu导数（个人认为是错的）为：</font><br>
def relu_backward(dA, cache):<br>
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
<br>
动态更新.................
