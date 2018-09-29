# deep-learning
personal practice
---------------
深度学习个人练习，该项目内容包括：

+ 实现了四种初始化方法：zero initialize, random initialize, xavier initialize, he initialize。

+ 深度神经网络

+ 正则化

+ dropout

+ 三种梯度下降方法：BGD, SGD, mini-batch

+ 六种优化算法：momentum、nesterov momentum、Adagrad、Adadelta、RMSprop、Adam

+ 梯度检验

+ batch normalization
------

![#f03c15](https://placehold.it/15/f03c15/000000?text=+) ***Note: 下列 1-10中网络架构主要为四大块： initialize parameters、forward propagation、backward propagation、 update parameters，其中在 fp 和 bp 的时候各个功能没有单独封装，这样会导致耦合度过高，结构不清晰。
11中优化了网络结构，使得耦合度更低，网络结构推荐用11中的结构。
重构了神经网络架构（见 deep_neural_network_release.py），把各功能函数分离出来，耦合度更低，结构更清楚，bp过程更加清晰。推荐此版本，用1-10时，可用此版本替换相应代码***

1、**deep_neural_network_v1.py**：自己实现的最简单的深度神经网络（多层感知机),不包含正则化,dropout,动量等...总之是最基本的,只有fp和bp。

2、**deep_neural_network_v2.py**:  自己实现的最简单的深度神经网络（多层感知机）,和v1的唯一区别在于：v1中fp过程,caches每一层存储的是（w,b,z,A_pre）,
而v2每一层存储的是（w,b,z,A）, 第0层存储的（None,None,None,X）,X即A0。    `个人更推荐用v2版本`.

关于具体的推导实现讲解，请移步本人的CSDN博客：https://blog.csdn.net/u012328159/article/details/79485767

3、**deep_neural_network_ng.py**: ---改正版ng在Coursera上的深度神经网络<br>
**具体主要改正的是对relu激活函数的求导，具体内容为:<br>**
```python
def relu_backward(dA, cache):
	"""
	Implement the backward propagation for a single RELU unit.
	
	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently
	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	Z = cache
	dZ = dA * np.int64(Z > 0)
	return dZ
```
**ng在作业中写的relu导数（个人认为是错的）为：<br>**
```python
def relu_backward(dA, cache):
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
```

4、**compare_initializations.py**： 比较了四种初始化方法（初始化为0，随机初始化，Xavier initialization和He initialization），具体效果见CSDN博客：https://blog.csdn.net/u012328159/article/details/80025785

5、 **deep_neural_network_with_L2.py**: 带L2正则项正则项的网络（在deep_neural_network.py的基础上增加了L2正则项）

6、 **deep_neural_network_with_dropout.py** ：带dropout正则项的网络（在deep_neural_network.py的基础上增加了dropout正则项），具体详见CSDN博客：https://blog.csdn.net/u012328159/article/details/80210363

7、 **gradient_checking.py** ： use gradient checking in dnn，梯度检验，可以检查自己手撸的bp是否正确。具体原理，详见我的CSDN博客：https://blog.csdn.net/u012328159/article/details/80232585

8、 **deep_neural_network_with_gd.py** ：实现了三种梯度下降，包括：batch gradient descent（BGD）、stochastic gradient descent（SGD）和 mini-batch gradient descent。具体内容见我的CSDN博客：https://blog.csdn.net/u012328159/article/details/80252012

9、 **deep_neural_network_with_optimizers.py** ：实现了深度学习中几种优化器，包括：momentum、nesterov momentum、Adagrad、Adadelta、RMSprop、Adam。关于这几种算法，具体内容，见本人的CSDN博客：https://blog.csdn.net/u012328159/article/details/80311892

10、 **机器学习资料整理.pdf** ：整理了一些我知道的机器学习资料，希望能够帮助到想学习的同学。博客同步地址：https://blog.csdn.net/u012328159/article/details/80574713

11、 **batch_normalization.py** ：实现了batch normalization, 改进了整个网络的架构，使得网络的架构更加清晰，耦合度更低。关于batch normalization的具体内容，见本人的CSDN博客：https://blog.csdn.net/u012328159/article/details/82840084

12、 **deep_neural_network_release.py**：重构了深度神经网络，把各功能函数分离出来，耦合度更低，结构更清楚，bp过程更加清晰。**推荐此版本**

<br>
<br>
--------
动态更新.................
