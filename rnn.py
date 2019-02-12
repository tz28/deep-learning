import numpy as np


def initialize_parameters(n_a, n_x, n_y):
	"""
	Initialize parameters with small random values
	Returns:
	parameters -- python dictionary containing:
						Wax -- Weight matrix multiplying the input, of shape (n_a, n_x)
						Waa -- Weight matrix multiplying the hidden state, of shape (n_a, n_a)
						Wya -- Weight matrix relating the hidden-state to the output, of shape (n_y, n_a)
						b --  Bias, numpy array of shape (n_a, 1)
						by -- Bias relating the hidden-state to the output, of shape (n_y, 1)
	"""
	np.random.seed(1)
	Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
	Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
	Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
	ba = np.zeros((n_a, 1))  # hidden bias
	by = np.zeros((n_y, 1))  # output bias
	parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

	return parameters


def softmax(x):
	#这里减去最大值，是为了防止大数溢出，根据softmax的参数冗余性，减去任意一个数，结果不变
	e_x = np.exp(x - np.max(x))
	return e_x / np.sum(e_x, axis=0)


def rnn_step_forward(xt, a_prev, parameters):
	"""
	Implements a single forward step of the RNN-cell that uses a tanh
    activation function

	Arguments:
	xt -- the input data at timestep "t", of shape (n_x, m).
	a_prev -- Hidden state at timestep "t-1", of shape (n_a, m)
	**here, n_x denotes the dimension of word vector, n_a denotes the number of hidden units in a RNN cell
	parameters -- python dictionary containing:
						Wax -- Weight matrix multiplying the input, of shape (n_a, n_x)
						Waa -- Weight matrix multiplying the hidden state, of shape (n_a, n_a)
						Wya -- Weight matrix relating the hidden-state to the output, of shape (n_y, n_a)
						ba --  Bias,of shape (n_a, 1)
						by -- Bias relating the hidden-state to the output,of shape (n_y, 1)
	Returns:
	a_next -- next hidden state, of shape (n_a, 1)
	yt_pred -- prediction at timestep "t", of shape (n_y, 1)
	"""

	# get parameters from "parameters"
	Wax = parameters["Wax"] #(n_a, n_x)
	Waa = parameters["Waa"] #(n_a, n_a)
	Wya = parameters["Wya"] #(n_y, n_a)
	ba = parameters["ba"]   #(n_a, 1)
	by = parameters["by"]   #(n_y, 1)

	a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba) #(n_a, 1)
	yt_pred = softmax(np.dot(Wya, a_next) + by) #(n_y,1)

	return a_next, yt_pred


def rnn_forward(X, Y, a0, parameters, vocab_size=27):
	x, a, y_hat = {}, {}, {}
	a[-1] = np.copy(a0)
	# initialize your loss to 0
	loss = 0

	for t in range(len(X)):
		# Set x[t] to be the one-hot vector representation of the t'th character in X.
		# if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector.
		x[t] = np.zeros((vocab_size, 1))
		if (X[t] != None):
			x[t][X[t]] = 1
		# Run one step forward of the RNN
		a[t], y_hat[t] = rnn_step_forward(x[t], a[t - 1], parameters) #a[t]: (n_a,1), y_hat[t]:(n_y,1)
		# Update the loss by substracting the cross-entropy term of this time-step from it.
		#这里因为真实的label也是采用onehot表示的，因此只要把label向量中1对应的概率拿出来就行了
		loss -= np.log(y_hat[t][Y[t], 0])

	cache = (y_hat, a, x)

	return loss, cache


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):

	gradients['dWya'] += np.dot(dy, a.T)
	gradients['dby'] += dy
	Wya = parameters['Wya']
	Waa = parameters['Waa']
	da = np.dot(Wya.T, dy) + gradients['da_next']  #每个cell的Upstream有两条，一条da_next过来的，一条y_hat过来的
	dtanh = (1 - a * a) * da  # backprop through tanh nonlinearity
	gradients['dba'] += dtanh
	gradients['dWax'] += np.dot(dtanh, x.T)
	gradients['dWaa'] += np.dot(dtanh, a_prev.T)
	gradients['da_next'] = np.dot(Waa.T, dtanh)

	return gradients


def rnn_backward(X, Y, parameters, cache):
	# Initialize gradients as an empty dictionary
	gradients = {}

	# Retrieve from cache and parameters
	(y_hat, a, x) = cache
	Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']

	# each one should be initialized to zeros of the same dimension as its corresponding parameter
	gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
	gradients['dba'], gradients['dby'] = np.zeros_like(ba), np.zeros_like(by)
	gradients['da_next'] = np.zeros_like(a[0])

	# Backpropagate through time
	for t in reversed(range(len(X))):
		dy = np.copy(y_hat[t])
		dy[Y[t]] -= 1 #计算y_hat - y,即预测值-真实值，因为真实值是one-hot向量，只有1个1，其它都是0，
		# 所以只要在预测值（向量）对应位置减去1即可，其它位置减去0相当于没变
		gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t - 1])

	return gradients, a

#梯度裁剪
def clip(gradients, maxValue):
	'''
	Clips the gradients' values between minimum and maximum.

	Arguments:
	gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
	maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

	Returns:
	gradients -- a dictionary with the clipped gradients.
	'''

	dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['dba'], gradients['dby']

	# clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby].
	for gradient in [dWax, dWaa, dWya, db, dby]:
		np.clip(gradient, -maxValue, maxValue, gradient)

	gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": db, "dby": dby}

	return gradients



def update_parameters(parameters, gradients, lr):
	parameters['Wax'] += -lr * gradients['dWax']
	parameters['Waa'] += -lr * gradients['dWaa']
	parameters['Wya'] += -lr * gradients['dWya']
	parameters['ba']  += -lr * gradients['dba']
	parameters['by']  += -lr * gradients['dby']
	return parameters


def sample(parameters, char_to_ix, seed):
	"""
	Sample a sequence of characters according to a sequence of probability distributions output of the RNN
	Arguments:
	parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
	char_to_ix -- python dictionary mapping each character to an index.
	seed -- used for grading purposes. Do not worry about it.
	Returns:
	indices -- a list of length n containing the indices of the sampled characters.
	"""

	# Retrieve parameters and relevant shapes from "parameters" dictionary
	Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
	vocab_size = by.shape[0]
	n_a = Waa.shape[1]
	# Step 1: Create the one-hot vector x for the first character (initializing the sequence generation).
	x = np.zeros((vocab_size, 1))
	# Step 1': Initialize a_prev as zeros
	a_prev = np.zeros((n_a, 1))

	# Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate
	indices = []

	# Idx is a flag to detect a newline character, we initialize it to -1
	idx = -1

	# Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
	# its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
	# trained model), which helps debugging and prevents entering an infinite loop.
	counter = 0
	newline_character = char_to_ix['\n']

	while (idx != newline_character and counter != 50):
		# Step 2: Forward propagate x using the equations (1), (2) and (3)
		a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)
		z = np.dot(Wya, a) + by
		y = softmax(z)

		# for grading purposes
		np.random.seed(counter + seed)

		# Step 3: Sample the index of a character within the vocabulary from the probability distribution y
		idx = np.random.choice(vocab_size, p = y.ravel())  # 等价于np.random.choice([0,1,...,vocab_size-1], p = y.ravel())，
		# 一维数组或者int型变量，如果是数组，就按照里面的范围来进行采样，如果是单个变量，则采用np.arange(a)的形式
		# Append the index to "indices"
		indices.append(idx)
		# Step 4: Overwrite the input character as the one corresponding to the sampled index.
		#每次生成的字符是下一个时间步的输入
		x = np.zeros((vocab_size, 1))
		x[idx] = 1

		# Update "a_prev" to be "a"
		a_prev = a
		# for grading purposes
		seed += 1
		counter += 1
	if (counter == 50):
		indices.append(char_to_ix['\n'])

	return indices


def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
	"""
	Execute one step of the optimization to train the model.

	Arguments:
	X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
	Y -- list of integers, exactly the same as X but shifted one index to the left.
	a_prev -- previous hidden state.
	parameters -- python dictionary containing:
						Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
						Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
						Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
						b --  Bias, numpy array of shape (n_a, 1)
						by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
	learning_rate -- learning rate for the model.

	Returns:
	loss -- value of the loss function (cross-entropy)
	gradients -- python dictionary containing:
						dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
						dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
						dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
						db -- Gradients of bias vector, of shape (n_a, 1)
						dby -- Gradients of output bias vector, of shape (n_y, 1)
	a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
	"""

	# Forward propagate through time
	loss, cache = rnn_forward(X, Y, a_prev, parameters)

	# Backpropagate through time
	gradients, a = rnn_backward(X, Y, parameters, cache)

	# Clip your gradients between -5 (min) and 5 (max)
	gradients = clip(gradients, 5)

	# Update parameters
	parameters = update_parameters(parameters, gradients, learning_rate)

	return loss, parameters, a[len(X) - 1]

def get_initial_loss(vocab_size, seq_length):
	return -np.log(1.0/vocab_size)*seq_length

def smooth(loss, cur_loss):
	return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
	txt = ''.join(ix_to_char[ix] for ix in sample_ix)
	txt = txt[0].upper() + txt[1:]  # capitalize first character
	print ('%s' % (txt, ), end='')


def model(data, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27):
	"""
	Trains the model and generates dinosaur names.

	Arguments:
	data -- text corpus
	ix_to_char -- dictionary that maps the index to a character
	char_to_ix -- dictionary that maps a character to an index
	num_iterations -- number of iterations to train the model for
	n_a -- number of units of the RNN cell
	dino_names -- number of dinosaur names you want to sample at each iteration.
	vocab_size -- number of unique characters found in the text, size of the vocabulary

	Returns:
	parameters -- learned parameters
	"""

	# Retrieve n_x and n_y from vocab_size
	n_x, n_y = vocab_size, vocab_size

	# Initialize parameters
	parameters = initialize_parameters(n_a, n_x, n_y)
	# Initialize loss (this is required because we want to smooth our loss, don't worry about it)
	loss = get_initial_loss(vocab_size, dino_names)

	# Initialize the hidden state of your LSTM
	a_prev = np.zeros((n_a, 1))

	# Optimization loop
	for j in range(num_iterations):

		# Use the hint above to define one training example (X,Y) (≈ 2 lines)
		index = j % len(data)
		X = [None] + [char_to_ix[ch] for ch in data[index]]
		Y = X[1:] + [char_to_ix["\n"]]
		# Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
		# Choose a learning rate of 0.01
		curr_loss, parameters, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
		# Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
		loss = smooth(loss, curr_loss)
		# Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
		if j % 2000 == 0:
			print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
			# The number of dinosaur names to print
			seed = 0
			for name in range(dino_names):
				# Sample indices and print them
				sampled_indices = sample(parameters, char_to_ix, seed)
				print_sample(sampled_indices, ix_to_char)
				seed += 1  # To get the same result for grading purposed, increment the seed by one.

			print('\n')

	return parameters


if __name__ == "__main__":
	data = open('dinos.txt', 'r').read()
	data = data.lower()
	chars = list(set(data))  # str->set,例如:'123'转set，会转为无序不重复的，形如:{'1','2','3'}
	print(chars)
	data_size, vocab_size = len(data), len(chars)
	print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
	char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
	ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
	print(ix_to_char)

	# Build list of all dinosaur names (training examples).
	with open("dinos.txt") as f:
		examples = f.readlines()
	examples = [x.lower().strip() for x in examples]

	# Shuffle list of all dinosaur names
	np.random.seed(0)
	np.random.shuffle(examples)

	parameters = model(examples, ix_to_char, char_to_ix)