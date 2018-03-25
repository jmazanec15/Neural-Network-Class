#!/usr/bin/env python3
from random import shuffle
import numpy as np
''' 
Create a class and functions for neural networks
to prep me for the summer
Guidance from http://neuralnetworksanddeeplearning.com
'''
class Network(object):
	def __init__(self, shape):
		# shape is a list of the number of neurons per layer
		# for each layer, a matrix with the weight values
		# between every neuron in this layer and every neuron in
		# the next layer
		# shape = [784, 30, 10]
		self.layers = len(shape)
		self.shape = shape
		# Randomly initialize array
		self.biases = [np.random.randn(y, 1) for y in shape[1:]]
		# Zip is a nice trick to pair the previous layer with the current layer
		self.weights = [np.random.randn(x, y) for x, y in zip(shape[1:], shape[:-1])]

	def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
		# Set up test data
		if test_data: 
			test_data_length = len(test_data)
		# Train for however many epochs are specfied
		for epoch in range(epochs):
			shuffle(training_data) # shuffle the training data
			## Split data up into mini batches
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
			## For each mini batch, update the weights
			# Basically, for each example, compute the output
			# Compute the error
			# Compute gradient  for each example
			# Average the gradients and then apply the step
			for mini_batch in mini_batches:
				self.update(mini_batch, learning_rate)

			# Show improvement for each epoch
			if test_data:
				accuracy = "%0.3f" % (self.evaluate(test_data)/float(test_data_length) * 100)
				print("Epoch {0}'s accuracy: {1}%".format(epoch, accuracy))
			else:
				print("Epoch {0} complete".format(epoch))


	def evaluate(self, test_data):
		test_results = [(np.argmax(self.calculate_output(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def update(self, mini_batch, learning_rate):
		# Compute error
		# Backpropagate to calculate the 
		# list of matrices for each weight
		weight_gradient = [np.zeros(w.shape) for w in self.weights]
		biases_gradient = [np.zeros(b.shape) for b in self.biases]
		for training_example in mini_batch:
			# get gradient for this example
			this_example_w, this_example_b = self.backpropagate(training_example)
			# Problematic
			weight_gradient = [sum(x) for x in zip(weight_gradient, this_example_w)]
			biases_gradient = [sum(x) for x in zip(biases_gradient, this_example_b)]
		# Average gradient and make steps
		self.weights = [w - dw / float(len(mini_batch)) * learning_rate for w, dw in zip(self.weights, weight_gradient)]
		self.biases  = [b - db / float(len(mini_batch)) * learning_rate for b, db in zip(self.biases, biases_gradient)]

	def backpropagate(self, training_example):
		# Initiate arrays for forward propagation and error
		z  = [np.array(self.shape[i]) for i in range(self.layers)]
		a  = [np.array(self.shape[i]) for i in range(self.layers)]
		sl = [np.array(self.shape[i]) for i in range(self.layers)] # error for each neuron
		# Initiate first layer as input array
		a[0] = np.array(training_example[0])
		# Step 0 ~ Forward propagate and store z's & a's
		for i, (b, w) in enumerate(zip(self.biases, self.weights)):
			z[i+1] 	= np.dot(w, a[i]) + b
			a[i+1] 	= sigmoid(z[i+1])
		# Step 1 ~ sL = ∇aC ⊙ σ′(zL) --> compute error in output layer hadamard
		sl[-1] = self.cost_derivative(training_example[1], a[-1]) * sigmoid_deriv(z[-1])
		# Step 2 ~ Backpropagate and compute the error for n-1 to 1 layers
		for i in reversed(range(1, self.layers - 1)): # only really need to compute it for 2 through n-1 layers of network
			sl[i] = np.dot(np.transpose(self.weights[i]), sl[i+1]) * sigmoid_deriv(z[i-1])
		# Step 3 ~ set gradient for biases
		dCdb = sl[1:]
		# Step 4 ~ set gradient for weights
		dCdw = [a_x * s_l for a_x, s_l in zip(a[1:], sl[1:])]
		return dCdw, dCdb

	def calculate_output(self, input_array):
		a = np.array(input_array)
		for b, w in zip(self.biases, self.weights):
			z = w @ a + b # @ is dot product
			a = sigmoid(z) # did it in two steps just to spell it out
		return a

	def cost_derivative(self, output_activations, y):
		return (output_activations - y)

## Helper functions
def sigmoid(v):
	return 1/(1+np.exp(-1*v))

def sigmoid_deriv(v):
	return (sigmoid(v)) * (1 - sigmoid(v))

def quadratic_cost(y_act, y_pred, num_examples):
	return 1/(2*num_examples)*sum((y_act - y_pred)**2)
