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
		# for each layer, a matrix with the weight values between
		# every neuron in this layer and every neuron in the next layer
		self.layers = len(shape)
		self.shape = shape
		# Randomly initialize weights and biases
		self.biases = [np.random.randn(y, 1) for y in shape[1:]]
		# Zip is a nice trick to pair the previous layer with the current layer
		self.weights = [np.random.randn(x, y) for x, y in zip(shape[1:], shape[:-1])]

	def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
		# Set up test data
		if test_data: test_data_length = len(test_data)
		# Train for however many epochs are specfied
		for epoch in range(epochs):
			shuffle(training_data) # shuffle the data
			## Split data up into mini batches
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
			# For each mini batch, update the weights using gradient descent
			for mini_batch in mini_batches:
				self.update(mini_batch, learning_rate)
			# Show improvement for each epoch
			if test_data:
				# accuracy = "%0.3f" % (self.evaluate(test_data)/float(test_data_length) * 100)
				print("Epoch {0}'s accuracy: {1} / {2}".format(epoch+1, self.evaluate(test_data), len(test_data)))
			else:
				print("Epoch {0} complete".format(epoch))

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.calculate_output(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def update(self, mini_batch, learning_rate):
		# Init gradients
		weight_gradient = [np.zeros(w.shape) for w in self.weights]
		biases_gradient = [np.zeros(b.shape) for b in self.biases]
		for training_example in mini_batch:
			this_example_dw, this_example_db = self.backpropagate(training_example) # get gradient for this example
			# Add changes to gradients
			weight_gradient = [sum(x) for x in zip(weight_gradient, this_example_dw)]
			biases_gradient = [sum(x) for x in zip(biases_gradient, this_example_db)]

		# Average gradient and make steps
		self.weights = [w-(learning_rate*dw/len(mini_batch)) for w, dw in zip(self.weights, weight_gradient)]
		self.biases  = [b-(learning_rate*db/len(mini_batch)) for b, db in zip(self.biases, biases_gradient)]

	def backpropagate(self, training_example):
		# Init arrays to be returned
		dCdw = [np.zeros(w.shape) for w in self.weights]		
		dCdb = [np.zeros(b.shape) for b in self.biases]
		# Initiate arrays for forward propagation and error
		z  = list()
		a  = [training_example[0]]
		# Forward propagate and store z's & a's
		for i, (b, w) in enumerate(zip(self.biases, self.weights)):
			z.append(np.dot(w, a[i]) + b)
			a.append(sigmoid(z[-1]))
		# Set output layer values
		sl = self.cost_derivative(a[-1], training_example[1]) * sigmoid_deriv(z[-1])
		dCdw[-1] = np.dot(sl, a[-2].transpose())
		dCdb[-1] = sl
		# Backpropagate and compute the error for n-1 to 1 layers
		for i in reversed(range(self.layers - 2)):
			sl = np.dot(self.weights[i+1].transpose(), sl) * sigmoid_deriv(z[i])
			dCdw[i] = np.dot(sl, a[i].transpose())
			dCdb[i] = sl
		return dCdw, dCdb

	def calculate_output(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b) 
		return a

	def cost_derivative(self, output_activations, y):
		return (output_activations - y)

# Helper functions
def sigmoid(v):
	return 1.0/(1.0+np.exp(-v))

def sigmoid_deriv(v):
	return (sigmoid(v)) * (1 - sigmoid(v))