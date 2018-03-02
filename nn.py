#!/usr/bin/env python3

''' Create a class and functions for neural networks
to prep me for the summer
'''

# Neural network is a Directed Acyclic Graph
# 
class Network(object):
	def __init__(self, shape):
		# shape is a list of the number of neurons per layer
		# for each layer, a matrix with the weight values
		# between every neuron in this layer and every neuron in
		# the next layer
		Weights = list()
		# for each layer, a value for each neuron in that layer
		Bias = list()
