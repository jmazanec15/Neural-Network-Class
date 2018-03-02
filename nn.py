#!/usr/bin/env python3
import random as rnd
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
		self.layers = len(shape)
		self.shape = shape
		# initialize weights with random values between -1 and 1
		self.weights = [[[2*rnd.random()-1 for _ in range(shape[l])] for _ in range(shape[l-1])] for l in range(1,self.layers)]
		# for each layer, a value for each neuron in that layer
		self.biases = [[2*rnd.random()-1 for _ in range(shape[l])] for l in range(1,self.layers)]
		# Need a cost function - prob start with just Quadratic

def main():
	network = Network([3, 4, 2])

if __name__ == "__main__":
	main()