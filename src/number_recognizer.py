#!/usr/bin/env python3

from nn import Network
from data_formatter import *

def main():
	training_data, validation_data, test_data = load_data_wrapper()
	neural_net = Network([784, 30, 10])
	neural_net.SGD(list(training_data), 10, 1000, 1, test_data=list(test_data))

if __name__ == "__main__":
	main()