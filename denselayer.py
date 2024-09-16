from layer import Layer
import numpy as np
class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)