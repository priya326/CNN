from layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activationprime) -> None:
        self.activation = activation
        self.activationprime = activationprime
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient,self.activationprime(self.input))