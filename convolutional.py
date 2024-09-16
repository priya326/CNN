import numpy as np
from scipy import signal
# from layer import layer

class Convolutional ():
    def __init__(self, depth, input_shape, kernel_size):
        # input_shape is a tuple with depth height and width of the input
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape 
        self.output_shape = (depth, input_height-kernel_size+1, input_width-kernel_size+1)
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in (self.depth):
            for j in (self.input_depth):
                self.output[i] = signal.correlate2d(self.input(i), self.kernels(i, j), "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input_shape)
        kernel_gradient = np.zeros(self.kernel_shape)

        for i in (self.depth):
            for j in (self.input_depth):
                kernel_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[i] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        self.kernels -= learning_rate * kernel_gradient
        self.biases -= learning_rate * output_gradient # because gradient of bias = output gradient



