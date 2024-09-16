from denselayer import DenseLayer
from activation import Activation
from tanh import Tanh
from meanerror import mse, mse_prime
import numpy as np
#input and output
X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    DenseLayer(2, 3),
    Tanh(),
    DenseLayer(3, 1),
    Tanh(),
]

epochs = 10
learning_rate = 0.1
for e in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            # forward
            output = x
            for layers in network:
                output = layers.forward(output)

            # error
            error += mse(y, output)

            # backward
            grad = mse_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(X)
        
        print(f"{e + 1}/{epochs}, error={error}")