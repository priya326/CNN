from activation import Activation
import numpy as np

class Sigmoid(Activation):
    def __init__(self) -> None:
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        def sigmoidprime(x):
            s = sigmoid(x)
            return s * (1-s)
        super().__init__(sigmoid, sigmoidprime)
