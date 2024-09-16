from activation import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x) #tanh
        tanh_prime = lambda x: 1-(np.tanh(x)) ** 2 #derivative of tanh
        super().__init__(tanh, tanh_prime) #sending to parent class