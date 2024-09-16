import numpy as np

def mse(y_true, y_pred):
        print(f"y_true shape: {y_true.shape}")
        print(f"y_pred shape: {y_pred.shape}")
        return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

#binary cross entropy
def bce(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
def bce_prime(y_true, y_pred):
    return ((1-y_true)/(1-y_pred) - y_true/y_pred) /np.size(y_true)