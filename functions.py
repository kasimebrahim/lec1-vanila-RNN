import numpy as np


def softmax(vector):
    return np.exp(vector) / np.sum(np.exp(vector))


# v = np.array([1, 10, 0]).reshape((3, 1))
# print(v)
# print(softmax(v))
