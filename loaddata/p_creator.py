import numpy as np


def M1(size):
    base = np.array(range(size * size))
    temp = base.reshape((size, size))
    np.random.shuffle(temp)
    np.random.shuffle(temp.T)
    return base


def M2(size):
    return np.random.permutation(range(size * size))
