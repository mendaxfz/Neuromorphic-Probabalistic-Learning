import numpy as np

def one_hot(x, num_labels=10):
    y = np.zeros(shape=[x.shape[0]]+[num_labels])
    y[np.arange(x.shape[0]),x] = 1
    return y
