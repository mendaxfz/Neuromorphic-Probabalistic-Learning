import numpy as np

def to_spike(x, th):
    return (x >= th).astype(np.float32)

def bernoulli_spikes(x):
    return (np.random.uniform(size=x.shape) < x).astype(np.float32)

def desired_spikes(x):
    y = -1 * np.ones_like(x)
    y[:,np.argmax(x)] = 1
    return y.astype(np.float32)
