import numpy as np
import os

class BaseNeuronModel:

    def __init__(self, shape, thr = 1., min = -2., max = 2., scale = 1.):
        self.shape = shape
        self.thr = thr
        self.scale = scale

        self.v = None
        self.activation = None
        self.in_error_gradient = None
        self.out_error_gradient = None
        self.g = None
        self.input = None

        self.min, self.max = min, max

        w_stdv = np.sqrt(2./shape[0])

        self.W = np.clip(np.random.normal(size=shape)*w_stdv, self.min, self.max).astype(np.float32)
        self.B = np.clip(np.random.uniform(size=[1,shape[-1]])*w_stdv, self.min, self.max).astype(np.float32)

    def __call__(self, x, last_layer=False, **kwargs):
        return self.forward(x, last_layer, **kwargs)

    def forward(self, x, last_layer=False, **kwargs):
        return NotImplementedError

    def backprop(self, dy):
        return NotImplementedError

    def update(self, lr):
        return NotImplementedError
