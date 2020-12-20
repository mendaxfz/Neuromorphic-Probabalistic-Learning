import numpy as np
from ..neuron_models import *

class BaseLayer:

    def __init__(self, out_units, in_units=None, neuron_model=None):
        shape = [in_units,out_units]
        self.out_units = out_units
        self.in_units = in_units
        self.shape = shape
        if neuron_model is None:
            neuron_model = LIF
        self.neuron_model = neuron_model(shape)

    def __call__(self,x, last_layer=False, **kwargs):
        return self.forward(x, last_layer, **kwargs)

    def forward(self,x, last_layer=False, **kwargs):
        return NotImplementedError

    def backprop(self,dx):
        return NotImplementedError

    def update(self,lr):
        return NotImplementedError
