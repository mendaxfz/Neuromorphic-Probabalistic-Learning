import numpy as np
from .base_layer import BaseLayer

class Linear(BaseLayer):

    def __init__(self, out_units, in_units=None, neuron_model=None):
        super().__init__(out_units, in_units, neuron_model)

    def forward(self, x, last_layer=False, **kwargs):
        return self.neuron_model(x, last_layer, **kwargs)

    def backprop(self,dy):
        return self.neuron_model.backprop(dy)

    def update(self, lr):
        return self.neuron_model.update(lr)
