import numpy as np
from ..snn_utils.encoding import bernoulli_spikes, desired_spikes
from tqdm import tqdm

class Model:

    def __init__(self):
        self.layers = list()
        self.loss = 0
        return

    def add(self, layer):
        if len(self.layers)>0:
            if layer.in_units is None:
                layer.in_units == self.layers[-1].out_units
            assert (self.layers[-1].out_units == layer.in_units), "Invalid layer size initialization"
        self.layers.append(layer)

    def compile(self, loss = None, optimizer = None):
        self.loss = loss
        self.optimizer = optimizer

    def loss_fn(self, y, y_pred):
        id = np.ones_like(y)
        return np.sum(0.5*np.square(np.maximum(0,id - y*y_pred)))

    def loss_grad(self,y,y_pred):
        id = np.ones_like(y)
        return np.maximum(0, id - y * y_pred) * (-1*y)

    def forward(self,x, **kwargs):
        z = x
        v = x
        for ix,layer in enumerate(self.layers):
            z = layer(z,(ix+1)==len(self.layers), **kwargs)
            v = layer.neuron_model.v
        y_pred = z
        return y_pred, v

    def backprop(self, dY, lr):
        dy = dY
        for layer in reversed(self.layers):
            dy = layer.backprop(dy)
            layer.update(lr)
        dX = dy
        return dX

    def train(self, x, y, lr, time_steps=10, loss=None, optimizer=None):
        for t in range(time_steps):
            x_sp = bernoulli_spikes(x)
            
            y_pred, v = self.forward(x_sp)
            self.y_pred = y_pred
            self.v = v

            y_sp = desired_spikes(y)
            
            self.loss += self.loss_fn(y_sp,v)

            dY = self.loss_grad(y_sp,v)
            self.output_error = dY
            

            self.layers[-1].neuron_model.out_error_gradient = dY

            self.backprop(dY, lr)

        return self.loss,y_pred


    def evaluate(self, x, time_steps, **kwargs):
        y_out = np.zeros([time_steps]+[self.layers[-1].shape[-1]])
        for t in range(time_steps):
            x_sp = bernoulli_spikes(x)
            y_pred, _ = self.forward(x_sp, **kwargs)
            y_out[t,:] = y_pred
        return y_out
