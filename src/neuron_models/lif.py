import numpy as np
from .base_model import BaseNeuronModel
from ..snn_utils.encoding import to_spike
from ..nvm.pcm import pcm_weights

class LIF(BaseNeuronModel):

    def __init__(self, shape, thr=1, min=-2., max=2., scale=1.):
        super().__init__(shape, thr, min, max, scale)

    def forward(self,x,last_layer=False,**kwargs):
        self.input = x
        W = self.W
        clone_to_pcm = kwargs.get("clone_to_pcm")
        precision = kwargs.get("precision")
        write_noise_stdv = kwargs.get("write_noise_stdv")
        if clone_to_pcm:
            W = pcm_weights(self.W)
        self.v = x.dot(W) + self.B
        self.activation = to_spike(self.v, self.thr)
        self.g = 0.5 * np.bitwise_and(self.v > 0 , self.v < 2.*self.scale)
        if last_layer:
            self.g = np.ones_like(self.v)
        return self.activation

    def backprop(self,dy):
        self.out_error_gradient = dy * self.g
        self.in_error_gradient =  self.out_error_gradient.dot(self.W.T)
        return self.in_error_gradient

    def update(self, lr):
        dW = - lr* self.input.T.dot(self.out_error_gradient)
        self.W = self.W + dW.astype(np.float32)

        dB = - lr*self.out_error_gradient
        self.B = self.B + dB.astype(np.float32)
