import numpy as np


def pcm_weights(w, precision=4, write_noise_stdv=0.0):
    quant_max = 2**(precision-1) -1
    w_max = np.amax(np.abs(w))
    w_norm = w/w_max
    w_scaled = w_norm * quant_max
    w_round = np.round(w_scaled)
    w_clip = np.clip(w_round,-quant_max,quant_max)
    w_renorm = w_clip * w_max / quant_max

    if write_noise_stdv > 0.0:
        w_renorm = w_renorm + np.random.normal(size=w.shape) * write_noise_stdv

    return w_renorm
