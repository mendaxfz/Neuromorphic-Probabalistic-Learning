import numpy as np
import numbers

class DataTransform:

    def __init__(self):
        return NotImplementedError

    def __call__(self):
        return NotImplementedError

    def pad(self,x,padding,fill,padding_mode):
        if isinstance(padding, numbers.Number):
            padding = ((padding,padding),(padding,padding),(0,0))
        x = np.pad(x,pad_width=padding,mode=padding_mode,constant_values=fill)
        return x

    def crop(self,x, start_x, start_y, height, width):
        return x[start_x:start_x+height,start_y:start_y+height,:]

    def hflip(self,x):
        w_len = x.shape[1]
        w_range_reversed = np.arange(w_len-1,-1,-1)
        return x[:,w_range_reversed,:]

class Scale(DataTransform):
    def __init__(self,scale):
        self.scale = scale

    def __call__(self,x):
        return self.scale*x.astype(np.float32)
