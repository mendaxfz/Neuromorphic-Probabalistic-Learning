from collections.abc import Iterator
import numpy as np

class DataLoader(Iterator):

    def __init__(self,x_data,y_data=None,transforms=None,batch_size=32,shuffle=False,num_classes=None):
        self.transforms = transforms
        self.num_classes = num_classes
        self.start = 0

        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_samples = x_data.shape[0]
        self.sample_indx = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.sample_indx)

    def __iter__(self): return self

    def __next__(self):
        if self.start >= self.num_samples:
            self.start = 0
            if self.shuffle:
                np.random.shuffle(self.sample_indx)
            raise StopIteration
        sample_indx = self.sample_indx[self.start:self.start+self.batch_size]
        self.start = self.start + self.batch_size
        return self.flow(sample_indx)

    def flow(self,sample_indx):
       x_batch = self.process_x(self.x_data[sample_indx])
       y_batch = self.process_y(self.y_data[sample_indx])
       return x_batch,y_batch

    def process_x(self,x):
        batch_size = self.batch_size
        x_processed = x.copy().astype(np.float32)
        if self.transforms is not None:
            for i in range(batch_size):
                for ifx,func in enumerate(self.transforms):
                    x_processed[i] = func(x_processed[i])
            return x_processed

    def process_y(self,y):
        if self.num_classes is not None:
            return self.to_categorical(y,self.num_classes)
        return y


    def to_categorical(self,y,num_classes):
        n = self.batch_size
        labels = y
        if len(y.shape)>1:
            labels = y.reshape(-1)
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), labels] = 1
        return categorical
