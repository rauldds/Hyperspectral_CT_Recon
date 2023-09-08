import torch
import numpy
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = numpy.random.normal(self.mean, self.std, size=tensor.shape)
        return tensor + noise
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class NPFlip(object):
    def __init__(self, axis, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, tensor):
        draw = numpy.random.binomial(1, p = self.p)
        return (numpy.flip(tensor, axis=self.axis).copy() if draw else tensor)
    
    def __repr__(self):
        return self.__class__.__name__ + '(flip dimension={0})'.format(self.axis)

class Normalize(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return (tensor - self.mean)/self.std
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Standardize(object):
    def __init__(self, min=0., max=0.05):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        return (tensor - self.min) / (self.max - self.min)
    
    def __repr__(self):
        return self.__class__.__name__ + '(min={0}, max={1})'.format(self.min, self.max)