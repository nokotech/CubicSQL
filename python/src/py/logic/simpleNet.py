# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common import crossEntropyError, softmax

class simpleNet:

    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        y = softmax( self.predict(x) )
        return crossEntropyError(y, t)

net = simpleNet()
print(net.W)
x = np.array( [0.6, 0.9] )
p = net.predict(x)
print(p)
print( np.argmax(p) )
t = np.array( [0, 0, 1] )
print( net.loss(x, t) )
