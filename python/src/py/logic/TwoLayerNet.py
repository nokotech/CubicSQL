# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common import *

class TwoLayerNet:

    def __init__(self, inputSize, hiddenSize, outputSize, weightInitStd=0.01):
        self.params = {}
        self.params['W1'] = weightInitStd * np.random.randn(inputSize, hiddenSize)
        self.params['b1'] = np.zeros(hiddenSize)
        self.params['W2'] = weightInitStd * np.random.randn(hiddenSize, outputSize)
        self.params['b1'] = np.zeros(outputSize)
    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        return crossEntropyError( self.predict(x), t )

    def acccuracy(self, x, t):
        y = np.argmax( self.predict(x), axis=1 )
        t = np.argmax( t, axis=1 )
        return np.sum(y == t) / float(x.shape[0])

    def numericalGradient(self, x, t):
        lossW = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numericalGradient( lossW, self.params['W1'] )
        grads['b1'] = numericalGradient( lossW, self.params['b1'] )
        grads['W2'] = numericalGradient( lossW, self.params['W2'] )
        grads['b2'] = numericalGradient( lossW, self.params['b2'] )
        return grads