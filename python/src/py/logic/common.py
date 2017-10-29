import numpy as np
import matplotlib.pylab as plt
import json as json

# neuron function
def AND(x1, x2):
    return 0 if( (np.sum(np.array([x1, x2])*np.array([0.5, 0.5])) - 0.75) <= 0 ) else 1
def XAND(x1, x2):
    return 0 if( (np.sum(np.array([x1, x2])*np.array([-0.5, -0.5])) + 0.75) <= 0 ) else 1
def OR(x1, x2):
    return 0 if( (np.sum(np.array([x1, x2])*np.array([0.5, 0.5])) - 0.25) <= 0 ) else 1
def XOR(x1, x2):
    return AND( XAND(x1, x2), OR(x1, x2) )

# activation function
def stepFunc(x):
    return np.array((x > 0), dtype=np.int)
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )
def relu(x):
    return np.maximum(0, x)
def identity(x):
    return np.maximum(0, x)
def softmax(x):
    _x = np.exp( x - np.max(x) )
    return _x / np.sum(_x)

# loss function
def meanSquaredError(y, t):
    return 0.5 * np.sum( (y-t)**2 )
def crossEntropyError(y, t):
    return -np.sum( t * np.log( y + (1e-7) ) )

# 
def numericalDiff(f, x):
    return ( f( x + (1e-4) ) - f( x - (1e-4) ) ) / ( 2 * (1e-4) )
def numericalDradient(f, x):
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmpVal = x[idx]
        x[idx] = tmpVal + (1e-4)
        fxh1 = f(x)
        x[idx] = tmpVal - (1e-4)
        fxh2 = f(x)
        grad[idx] = ( fxh1 - fxh2 ) / (2*h)
        x[idx] = tmpVal
    return grad

#
def gradientDescent(f, initX, lr=0.01, stepNum=100):
    x = initX
    for i in range(stepNum):
        x -= lr * numericalDradient(f, x)
    return x

# graph
def showPlot(x, y):
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
def forward(x):
    # 一層目
    W1 = np.array([[0.1, 0.3 ,0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2 , 0.3])
    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    print(Z1)

    # 二層目
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print(Z2)

    # 三層目（分類問題）
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Z3 = identity(A3)
    print(Z3)
    return Z3

# main
if __name__ == '__main__':
    # X = np.array([1.0, 0.5])
    # Y = forward(X)
    # print(Y)
    # ------------------------------------ #
    # A = np.array([0.3, 2.9, 4.0])
    # Y = softmax(A)
    # print(Y)
    # print(np.sum(Y))
    # print(1e-7 * 10**4 )
    # ------------------------------------ #
    pass

# test 
if __name__ == '__main2__':
    print('  OR(0, 0) = %d' % OR(0, 0))
    print('  OR(0, 1) = %d' % OR(0, 1))
    print('  OR(1, 0) = %d' % OR(1, 0))
    print('  OR(1, 1) = %d' % OR(1, 1))

    print(' XOR(0, 0) = %d' % XOR(0, 0))
    print(' XOR(0, 1) = %d' % XOR(0, 1))
    print(' XOR(1, 0) = %d' % XOR(1, 0))
    print(' XOR(1, 1) = %d' % XOR(1, 1))

    print(' AND(0, 0) = %d' % AND(0, 0))
    print(' AND(0, 1) = %d' % AND(0, 1))
    print(' AND(1, 0) = %d' % AND(1, 0))
    print(' AND(1, 1) = %d' % AND(1, 1))

    print('XAND(0, 0) = %d' % XAND(0, 0))
    print('XAND(0, 1) = %d' % XAND(0, 1))
    print('XAND(1, 0) = %d' % XAND(1, 0))
    print('XAND(1, 1) = %d' % XAND(1, 1))

    x = np.arange(-5.0, 5.0, 0.1)
    plt.plot(x, stepFunc(x))
    plt.plot(x, sigmoid(x))
    plt.plot(x, relu(x))
    plt.ylim(-0.1, 1.1)
    #plt.show()

    A = np.array([[1, 2, 3], [4, 5, 6]])
    print("次元数 = %d" % np.ndim(A))
    print("列行 = %s" % json.dumps(A.shape))

    A = np.array([[1,2], [3,4]])
    B = np.array([[5,6], [7,8]])
    print("積")
    print(np.dot(A, B))
