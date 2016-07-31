''' A simple 1-layer neural network example'''
''' a = f(WX) , a->y'''
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_output_to_derivative(a):
    return a*(1-a)


X = np.array([[1.123,3.243]])
y = np.array([[0.3]])

np.random.seed(1)

# one hidden layer
W = 2*np.random.random((2,1)) - 1

for i in xrange(100000):
    # Forward
    z = np.dot(X, W)
    a = sigmoid(z)
    # Compute Gradients dC_dW
    dC_da = a - y # 1 x 1
    da_dz = sigmoid_output_to_derivative(a) # 1 x 1
    dz_dW = X # 1 x 2
    dC_dW = np.dot(np.dot(dC_da, da_dz), dz_dW) # 1 x 2
    # update W
    W = W - dC_dW.T * 0.01
    print '[Iter] %d', i
    print a
