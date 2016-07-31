''' A simple 1-layer neural network example, with multiple X'''
''' a = f(WX) , a->y'''
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_output_to_derivative(a):
    return a*(1-a)


X = np.array([[0,0,1],[1,0,0],[0,0.5,0.5]])
y = np.array([[0.4,0.3,0.8]]).T

np.random.seed(1)

# one hidden layer
W = 2*np.random.random((3,1)) - 1

for i in xrange(10000):

    for n in xrange(X.shape[0]):  # iterate each sample
        # forward
        z = np.dot(X[n].reshape(1,-1), W)
        a = sigmoid(z)
        # Compute Gradients dC_dW
        dC_dz = (a - y[n].reshape(1,-1)) * sigmoid_output_to_derivative(a)
        dz_dW = X[n].reshape(1,-1)
        dC_dW = dC_dz * dz_dW
        # Update W
        W = W - dC_dW.T * 0.05

    print '[Iter] %d', i
    print sigmoid(np.dot(X,W))
