''' A simple 2-layer neural network example, with multiple X'''
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
W1 = 2*np.random.random((3,2)) - 1 # 3 x 2
W2 = 2*np.random.random((2,1)) - 1 # 2 x 1
alpha1, alpha2 = (1, 1)

for i in xrange(10000):
    for n in xrange(X.shape[0]):  # iterate each sample
        # forward
        z1 = np.dot(X[n].reshape(1,-1), W1)
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2)
        a2 = sigmoid(z2)
        # Compute Gradients dC_dW2
        dC_dz2 = np.dot((a2 - y[n].reshape(1,-1)),sigmoid_output_to_derivative(a2))
        dz2_dW2 = a1
        dC_dW2 = np.dot(dC_dz2, dz2_dW2)
        # Compute Gradients dC_dW1
        dC_dz1 = np.dot(sigmoid_output_to_derivative(a1), np.dot(W2, dC_dW2)).T
        dz1_dW1 = X[n].reshape(1,-1)
        dC_dW1 = np.dot(dC_dz1, dz1_dW1)
        # Update W
        W2 = W2 - dC_dW2.T * alpha2
        W1 = W1 - dC_dW1.T * alpha1

    print '[Iter] %d', i
    print sigmoid(np.dot(sigmoid(np.dot(X,W1)), W2))
