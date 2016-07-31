''' 2-layer, L2 loss, some samples, more systematic gradient computation '''
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_d(a):
    return a*(1-a)

X = np.array([[0,0,1],[1,0,0],[0,0.5,0.5]])
Y = np.array([[0.4,0.3,0.8]]).T

np.random.seed(1)

# 2 hidden layers
W1 = 2*np.random.random((3,2)) - 1 # 3 x 2
W2 = 2*np.random.random((2,1)) - 1 # 2 x 1
alpha1, alpha2 = (1, 1)

for i in xrange(5000):
    for n in xrange(X.shape[0]):  # iterate each sample
        # The n-th sample
        x = X[n].reshape(1,-1) # 1 x 3
        y = Y[n].reshape(1,-1) # 1 x 1
        # forward
        z1 = x.dot(W1)      #  1 x 2
        a1 = sigmoid(z1)    # 1 x 2
        z2 = a1.dot(W2)     # 1 x 1
        a2 = sigmoid(z2)    # 1 x 1
        # Compute Gradients dL_dW2
        dL_dz2 = (a2 - y) * sigmoid_d(a2) # 1 x 1
        dL_dW2 = a1.T.dot(dL_dz2) # 2 x 1
        # Compute Gradients dL_dW1
        dL_dz1 = dL_dz2.dot(W2.T) * sigmoid_d(a1) # 1 x 2
        dL_dW1 = x.T.dot(dL_dz1) # 3 x 2
        # Update W
        W1 = W1 - dL_dW1 * alpha1
        W2 = W2 - dL_dW2 * alpha2

    print '[Iter] %d', i
    print sigmoid(np.dot(sigmoid(np.dot(X,W1)), W2))
