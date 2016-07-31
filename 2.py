''' 1-layer, L2 loss, some samples '''
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_d(a):
    return a*(1-a)

X = np.array([[0,0,1],[1,0,0],[0,0.5,0.5]])
Y = np.array([[0.4,0.3,0.8]]).T

np.random.seed(1)

# one hidden layer
W = 2*np.random.random((3,1)) - 1

for i in xrange(5000):
    for n in xrange(X.shape[0]):  # iterate each sample
        # The n-th sample
        x = X[n].reshape(1,-1) # 1 x 3
        y = Y[n].reshape(1,-1) # 1 x 1
        # forward
        z = np.dot(x, W)
        a = sigmoid(z)
        # Compute Gradients dL_dW
        dL_dz = (a - y) * sigmoid_d(a) # 1 x 1
        dz_dW = x # 1 x 3
        dL_dW = dz_dW.T.dot(dL_dz) # 3 x 1
        # Update W
        W = W - dL_dW * 0.5
    print '[Iter] %d', i
    print sigmoid(np.dot(X,W))
