import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_d(a):
    return a*(1-a)

X = np.array([[1.123,3.243,0.2]])
y = np.array([[0.3]]).T

np.random.seed(1)

# one hidden layer
W = 2*np.random.random((3,1)) - 1

for i in xrange(5000):
    # Forward
    z = np.dot(X, W)
    a = sigmoid(z)
    # Compute Gradients dC_dW
    dL_da = a - y # 1 x 1
    da_dz = sigmoid_d(a) # 1 x 1
    dz_dW = X # 1 x 2
    dL_dW = dL_da * da_dz * dz_dW # 1 x 2
    # update W
    W = W - dL_dW.T * 0.01
    print '[Iter] %d', i
    print a
