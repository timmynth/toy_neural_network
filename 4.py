import numpy as np
# X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
# y = np.array([[0,1,1,0]]).T

X = np.array([[0,0,1]])
y = np.array([[0.4]]).T

alpha,hidden_dim = (0.5,2)
synapse_0 = 2*np.random.random((3,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,1)) - 1
for j in xrange(60000):
    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
    print layer_1.shape # 1 x 2
    print layer_2.shape # 1 x 1
    print '===='


    layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
    print layer_2_delta.shape # 1 x 1
    print layer_1_delta.shape # 1 x 2

    synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
    synapse_0 -= (alpha * X.T.dot(layer_1_delta))

    print layer_1.T.dot(layer_2_delta).shape # 2 x 1
    print X.T.dot(layer_1_delta).shape  # 3 x 2

    exit(-1)
    print 'Iter = ', j
    print layer_2
