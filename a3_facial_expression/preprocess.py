from sklearn.decomposition import *
from autoencoder import *

def PCA(train_inputs, valid_inputs, test_inputs):
    n = 150
    pca = PCA(n_components=n, whiten=True).fit(train_inputs)   
    eigenfaces = pca.components_.reshape((n, 32, 32))
    train_pre_inputs = pca.transform(train_inputs)
    valid_pre_inputs = pca.transform(valid_inputs)
    test_pre_inputs = pca.transform(test_inputs)
    return train_pre_inputs, valid_pre_inputs, test_pre_inputs, eigenfaces

def NB(train_inputs, valid_inputs, test_inputs):
    v = nbayes(train_inputs, train_targets,0, (100-p)) 
    train_pre_inputs = train_inputs[:,v]
    valid_pre_inputs = valid_inputs[:,v]
    test_pre_inputs = test_inputs[:,v]
    return train_pre_inputs, valid_pre_inputs, test_pre_inputs
    
def nbayes(X, Y, lam, mu):
    """ Helper function for NB. """
    n_examples, n_dims = X.shape
    Y = Y.squeeze()
    
    K = 7
    mean = np.empty((K, n_dims), dtype=np.float)
    var = np.empty((K, n_dims), dtype=np.float)

    for k in range(K):
        mean[k, :] = X[Y == (k+1), :].mean(axis=0)
        var[k, :] = X[Y == (k+1), :].var(axis=0)
        
    v = mean.var(axis=0) - lam * sum(var, axis=0)
    return v > np.percentile(v, mu)


def AutoEncoder(train_inputs, valid_inputs, test_inputs):
    num_hiddens = 2000
    eps = 0.1 # leanrning rate
    momentum = 0.5
    W1, W2, b1, b2, train_pre_inputs, valid_pre_inputs, test_pre_inputs = AutoEncoder(train_inputs, valid_inputs, num_hiddens, eps, momentum, 200)    
    return train_pre_inputs.T, valid_pre_inputs.T, test_pre_inputs.T