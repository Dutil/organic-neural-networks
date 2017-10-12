import os
import gzip
import pickle
import numpy as np
from sklearn.decomposition import PCA


def whiten_input(data):
    # Using sklearn implementation, with option whiten=True
    print "whitening input..."
    zca_matrix = zca_whitening_matrix(data['train']['features'].T)
    mean_train = data['train']['features'].mean(axis=0)

    for s in data:
        data[s]['features'] = np.dot(data[s]['features']-mean_train, zca_matrix).astype('float32')

    return data

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix

    taken from: https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python

    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]

    return ZCAMatrix

def get_mnist():

    path = os.environ["MNIST_PKL_GZ"]
    if not os.path.exists(path):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, path)

    f = gzip.open(path, 'rb')
    try:
        split = pickle.load(f, encoding="latin1")
    except TypeError:
        split = pickle.load(f)
    f.close()

    which_sets = "train valid test".split()
    data = dict((which_set, dict(features=x.astype("float32"),
                                 targets=y.astype("int32")))
                for which_set, (x, y) in zip(which_sets, split))
        
    return data

def get_data(whiten=False, dataset='mnist'):

    data = None
    if dataset == 'mnist':
        data = get_mnist()
    else:
        raise ValueError("The dataset {} is not supported yet.")

    if whiten:
        data = whiten_input(data)

    return data