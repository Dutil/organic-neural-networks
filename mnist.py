import os
import gzip
import pickle
import numpy as np
from sklearn.decomposition import PCA
             
def whiten_input(data):
    
    # Using sklearn implementation, with option whiten=True
    print "whitening input..."
    pca = PCA(whiten=True, svd_solver='full')
    pca.fit(data['train']['features'])
             
    for s in data:
        data[s]['features'] = pca.transform(data[s]['features'])
        
    return data

def get_data(whiten=False):

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
    if whiten:
        data = whiten_input(data)
        
    return data
