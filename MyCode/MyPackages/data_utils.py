from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from cv2 import imread
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def data_generator(meana, meanb, cova, covb, num_classa, num_classb):
    """

    :param meana: the mean of Gaussian distribution of class a
    :param meanb: the mean of Gaussian distribution of class b
    :param cova: the covariance matrix of class a
    :param covb: the covariance matrix of class b
    :param num_classa: the number of data points of class a
    :param num_classb: the number of data points of class b

    :return: the data of class a and class b
    """
    classa = np.random.multivariate_normal(meana, cova, size=num_classa)
    classb = np.random.multivariate_normal(meanb, covb, size=num_classb)
    data = (classa, classb)
    return data