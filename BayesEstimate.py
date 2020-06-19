import numpy as np
import matplotlib.pyplot as plt

def MaxLikelyhoodEs(num_classes, prior, input, model):
    """

    :param num_classes: the num of classes for classification task
    :param prior: a list of len(num_clases). prior probability of each classes
    :param input: the dataset for classification. maybe a numpy array?
    :param model: the model for classification

    :return: the maximum likelihood value of params of model with respect to the input dataset
    """
    pass

    return

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

