import numpy as np
import matplotlib.pyplot as plt

def MaxLikelyhoodEs(num_classes, prior, input, model="Gaussian"):
    """

    :param num_classes: the num of classes for classification task
    :param prior: a list of len(num_clases). prior probability of each classes
    :param input: the dataset for classification. a num_classes length tuple: (inputa, inputb...)
    :param model: the model for classification

    :return: the maximum likelihood value of params of model with respect to the input dataset
    """
    assert len(input) == num_classes
    if model == "Gaussian":


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

if __name__ == '__main__':
    mean_a = [-1,0]
    mean_b = [1,0]
    cova = [[1, 0.5], [0.5, 1]]
    covb = [[1, -0.5], [-0.5, 1]]
    num = 1000
    classa, classb = data_generator(mean_a, mean_b, cova, covb, num, num)
    fig, ax = plt.subplots()
    ax.scatter(classa[:,0], classa[:,1])
    ax.scatter(classb[:,0], classb[:,1])
    fig.show()