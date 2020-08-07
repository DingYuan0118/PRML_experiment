import numpy as np
import matplotlib.pyplot as plt
from MyPackages.data_utils import data_generator

class MaxLikelyHood:
    """
    a maxlikely hood estimate. usually use gaussian estimate
    """
    def __init__(self):
        # init the mean and variance
        self.sigma = None
        self.mean = None
        self.model = {} # the key of the model dict represents the class, must be int
                        # the value of the dict is also a dict indicating the estimate for each class

    def fit(self, input):
        """
        :param input: the training dataset for classification with dict shape {classa:[nums, features], classb:[nums, features]}
        :return: model
        """
        for key, data in input.items():
            # u is mean of data, sigma is the variance
            # They are both max likelyhood estimate
            u = np.mean(data, axis=0)
            sigma = np.cov(data, rowvar=False)
            self.model[key] = {"mean": u, "var": sigma}
            print("class {}:MLE mean:{} var:{}".format(key, u, sigma))

        return self.model

    def predict(self, input):
        """

        :param input: the test dataset for classification with dict shape {classa:[nums, features], classb:[nums, features]}
        :return: the precision of model for the test data
        """
        predict_class = []
        labels = []
        total_data = []
        for class_name , data in input.items():
            D = data.shape[1] # the Dimension of data
            total_data.append(data)
            for xi in data:
                predict_individual = 0 # predict density for each data
                labels.append(class_name)  # with a shape (total nums,) matches the predict_class
                predict_class_individual = None
                for predict_name, mle in self.model.items():
                    # the key in self.model matches the key in input
                    u = mle["mean"]
                    sigma = mle["var"]
                    # high dimension gaussian distribution
                    predict_density = 1 / ((2 * np.pi)**(D/2)) * 1 / np.sqrt(np.linalg.det(sigma)) * np.exp(-1/2 * (xi - u) @ (np.linalg.inv(sigma)) @ (xi - u).T)
                    if predict_density >= predict_individual:
                        predict_individual = predict_density
                        predict_class_individual = predict_name
                predict_class.append(predict_class_individual) # with a shape (total nums, )'
        total_data = np.concatenate(total_data, axis=0)
        true_index = np.array(predict_class) == np.array(labels)
        false_index = np.array(predict_class) != np.array(labels)
        precision = np.sum(true_index) / len(labels)
        positive = total_data[true_index]
        negetive = total_data[false_index]

        return precision , positive, negetive


if __name__ == '__main__':
    mean_a = [-2,0]
    mean_b = [2,0]
    cova = [[1, 0.5], [0.5, 1]]
    covb = [[1, -0.5], [-0.5, 1]]
    num = 1000
    classa, classb = data_generator(mean_a, mean_b, cova, covb, num, num)
    train_data = {1 : classa, 2 : classb}
    testa, testb  = data_generator(mean_a, mean_b, cova, covb, 400, 400)
    test_data = {1 : testa, 2 : testb}

    MLE = MaxLikelyHood()
    MLE.fit(train_data)
    precision , positive, negetive = MLE.predict(test_data)
    print(precision)
    plt.figure("train data")
    plt.scatter(classa[:, 0], classa[:, 1], label='classa')
    plt.scatter(classb[:, 0], classb[:, 1], label='classb')
    plt.legend()
    plt.title("train data")

    plt.figure("test result")
    plt.scatter(positive[:,0], positive[:,1], label="true")
    plt.scatter(negetive[:,0], negetive[:,1], label="false")
    plt.legend()
    plt.title("result")
    plt.show()

