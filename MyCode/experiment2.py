import numpy as np


class Kmeans:
    """
    the implementation for Kmeans
    """

    def __init__(self, disMeature, loss):
        self.distance = disMeature
        self.loss = loss
        self.centers = None
        self.clusters = {}

    def fit(self, input, num_centers, m):
        """
        :param input:  the input dataset  with shape (numbers, features)
                num_centers: the number of center
                m means the maximum iterations
        :return: the clusters results
        """
        centers_index = np.random.choice(len(input), num_centers, replace=False)
        self.centers = input[centers_index] # with shape (num_centers, features)
        # delete these centers from dataset to avoid resampling
        # input = np.delete(input, centers_index, axis=0) no use?
        min_loss = float("inf")
        assert len(self.centers) == num_centers
        for iter in range(m):
            # each time need init the clusters
            for i in range(num_centers):
                # init the clusters with a list
                self.clusters[i] = list()
            # the E step
            for xi in input:
                # init the distance
                min_distance = float('inf')
                cluster = None
                for index, center in enumerate(self.centers):
                    distance = self.distance(xi, center)
                    if distance <= min_distance:
                        min_distance = distance
                        cluster = index
                self.clusters[cluster].append(xi)

            # the M step
            for cluster, data in self.clusters.items():
                self.centers[cluster] = np.mean(np.array(data), axis=0)

            loss = self.loss(self.clusters, self.centers)
            print("{} iters: loss: {}".format(iter+1, loss))

            if loss < min_loss:
                min_loss = loss
            else:
                print("{} iters stop".format(iter))
                break

        return self.clusters, self.centers

def disMeature(vecA, vecB):
    """
    the Euclidean distance for vectors

    :return: the distance between vecA and vecB
    """
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def loss(clusters, centers):
    """
    compute the total distortion measure for the clusters results
    :param clusters: input clusters
    :return: the total measure loss
    """
    loss = 0
    for index, dataset in clusters.items():
        loss += np.sum(np.power(np.array(dataset) - centers[index], 2))
    return loss

if __name__ == '__main__':
    X = np.load('./data/Gaussian_train.npy')
    print(X.shape)
    K_means = Kmeans(disMeature=disMeature, loss=loss)
    clusters, centers = K_means.fit(X, 2, 10)











