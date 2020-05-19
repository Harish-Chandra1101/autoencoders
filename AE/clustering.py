from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class Cluster:

    def __init__(self, data, cluster_config):
        self.__data = data
        self.__cluster_config = cluster_config

    @property
    def cluster_config(self):
        return self.__cluster_config

    def kmeans(self):
        cluster_labels = KMeans(n_clusters=self.__cluster_config['num_clusters'], random_state=0).fit_predict(self.__data)
        return cluster_labels

    def em(self):
        cluster_labels = GaussianMixture(n_components=self.__cluster_config['num_clusters'], covariance_type=self.__cluster_config['cov_type'], max_iter=self.__cluster_config['num_epochs'], init_params=self.__cluster_config['init_params']).fit_predict(self.__data)
        return cluster_labels

