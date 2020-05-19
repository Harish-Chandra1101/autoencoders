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
        config = self.__cluster_config['kmeans']
        cluster_labels = KMeans(n_clusters=config['num_clusters'], random_state=0).fit_predict(self.__data)
        return cluster_labels

    def em(self):
        config = self.__cluster_config['em']
        cluster_labels = GaussianMixture(n_components=config['num_clusters'], covariance_type=config['cov_type'], max_iter=self.config['num_epochs'], init_params=config['init_params']).fit_predict(self.__data)
        return cluster_labels

