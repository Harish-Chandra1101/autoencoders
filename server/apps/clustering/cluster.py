from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from server.apps.logging.logger import ClusteringLogger


class Cluster:
    """
    Performs clustering of the given data
    """

    def __init__(self, data, cluster_config):
        """
        :param data: data from autoencoders feature extraction
        :param cluster_config:
        Sample cluster config json for kmeans
        {
            "num_clusters": 3
        }
        """
        self.__data = data
        self.__cluster_config = cluster_config
        self.cluster_logger = None

    def kmeans(self):
        self.cluster_logger = ClusteringLogger("kmeans")
        self.cluster_logger.logger.info("Cluster Config Received is {}".format(self.__cluster_config))

        cluster_labels = KMeans(n_clusters=self.__cluster_config['num_clusters']).fit_predict(self.__data)

        self.cluster_logger.logger.info("Clustering completed")
        self.cluster_logger.logger.info("Returning clustering labels")

        return cluster_labels
