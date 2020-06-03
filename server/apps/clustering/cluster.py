import numpy as np
from sklearn.cluster import KMeans
from apps.clustering.utils import remove_nan


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

    def kmeans(self):
        data = Cluster.truncate_nan_values(self.__data)
        cluster_labels = KMeans(n_clusters=self.__cluster_config['num_clusters']).fit_predict(data)
        return cluster_labels

    @staticmethod
    def truncate_nan_values(data):
        data = data[:, ~np.isnan(data).any(axis=0)]
        return data
