from sklearn.cluster import KMeans


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
        cluster_labels = KMeans(n_clusters=self.__cluster_config['num_clusters']).fit_predict(self.__data)
        return  cluster_labels