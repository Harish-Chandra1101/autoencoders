from sklearn.cluster import KMeans


class Cluster:
    def __init__(self, data, cluster_config):
        self.__data = data
        self.__cluster_config = cluster_config

    def kmeans(self):
        config = self.__cluster_config['kmeans']
        cluster_labels = KMeans(n_clusters=config['num_clusters']).fit_predict(self.__data)
        return cluster_labels
