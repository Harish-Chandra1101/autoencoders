import logging


class FeatureExtractionLogger(object):
    def __init__(self, algorithm):
        self.algorithm = algorithm
        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
        self.logger = logging.getLogger(self.algorithm)
        fh = logging.FileHandler('feature_extraction.log')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s  %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)


class ClusteringLogger(object):
    def __init__(self, algorithm):
        self.algorithm = algorithm
        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
        self.logger = logging.getLogger(self.algorithm)
        fh = logging.FileHandler('cluster.log')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s  %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
