import numpy as np
from server.apps.activations.activations import Activation


class Autoencoder(object):
    """
    This class is used to perform feature extraction with the help of autoencoders
    Attributes:
        data : The preprocessed csv data
        config: The json which contains the parameters for the autoencoders
    """

    def __init__(self, config, data):
        """

        :param config:
        Sample Config json
        {
            "n_visible": 4,
            "n_hidden": 50,
            "numpy_rng": "",
            "activation": "sigmoid",
            "learning_rate": 0.01,
            "v_bias": "",
            "h_bias": "",
            "epochs": 10000
        }
        :param data: preprocessed csv data
        """

        self.__config = config
        self.__data = data

        self.__data = np.array(self.__data)
        if self.__config["numpy_rng"] is "":
            self.__numpy_rng = np.random.RandomState(1234)
        else:
            self.__numpy_rng = self.__config["numpy_rng"]

        self.n_visible = self.__config["n_visible"]
        self.n_hidden = self.__config["n_hidden"]

        norm_factor = 1. / self.n_visible
        self.weights = np.array(self.__numpy_rng.uniform(low=-norm_factor, high=norm_factor, size=(self.n_visible, self.n_hidden)))
        self.weights = np.array(self.weights)

        if self.__config["h_bias"] is "":
            self.h_bias = np.ones(self.__config["n_hidden"])
        else:
            self.h_bias = self.__config["h_bias"]

        if self.__config["v_bias"] is "":
            self.v_bias = np.ones(self.__config["n_visible"])
        else:
            self.v_bias = self.__config["v_bias"]

        if self.__config["epochs"] is "":
            self.epochs = 1000
        else:
            self.epochs = self.__config["epochs"]

        self.W_prime = self.weights.T
        self.__hidden = None
        self.activation_function = self.__config["activation"]
        self.learning_rate = self.__config["learning_rate"]
        self.activation = Activation()
        self.error_list = []
        self.epoch_list = [i for i in range(0, self.epochs)]
        self.training_complete = 0

    @property
    def config(self):
        return self.__config

    @property
    def data(self):
        return self.__data

    @property
    def hidden(self):
        return self.__hidden

    def get_hidden_values(self):
        """
        The encoding part
        :return: hidden layer activations
        """
        h = np.dot(self.__data, self.weights) + self.h_bias
        return getattr(self.activation, self.activation_function)(h)

    def get_reconstructed_input(self):
        """
        The decoding part
        :return: Reconstrcucted input
        """
        return getattr(self.activation, self.activation_function)(np.dot(self.__hidden, self.W_prime) + self.v_bias)

    def train(self):
        """
        Train the autoencoder
        """
        self.__hidden = self.get_hidden_values()
        reconstructed_input = self.get_reconstructed_input()

        error = np.sum((self.__data - reconstructed_input) ** 2)
        print("Error " + str(error))
        self.error_list.append(error)

        L_h2 = self.__data - reconstructed_input
        L_h1 = np.dot(L_h2, self.weights) * self.__hidden * (1 - self.__hidden)

        L_vbias = L_h2
        L_hbias = L_h1

        L_W = np.dot(self.__data.T, L_h1) + np.dot(L_h2.T, self.__hidden)

        self.weights += self.learning_rate * L_W
        self.h_bias += self.learning_rate * np.mean(L_hbias, axis=0)
        self.v_bias += self.learning_rate * np.mean(L_vbias, axis=0)

    def train_ae(self):
        """
        Train the autoencoder network for specific number of epochs
        """
        for epoch in range(self.epochs):
            self.train()
            self.training_complete = 1

    def get_features(self):
        if self.training_complete:
            return self.__hidden
        else:
            return "Please train the network to get the features"

    def get_final_weights(self):
        if self.training_complete:
            return self.weights
        else:
            return "Please train the network to get the final weights"

    def get_final_visible_bias(self):
        if self.training_complete:
            return self.v_bias
        else:
            return "Please train the network to get the final hidden bias"

    def get_final_hidden_bias(self):
        if self.training_complete:
            return self.h_bias
        else:
            return "Please train the network to get the final visible"















