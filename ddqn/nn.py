from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.optimizer_v1 import adam


class NN:
    """
    The NN class wraps a keras Sequential model to reduce the interface methods
    Notice:
        Difference to dqn is just the setter and getter methods for the weights
    """
    def __init__(self, alpha: float, decay: float):
        """
        Initialize a keras model with hyperparams
        :param alpha: learning factor
        :param decay: learning decay factor
        """
        self.alpha = alpha
        self.decay = decay
        self.model = self.init_model()

    def init_model(self):
        """
        Initializing our keras sequential model
        :return: initialized model
        """
        model = Sequential()
        model.add(Dense(64, input_dim=4, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=self.alpha))
        return model

    def predict(self, *args, **kwargs):
        """
        By wrapping the keras predict method we can handle our net as a standalone object
        :param args: interface to keras.model.predict
        :return: prediction
        """
        return self.model.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """
        By wrapping the keras fit method we can handle our net as a standalone object
        :param args: interface to keras.model.fit
        :return: history object
        """
        return self.model.fit(*args, **kwargs)

    def get_weights(self):
        """
        Passing the arguments to keras get_weights
        """
        return self.model.get_weights()

    def set_weights(self, *args, **kwargs):
        """
        Passing the arguments to keras set_weights
        """
        self.model.set_weights(*args, *kwargs)
