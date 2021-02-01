from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.optimizer_v1 import adam


class NN:
    """
    The NN class wraps a keras Sequential model to reduce the interface methods
    """
    def __init__(self, alpha: float, decay: float):
        self.alpha = alpha
        self.decay = decay
        self.model = self.init_model()

    def init_model(self):
        """
        Initializing our keras sequential model
        :return: initialized model
        """
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.compile(loss='mse', optimizer=adam(lr=self.alpha, decay=self.decay))
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
