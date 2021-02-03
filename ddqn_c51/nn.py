from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.optimizer_v1 import adam


class NN:
    """
    The NN class wraps a keras Sequential model to reduce the interface methods
    Notice:
        Difference to dqn is just the setter and getter methods for the weights
    """
    def __init__(self, env, atoms, alpha: float = 0.001, decay: float = 0.0001):
        """
        We initialize our functional model, therefore we need Input Shape and Output Shape
        :param env:
        :param alpha:
        :param decay:
        """
        self.alpha = alpha
        self.decay = decay
        self.model = None
        self.atoms = atoms
        # new to D-DDQN
        self.init_model(env.observation_space.shape[0], env.action_space.n)

    def init_model(self, input_shape: int, n_actions: int):
        """
        Initializing our keras sequential model
        :return: initialized model
        """
        input = Input(shape=(input_shape,))
        h1 = Dense(64, activation='relu')(input)
        h2 = Dense(64, activation='relu')(h1)
        outputs = []
        for _ in range(n_actions):
            outputs.append(Dense(self.atoms, activation='softmax')(h2))
        self.model = Model(input, outputs)

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
