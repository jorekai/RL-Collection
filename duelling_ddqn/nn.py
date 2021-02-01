from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, BatchNormalization, Lambda
from tensorflow.python.keras.optimizer_v1 import Adam


class NN:
    """
    The biggest change for the duelling DQN is, that we must define a more complex DQN architecture
    The architecture must define our Value and Advantage Layers
    The way we define our model:
        <<<>>>
        The Keras functional API is a way to create models that are more flexible than the tf.keras.Sequential API.
        The functional API can handle models with NON-LINEAR topology, SHARED layers, and even MULTIPLE inputs or outputs.
        Read more here: https://keras.io/guides/functional_api/
        <<<>>>
    """

    def __init__(self, env, alpha: float = 0.001, decay: float = 0.0001):
        """
        We initialize our functional model, therefore we need Input Shape and Output Shape
        :param env:
        :param alpha:
        :param decay:
        """
        self.alpha = alpha
        self.decay = decay
        self.model = None
        # new to D-DDQN
        self.init_model(env.observation_space.shape[0], env.action_space.n)

    def init_model(self, input_shape: int, n_actions: int):
        inp = Input(shape=(input_shape,))
        layer_shared1 = Dense(32, activation='relu', kernel_initializer='he_uniform')(inp)
        layer_shared1 = BatchNormalization()(layer_shared1)
        layer_shared2 = Dense(32, activation='relu', kernel_initializer='he_uniform')(layer_shared1)
        layer_shared2 = BatchNormalization()(layer_shared2)

        layer_v1 = Dense(32, activation='relu', kernel_initializer='he_uniform')(layer_shared2)
        layer_v1 = BatchNormalization()(layer_v1)
        layer_a1 = Dense(32, activation='relu', kernel_initializer='he_uniform')(layer_shared2)
        layer_a1 = BatchNormalization()(layer_a1)
        # the value layer ouput is a scalar value
        layer_v2 = Dense(1, activation='linear', kernel_initializer='he_uniform')(layer_v1)
        # The advantage function subtracts the value of the state from the Q
        # function to obtain a relative measure of the importance of each action.
        layer_a2 = Dense(n_actions, activation='linear', kernel_initializer='he_uniform')(layer_a1)

        # the q layer combines the two streams of value and advantage function
        # the lambda functional layer can perform lambda expressions on keras layers
        # read more here : https://keras.io/api/layers/core_layers/lambda/
        # the lambda equation is defined in https://arxiv.org/pdf/1511.06581.pdf on equation (9)
        layer_q = Lambda(lambda x: x[0][:] + x[1][:] - K.mean(x[1][:]), output_shape=(n_actions,))(
            [layer_v2, layer_a2])

        self.model = Model(inp, layer_q)
        self.model.compile(optimizer=Adam(lr=self.alpha, decay=self.decay), loss='mse')

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
