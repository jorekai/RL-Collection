import math
from copy import deepcopy

import numpy as np

from dqn.memory import ReplayMemory
from dqn.nn import NN


class DDQNAgent:
    """
    The DDQN Agent, notice the difference to DQN lies in the target network
    """

    def __init__(self,
                 env,
                 memory: ReplayMemory,
                 net: NN,
                 epsilon_init: float = 1,
                 gamma: float = 0.99,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.999):
        """
        The initialization method of our agent
        :param env: gym environment
        :param memory: replay memory of any size
        :param net: neural network model of any size
        :param epsilon_init: float value for exploration initialisation
        """
        self.env = env
        self.memory = memory
        self.net = net
        self.target_network = deepcopy(net)
        # hyperparameters
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

    def act(self, state):
        """
        Choose an action according to our epsilon-greedy policy
        :param state: the state observation of our environment we need to predict from
        :return: action choice
        """
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.net.predict(state))

    def replay(self, batch_size: int):
        """
        The basic idea to stabilize DQN algorithm is by replay past experiences in batches randomly
        :param batch_size: int > 0
        :return: void
        """
        x, y = [], []  # x: the input state vector, y: the target state vector
        experiences = self.memory.get_batch(batch_size)

        for state, action, reward, next_state, done in experiences:
            target = self.net.predict(state)
            target[0][action] = reward if done else reward + self.gamma * np.max(self.net.predict(next_state)[0])
            x.append(state[0])
            y.append(target[0])
        self.net.fit(np.array(x), np.array(y), batch_size=len(x), verbose=0)

    def decay_epsilon(self, step: int):
        """
        Decay our exploration parameter logarithmic
        :param step: environment step as integer
        :return: void
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((step + 1) * self.epsilon_decay)))

    def update_target_model(self):
        """
        This method is new to DDQN, just update the weights every n-th step
        :return: void
        """
        self.target_network.set_weights(self.net.get_weights())
