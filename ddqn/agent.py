import math

import numpy as np

from ddqn.memory import ReplayMemory
from ddqn.nn import NN


class DDQNAgent:
    """
    The DDQN Agent, notice the difference to DQN lies in the use of a target network
    """

    def __init__(self,
                 env,
                 memory: ReplayMemory,
                 net: NN,
                 target_net: NN,
                 epsilon_init: float = 0.5,
                 gamma: float = 0.99,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.99):
        """
        We initialize necessary objects and hyperparams
        :param env: gym environment
        :param memory: a replay memory buffer
        :param net: local network
        :param target_net: target network
        :param epsilon_init: initial exploration factor
        :param gamma: the discount for future q-values
        :param epsilon_min: the exploration factor to stop at
        :param epsilon_decay: a decay factor (0,1)
        """
        self.env = env
        self.memory = memory
        self.net = net
        self.target_network = target_net  # <---- new to ddqn, initialize
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
            target_next_state = self.net.predict(next_state)
            target_next_state_offline = self.target_network.predict(next_state)
            a = np.argmax(target_next_state)  # select the maximum action from Online network

            # new to ddqn, we evaluate the action from our offline network
            target[0][action] = reward if done else reward + self.gamma * target_next_state_offline[0][a]
            x.append(state[0])
            y.append(target[0])
        self.net.fit(np.array(x), np.array(y), verbose=0)
        self.update_target_model()

    def decay_epsilon(self):
        """
        Decay our exploration parameter logarithmic
        :param step: environment step as integer
        :return: void
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_model(self):
        """
        This method is new to DDQN, just update the weights every n-th step
        :return: void
        """
        self.target_network.set_weights(self.net.get_weights())
