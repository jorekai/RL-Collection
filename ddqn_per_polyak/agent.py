import math

import numpy as np

from ddqn_per_polyak.memory import ReplayMemory
from ddqn_per_polyak.nn import NN


class PolyakAgent:
    """
    The PolyakAgent Agent, notice the difference to DQN lies in the target network
    """

    def __init__(self,
                 env,
                 memory: ReplayMemory,
                 net: NN,
                 target_net: NN,
                 epsilon_init: float = 1,
                 gamma: float = 0.99,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.05,
                 tau: float = 0.8):
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
        self.target_network = target_net  # <---- new to ddqn, initialize
        # hyperparameters
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau

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
            # new to ddqn, we get the maximum state,action value from our target network
            target[0][action] = reward if done else reward + self.gamma * np.max(
                self.target_network.predict(next_state)[0])
            x.append(state[0])
            y.append(target[0])
        self.net.fit(np.array(x), np.array(y), batch_size=len(x), verbose=0)
        self.update_target_model(self.tau)

    def decay_epsilon(self):
        """
        Decay our exploration parameter logarithmic
        :param step: environment step as integer
        :return: void
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_model(self, tau: float):
        """
        The polyak update method supports a tau argument to fine tune the update value
        :type tau: float value for update strength
        :return: void
        """
        local_weights = self.net.get_weights()
        target_weights = self.target_network.get_weights()
        # polyak update means to scale down the update factor of our weights
        polyak_weights = [tau * lw + ((1 - tau) * tw) for lw, tw in zip(local_weights, target_weights)]
        self.target_network.set_weights(polyak_weights)

    def get_error(self, transition):
        state, action, reward, next_state, done = transition
        target = self.net.predict(state)
        target_old = np.array(target)
        # new to ddqn, we get the maximum state,action value from our target network
        target[0][action] = reward if done else reward + self.gamma * np.max(self.target_network.predict(next_state)[0])
        td_error = np.abs(target[0][action] - target_old[0][action])
        return td_error
