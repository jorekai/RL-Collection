import numpy as np

from dqn.memory import ReplayMemory


class DQNAgent:
    """
    The DQN Agent
    """

    def __init__(self, env, memory: ReplayMemory, net, epsilon_init: float = 1):
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
        self.epsilon = epsilon_init

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
        x, y = [], []  # x: the input state vector, y: the target state vector
        experiences = self.memory.get_batch(batch_size)
        

