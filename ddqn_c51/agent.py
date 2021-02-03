import math

import numpy as np

from ddqn_c51.memory import ReplayMemory
from ddqn_c51.nn import NN


class C51Agent:
    """
    The DDQN Agent, notice the difference to DQN lies in the use of a target network
    """

    def __init__(self,
                 env,
                 memory: ReplayMemory,
                 net: NN,
                 target_net: NN,
                 atoms: int = 51,
                 epsilon_init: float = 1,
                 gamma: float = 0.99,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.999):
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
        self.atoms = atoms
        # C51
        self.reward_min = -5
        self.reward_max = 5
        self.atoms = atoms
        self.delta_z = float(self.reward_max - self.reward_min) / (self.atoms - 1)
        self.distribution = [self.reward_min + i * self.delta_z for i in range(self.atoms)]

    def act(self, state):
        """
        Choose an action according to our epsilon-greedy policy
        :param state: the state observation of our environment we need to predict from
        :return: action choice
        """
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        return self.dist_to_action(self.net.predict(state))

    def dist_to_action(self, distribution):
        stacked_dist = np.vstack(distribution)
        q = np.sum(np.multiply(stacked_dist, np.array(self.distribution)), axis=1)
        return np.argmax(q)

    def replay(self, batch_size: int):
        """
        The basic idea to stabilize DQN algorithm is by replay past experiences in batches randomly
        :param batch_size: int > 0
        :return: void
        """
        states, actions, rewards, next_states, dones = zip(*self.memory.get_batch(batch_size))
        z = self.net.predict(next_states)
        z_ = self.target_network.predict(next_states)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.distribution)), axis=1)
        print(q)
        q = q.reshape((batch_size, self.env.action_space.n), order='F')
        next_actions = np.argmax(q, axis=1)
        m_prob = [np.zeros((batch_size, self.atoms))
                  for _ in range(self.env.action_space.n)]
        for i in range(batch_size):
            if dones[i]:
                Tz = min(self.reward_max, max(self.reward_min, rewards[i]))
                bj = (Tz - self.reward_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[actions[i]][i][int(l)] += (u - bj)
                m_prob[actions[i]][i][int(u)] += (bj - l)
            else:
                for j in range(self.atoms):
                    Tz = min(self.reward_max, max(
                        self.reward_min, rewards[i] + self.gamma * self.distribution[j]))
                    bj = (Tz - self.reward_min) / self.delta_z
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[actions[i]][i][int(
                        l)] += z_[next_actions[i]][i][j] * (u - bj)
                    m_prob[actions[i]][i][int(
                        u)] += z_[next_actions[i]][i][j] * (bj - l)
        self.net.fit(states, m_prob)

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
