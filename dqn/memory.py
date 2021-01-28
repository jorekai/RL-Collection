from collections import deque
from random import random

from run import Experience


class ReplayMemory:
    def __init__(self, size: int):
        """
        We initialize a replay memory to sample experiences from it and append past experiences
        :param size: int, maximum length of experiences to remember
        """
        self.memory = deque(maxlen=size)

    def append(self, experience: Experience):
        """
        Append a experience tuple to our memory object
        :param experience: Tuple(state, action, reward, next_state, done)
        :return: void
        """
        if len(self.memory) >= self.memory.maxlen:  # pop oldest element if our memory is full before inserting
            self.memory.popleft()
        self.memory.append(experience)

    def get_batch(self, batch_size: int):
        """
        Return a batch of experiences of our expected batch size if possible
        :param batch_size: int > 0
        :return: List[Tuple(state, action, reward, next_state, done)]
        """
        return random.sample(self.memory, min(len(self.memory), batch_size))
