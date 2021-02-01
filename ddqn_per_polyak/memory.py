import heapq
from itertools import count


class ReplayMemory:
    """
    PER: Thanks to the python module heapq we do not need to implement our own priority heap
    INFO: heapq works on standard python arrays
    """

    def __init__(self, size: int):
        """
        We initialize a replay memory to sample experiences from it and append past experiences
        :param size: int, maximum length of experiences to remember
        """
        self.maxlen = size
        self.memory = []
        # for equal priorities we must define a tiebreaker
        self.tiebreaker = count()

    def append(self, experience, TDerror):
        """
        Append a experience tuple to our memory object
        :param TDerror: the temporal difference error defines priority
        :param experience: Tuple(state, action, reward, next_state, done)
        :return: void
        """
        heapq.heappush(self.memory, (-TDerror, next(self.tiebreaker), experience))
        if len(self.memory) > self.maxlen:
            self.memory = self.memory[:-1]
        heapq.heapify(self.memory)

    def get_batch(self, batch_size: int):
        """
        Return a batch of experiences of our expected batch size if possible
        :param batch_size: int > 0
        :return: List[Tuple(state, action, reward, next_state, done)]
        """
        batch = heapq.nsmallest(batch_size, self.memory)
        batch = [experience for _, _, experience in batch]  # return the S,A,R,S_,D and ignore the others
        self.memory = self.memory[batch_size:]
        return batch
