

class DQNAgent:
    """
    The DQN Agent
    """
    def __init__(self, memory, net):
        """
        The initialization method of our agent
        :param memory: replay memory of any size
        :param net: neural network model of any size
        """
        self.memory = memory
        self.net = net

    def act(self):
        # implement the action method
        pass
