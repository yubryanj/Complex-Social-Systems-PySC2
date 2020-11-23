import numpy as np

class RandomAgent:
    """
    Agent that takes a random action at every step
    """

    def __init__(self, environment):
        self.action_space = environment.action_space.n

    def predict(self, observations):
        return [np.random.randint(self.action_space)], None