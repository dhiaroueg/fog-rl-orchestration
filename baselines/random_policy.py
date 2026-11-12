import numpy as np
from .base import Policy


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, n_actions=2):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def select(self, obs):
        return int(self.rng.integers(self.n_actions))