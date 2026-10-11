import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ToyFogEnv(gym.Env):
    """2 noeuds : rapide (vitesse 2) et lent (vitesse 1). Recompense = -latence."""

    def __init__(self, n_requests=500, mean_interarrival=20.0):
        super().__init__()
        self.speeds = np.array([2.0, 1.0])
        self.n_requests = n_requests
        self.mean_ia = mean_interarrival
        self.t_ref = 200.0
        self.action_space = spaces.Discrete(2)  # 0 = rapide, 1 = lent
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)

    def _obs(self):
        q = np.clip(self.backlog / self.t_ref, 0, 1)
        return np.array([q[0], q[1], self.size / 100.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.backlog = np.zeros(2)
        self.t = 0
        self.size = self.np_random.uniform(5, 100)
        return self._obs(), {}

    def step(self, action):
        service = self.size / self.speeds[action]
        resp = self.backlog[action] + service  # attente + traitement
        self.backlog[action] += service
        reward = -resp / self.t_ref
        dt = self.np_random.exponential(self.mean_ia)
        self.backlog = np.maximum(self.backlog - dt, 0)
        self.size = self.np_random.uniform(5, 100)
        self.t += 1
        truncated = self.t >= self.n_requests
        return self._obs(), reward, False, truncated, {"resp": resp}