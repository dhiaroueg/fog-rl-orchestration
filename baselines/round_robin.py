"""Politique Round-robin : alterne entre les nœuds."""

from .base import Policy


class RoundRobinPolicy(Policy):
    name = "Round-robin"

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.next_action = 0

    def select(self, obs) -> int:
        del obs
        action = self.next_action
        self.next_action = (self.next_action + 1) % self.n_actions
        return action