"""Processus d'arrivée : pour chaque scénario, comment tirer les instants.

Signature commune : f(rng, lam, n, duration_ms) -> tableau d'instants (ms)
  rng         : générateur NumPy déjà initialisé (reproductible)
  lam         : débit moyen en requêtes par ms
  n           : nombre de requêtes voulu (ou None)
  duration_ms : durée voulue (ou None)
"""
import numpy as np


def poisson_times(rng, lam, n=None, duration_ms=None):
    """Processus de Poisson : écarts exponentiels de moyenne 1/lam."""
    if n is not None:
        # L'instant de la k-ième arrivée est la somme des k premiers écarts.
        return np.cumsum(rng.exponential(1.0 / lam, size=n))

    # Mode durée : on tire des blocs jusqu'à dépasser la durée, puis on tronque.
    chunks, t = [], 0.0
    block = max(1024, int(lam * duration_ms * 1.2) + 1)
    while t < duration_ms:
        ts = t + np.cumsum(rng.exponential(1.0 / lam, size=block))
        chunks.append(ts)
        t = ts[-1]
    times = np.concatenate(chunks)
    return times[times < duration_ms]


# Registre : ajouter un scénario = ajouter une fonction et une ligne ici.
ARRIVAL_PROCESSES = {
    "poisson": poisson_times,
}