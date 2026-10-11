# Résultat test N0 (mini env 2 noeuds)
Entraînement : 100 000 pas, seed 0. Évaluation : seeds 1000-1029.

| Politique | Latence moy. (ms) | Part vers nœud lent |
|---|---|---|
| PPO | 100.6 | 0.26 |
| DQN | 105.3 | 0.49 |
| A2C | 115.2 | 0.46 |
| Round-robin | 799.2 | 0.50 |
| Toujours rapide | 1528.8 | 0.00 |
| Aléatoire | 827.7 | 0.49 |

Conclusion : N0 validé. PPO apprend à privilégier le nœud rapide et à déborder vers le lent.