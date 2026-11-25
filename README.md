# fog-rl-orchestration

Orchestration d'un réseau fog par apprentissage par renforcement (RL) : à chaque requête entrante, un agent choisit le nœud
(Fog A, B, C, D, E ou Cloud) qui optimise un compromis **latence / échéances / énergie / coût**. Les algorithmes **PPO, DQN et A2C**
(Stable-Baselines3) sont entraînés dans un simulateur Gymnasium, comparés à 5 heuristiques classiques, puis déployés derrière un
orchestrateur FastAPI sur 3 machines virtuelles, avec supervision Prometheus / Grafana.

Projet de fin d'études — cahier des charges v2.0 (6 octobre 2026). Question de recherche : *une politique de routage apprise
réduit-elle latence et énergie par rapport aux heuristiques, et ce gain se maintient-il sur un déploiement réel ?*

## Contenu

| Brique | Dossier | Rôle |
|---|---|---|
| Contrat MDP | `fogrl/state_spec.py`, `fogrl/reward.py` | observation (22 valeurs à 4 nœuds, 30 en disposition remplie), action, récompense |
| Simulateur | `fogrl/sim/` | environnement Gymnasium : files FIFO, énergie linéaire, batterie virtuelle, pannes, trafic (Poisson, pics, diurne, flash crowd, trace) |
| Politiques | `fogrl/policies/` | Random, Round-robin, Least-connections, Seuils statiques, Greedy + DQN / PPO / A2C derrière la même interface `select` |
| Apprentissage | `fogrl/training/` | curriculum N0 → N3, Optuna, ablation des poids de récompense |
| Évaluation | `fogrl/evaluation/` | matrices d'expériences, statistiques, figures, rapport de recette, comparaison simulation / réel |
| Prototype réel | `fogrl/worker/`, `fogrl/orchestrator/`, `fogrl/loadgen/` | nœuds HTTP, orchestrateur (repli, hot-swap, journal JSONL), générateur de charge |
| Calibration | `fogrl/calibration/` | mesure des VM et ajustement du modèle de service |
| Déploiement | `fogrl/deploy/`, `docker/`, `scripts/` | Compose (local, par VM, contrôle), Prometheus, Grafana, pile locale sans Docker |

## Démarrage rapide

```bash
python -m venv venv && source venv/bin/activate                       # Windows : venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
make install                  # dépendances + installation du paquet en mode éditable
make test-fast                # tests rapides
make local-smoke              # prototype complet SANS Docker : workers + orchestrateur + générateur de charge
```

Sous Windows, travaillez dans **WSL** (Ubuntu) : `make`, les scripts `*.sh` et la pile locale (`make local-smoke`, groupes de
processus POSIX) l'exigent, et le projet n'a été testé que sous Linux. Si vous commitez depuis Windows, le bit exécutable des
scripts est perdu : c'est sans conséquence, car les scripts sont appelés par `bash …` et l'image Docker applique `chmod +x`.
`make help` liste toutes les cibles.

## Parcours de travail

```bash
make sim-baselines            # les 5 heuristiques sur tous les scénarios
make train-n1                 # contrôle : PPO/DQN/A2C doivent battre Round-robin et Least-connections
make train-n3                 # modèles livrés (3, 4 ou 6 nœuds, randomisation de domaine)
make sim-full && make report  # matrice complète puis critères de recette (ENF-02, ENF-03, H1, H2, OS-4)
python scripts/gen_compose.py --hosts vm1=IP1,vm2=IP2,vm3=IP3          # fichiers Compose / Prometheus / Grafana
# puis, sur les VM : make calibrate ; make calibration-fit DATA=… ; make campaign ; make compare   (calibration, campagne, jalon G4)
```

Le détail pas à pas (installation des VM, déploiement, tests d'exigences, dépannage) est dans [`docs/runbook.md`](docs/runbook.md).

## Documentation

- [`docs/avancement.md`](docs/avancement.md) — matrice cahier des charges → code → tests, avec ce qui reste à mesurer sur les VM
- [`docs/runbook.md`](docs/runbook.md) — de l'installation à la recette
- [`docs/architecture.md`](docs/architecture.md) — composants, API HTTP, métriques, topologies
- [`docs/mdp.md`](docs/mdp.md) — état, action, récompense, transitions, curriculum
- [`docs/decisions.md`](docs/decisions.md) — décisions de conception (D1 à D12) et écarts justifiés par rapport au cahier des charges

## Reproductibilité

Graines d'entraînement 0–99, d'évaluation 1000 et plus. Chaque modèle (`models/<niveau>/<algo>/seed<k>/`) embarque `contract.json`
(ordre des nœuds, normalisation, hyperparamètres, commit, versions). Les modèles, `runs/`, `results/` et `figures/` sont
régénérables et ignorés par Git ; les environnements `configs/env/` sont, eux, versionnés. Les fichiers de `deploy/` sont générés
(adresses des VM propres à chaque installation).

## Intégration continue

`.github/workflows/ci.yml` : tests rapides et lents, puis génération de `deploy/`, validation `docker compose config`, construction
de l'image du worker et démarrage d'un conteneur.
