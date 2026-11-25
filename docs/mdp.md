# Spécification du MDP (OS-1)

Formalisation du problème de routage telle qu'implémentée dans `fogrl/state_spec.py` (contrat d'observation),
`fogrl/reward.py` (récompense) et `fogrl/sim/env.py` (transitions). Elle suit le CDC §5.2 ; les écarts sont expliqués dans
[`decisions.md`](decisions.md).

## 1. Pas de décision et épisode

- **Pas de décision** : l'arrivée d'une requête. L'agent choisit un nœud, la requête est soumise, puis l'environnement avance
  jusqu'à l'arrivée suivante.
- **Épisode** : 2 000 requêtes à l'entraînement, 5 000 à l'évaluation. Un épisode se termine par une *troncature* (limite de
  temps), jamais par un état terminal ; les métriques de l'épisode sont dans `info["episode_metrics"]`.
- **Facteur d'actualisation** : γ = 0,99 (valeur des configurations `configs/train/*.yaml`).
- **Graines** : entraînement 0–99, évaluation ≥ 1 000 (jamais mélangées).

## 2. Espace d'observation

Vecteur de `4·N + 6` valeurs `float32`, toutes bornées dans [0, 1], pour N emplacements de nœuds :

| Bloc | Valeurs | Normalisation (constantes de `Normalization`) |
|---|---|---|
| Par nœud (×N) | `queue_ms` | travail en attente / 1 000 ms |
| | `cpu_load` | charge CPU lissée (moyenne mobile exponentielle, τ = 1 s), déjà dans [0, 1] |
| | `net_rtt_ms` | RTT mesuré (moyenne mobile, α = 0,3) / 200 ms |
| | `battery_fraction` | batterie virtuelle restante ; 1,0 pour un nœud sur secteur |
| Requête (5) | `type_light`, `type_medium`, `type_heavy` | one-hot |
| | `work_ms` | travail de la requête / 300 ms |
| | `deadline_ms` | échéance / 2 000 ms |
| Global (1) | `arrival_rate_rps` | débit d'arrivée récent (fenêtre glissante) / 200 req/s |

- **Disposition compacte** (N0–N2) : N = nombre de nœuds actifs ; 4 nœuds → **22 valeurs**, comme dans le CDC.
- **Disposition remplie** (N3, modèles livrés) : N = 6 emplacements `fogA, fogB, fogC, fogD, fogE, cloud` → **30 valeurs**. Un
  emplacement vide est encodé « saturé » (file 1, CPU 1, RTT 1, batterie 0) et son action est masquée à l'inférence.
- Toute mesure négative ou non finie est refusée (`ValueError`) : l'orchestrateur bascule alors sur son repli plutôt que de
  laisser l'agent décider sur une donnée corrompue.
- Les mesures sont celles du dernier sondage (≤ 200 ms), sauf `in_flight`, absent du vecteur (voir D2 et D5).

Les noms des 22 ou 30 valeurs sont produits par `observation_feature_names` (`fogA.queue_ms, fogA.cpu_load, …`).

## 3. Espace d'actions

Discret, `Discrete(N)` : l'indice est la position de l'emplacement dans la configuration (ordre de `pool6.yaml`).
Une action vers un emplacement vide ou en panne est un **rejet** à l'entraînement ; à l'inférence, ces actions sont masquées.

## 4. Récompense

```
r = −( w1·T/Tref + w2·E/Eref + w3·C/Cref + w4·[T > échéance] + w5·[rejet] )
```

| Terme | Définition | Poids initiaux |
|---|---|---|
| `T` | latence de bout en bout (attente + service + réseau), **plafonnée** à 3 × l'échéance | `w1 = 1,0` (Tref = 200 ms) |
| `E` | énergie dynamique fog de la requête : `(P_max − P_idle) / serveurs × durée de service` (nulle pour le cloud) | `w2 = 0,5` (Eref = 1 J) |
| `C` | coût cloud de la requête | `w3 = 0,3` (Cref = 1) |
| échéance | 1 si `T` dépasse l'échéance ou si la requête est rejetée | `w4 = 2,0` |
| rejet | 1 si la requête est rejetée (file pleine, nœud en panne ou absent) | `w5 = 5,0` |

Un rejet reçoit la latence plafonnée, la pénalité d'échéance et celle de rejet : il ne peut jamais être meilleur qu'une
réponse tardive (D6). Les poids sont ajustés par l'étude d'ablation (`make ablation`).

## 5. Transitions

- **Files** : FIFO par nœud, à plusieurs serveurs (`servers`), calcul sans file d'événements (instants de libération des serveurs).
- **Temps de service** : travail de la requête / vitesse du nœud, avec variabilité (`service_cv`) et contention à forte charge
  (`contention`), ou loi mesurée par la calibration (`CalibratedServiceModel`, D10).
- **Réseau** : RTT du profil avec gigue log-normale (`rtt_sigma`) ; l'agent voit une moyenne mobile.
- **Énergie** : modèle linéaire du CDC §5.3, `P = P_idle + (P_max − P_idle)·U` (fogA 2,7/6,4 W, fogB 5/15 W, fogC 10/35 W, …),
  partagé à l'identique par le simulateur et le worker réel (`fogrl/sim/energy.py`). **Batterie virtuelle** (fogA, fogB) : elle
  se décharge selon la puissance consommée et se recharge lentement ; à zéro, le nœud s'éteint (il refuse les requêtes) et ne
  redémarre qu'une fois une fraction minimale rechargée.
- **Pannes** : intervalles d'indisponibilité par nœud (`outages`) ; le nœud rejette puis redevient disponible.
- **Trafic** : Poisson (`low`, `medium`, `high`), MMPP à 2 états (`spikes`), sinusoïde (`diurnal`), rampe ×5 (`flash_crowd`),
  rejeu d'une trace remise à l'échelle (`azure`) ; la charge `load` est exprimée en fraction de la capacité théorique des
  nœuds fog actifs (le cloud est exclu).
- **Randomisation de domaine** (N3) : temps de service ±20 %, RTT ±30 %, débit ±20 %, batterie initiale 60–100 %.

## 6. Curriculum

Les niveaux sont entraînés **indépendamment** (un niveau ne repart pas des poids du précédent) ; chacun valide une étape du
raisonnement avant d'ajouter un terme :

| Niveau | Configuration | Disposition | Trafic | Récompense | Critère |
|---|---|---|---|---|---|
| N0 | 2 nœuds (fogA lent, fogC rapide) | compacte | faible | latence seule | un PPO doit privilégier le nœud rapide (jalon G2) |
| N1 | `n4` | compacte | moyen | latence + échéance + rejet | battre Round-robin et Least-connections |
| N2a | `n4` | compacte | moyen + pics | + énergie | |
| N2 | `n4` | compacte | moyen + pics | + énergie + coût (récompense complète) | |
| N3 | `n3`, `n4`, `n6` | remplie (30 valeurs) | 6 scénarios, randomisation de domaine | complète | modèles livrés (L3) ; la trace `azure` est réservée au test H3 |

## 7. Contrat de modèle

Chaque modèle est un dossier `models/<niveau>/<algo>/seed<k>/` contenant `model.zip`, `vecnormalize.pkl` (statistiques de
récompense) et `contract.json` : version du contrat, ordre des emplacements, dimension, normalisation, environnement complet
et son empreinte, hyperparamètres, commit Git, versions des bibliothèques, durée d'entraînement. Au chargement dans
l'orchestrateur (`--policy rl:<dossier>` ou `POST /policy`), `check_contract` refuse un modèle dont l'ordre des nœuds, la
dimension ou la normalisation diffèrent de la configuration courante.
