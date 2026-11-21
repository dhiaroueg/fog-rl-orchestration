# Décisions de conception

Ce document consigne les choix qui complètent ou corrigent le cahier des charges v2.0, avec leur raison et l'endroit du code
où ils s'appliquent. Les décisions D1 à D5 sont celles qui devaient être tranchées avant d'entraîner le moindre modèle ;
les suivantes (D6 à D12) ont été prises pendant la réalisation.

Légende : **CDC** = cahier des charges v2.0 (6 octobre 2026).

---

## D1 — Deux dispositions de l'observation : compacte et remplie

**Problème.** Le CDC annonce 22 valeurs pour 4 nœuds (§5.2) et exige un fonctionnement à 3, 4 et 6 nœuds (ENF-06). Un modèle
Stable-Baselines3 entraîné avec N nœuds attend exactement `4·N + 6` valeurs : sans précaution, il faudrait un modèle par taille.

**Décision.**
- *Disposition compacte* : les N nœuds actifs dans l'ordre de la configuration (4 nœuds → 22 valeurs, comme dans le CDC).
  Utilisée par les niveaux N0 à N2 du curriculum, où la taille est fixe.
- *Disposition remplie* (« padded ») : 6 emplacements fixes (`fogA … fogE, cloud` → 30 valeurs). Un nœud absent est encodé par
  `absent_node` (file saturée, CPU à 1, RTT maximal, batterie vide) et son action est masquée. Utilisée par le niveau N3 : un
  seul modèle sert les configurations `n3`, `n4` et `n6`.

**Conséquences.** L'ordre des emplacements est un contrat : il est sauvegardé dans `contract.json` à côté de chaque modèle et
vérifié au chargement et au `POST /policy` (`check_contract`). Ne réordonnez pas les nœuds de `configs/nodes/pool6.yaml` après
un entraînement.

**Code.** `fogrl/state_spec.py` (`pad_nodes`, `absent_node`, `contract_metadata`, `check_contract`), `configs/env/*.yaml` (`layout`).

## D2 — Interface commune `select(nœuds, requête, contexte)` sur l'état brut

**Problème.** Le CDC demande `select(état) → action` pour toutes les politiques (EF-08), mais les baselines n'ont pas toutes
besoin de la même information : Least-connections compte les requêtes en cours (`in_flight`), qui n'est pas dans le vecteur
d'observation de l'agent ; Greedy estime une latence en unités physiques.

**Décision.** Toutes les politiques implémentent `select(nodes, request, ctx)` avec l'état **brut** (`NodeState`,
`RequestState`, `DecisionContext`). La politique RL encode elle-même l'état avec `encode_state` (le même code qu'à
l'entraînement), applique le masque et renvoie l'indice du nœud. Les huit politiques (Random, Round-robin, Least-connections,
Seuils statiques, Greedy, et DQN/PPO/A2C) tournent donc à l'identique dans le simulateur et dans l'orchestrateur.

**Code.** `fogrl/policies/base.py`, `baselines.py`, `rl.py`, `factory.py` (`make_policy`, spécification `rl:models/n3/ppo/seed0`).

## D3 — `VecNormalize` ne normalise que la récompense

**Problème.** Le CDC prévoit « normalisation des observations et des récompenses » (§10.3). Or les observations sont **déjà**
ramenées dans [0, 1] par des constantes fixes (`Normalization`), identiques en simulation et en réel. Y ajouter une
normalisation glissante ferait dépendre le modèle de statistiques apprises, à transporter (`vecnormalize.pkl`) et à garder
synchrones avec l'orchestrateur, sans bénéfice.

**Décision.** `normalize: {obs: false, reward: true}` dans toutes les configurations d'entraînement. La récompense (somme de
termes d'échelles très différentes) est normalisée par `VecNormalize`, pas les observations. Le fichier `vecnormalize.pkl` (statistiques de
récompense) est tout de même sauvegardé avec le modèle ; l'inférence ne s'en sert que pour un modèle dont les observations
auraient été normalisées (`obs_normalized` dans `contract.json`).

**À savoir.** Les constantes de `Normalization` (file 1 000 ms, RTT 200 ms, travail 300 ms, échéance 2 000 ms, débit 200 req/s)
sont des valeurs de départ ; si la calibration des VM les remet en cause, il faut réentraîner (le contrat les contrôle).

**Code.** `configs/train/*.yaml`, `fogrl/training/train.py` (`make_vec_env`).

## D4 — Pas de masque d'actions à l'entraînement, masque à l'inférence

**Problème.** DQN, PPO et A2C de Stable-Baselines3 ne gèrent nativement aucun masque d'actions (il faudrait `sb3-contrib`, et
seulement pour PPO). Masquer pour un algorithme et pas pour les autres fausserait la comparaison OS-3.

**Décision.** Aucun masque à l'entraînement (le simulateur applique les pannes ; un nœud absent ou en panne est simplement une
mauvaise action, pénalisée). À l'inférence, la politique RL met les logits (PPO/A2C) ou les valeurs Q (DQN) des nœuds
indisponibles à `-inf` avant l'argmax : un nœud en panne ou absent n'est jamais choisi (EF-09), sans réentraîner.

**Code.** `fogrl/policies/rl.py`.

## D5 — Observation volontairement périmée ; `in_flight` toujours frais

**Problème.** L'orchestrateur ne connaît l'état des nœuds que par le sondage de `/state` (toutes les 200 ms dans le CDC). Un
simulateur qui donnerait l'état exact à l'agent produirait des politiques inapplicables au réel.

**Décision.** Le simulateur rafraîchit ses mesures toutes les `poll_interval_ms` (200 ms) : l'agent voit des valeurs périmées
d'au plus 200 ms, alors que les conséquences de sa décision sont calculées sur l'état réel. Une exception, voulue :
`in_flight` (requêtes envoyées et non terminées) est **compté par l'orchestrateur lui-même**, donc toujours à jour, dans le
simulateur comme dans le prototype.

**Code.** `fogrl/sim/env.py` (`_advance_world`, `_observe`), `fogrl/orchestrator/monitor.py`.

---

## D6 — Plafond de latence dans la récompense

La récompense est `r = −(w1·T/Tref + w2·E/Eref + w3·C/Cref + w4·[T > échéance] + w5·[rejet])` (CDC §5.2, poids 1 / 0,5 / 0,3 / 2 / 5).
La latence est **plafonnée** à `latency_cap_factor × échéance`, et un rejet reçoit la latence plafonnée plus les pénalités
d'échéance et de rejet. Sans plafond, une réponse très tardive coûterait plus cher qu'un rejet et l'agent apprendrait à
rejeter. Avec lui, rejeter n'est jamais plus avantageux que répondre en retard. (`fogrl/reward.py`, testé dans `tests/test_reward.py`.)

## D7 — RL-Zoo3 n'est pas utilisé ; Optuna l'est directement

Le CDC cite « Stable-Baselines3, RL-Zoo3 + Optuna » (§5.4). RL-Zoo3 suppose des environnements enregistrés par nom et ignore le
curriculum, la disposition remplie et le fichier de contrat du projet. Le dépôt utilise SB3 directement, avec des
configurations YAML versionnées (`configs/train/`) et Optuna dans `fogrl/training/tune.py`. La reproductibilité (ENF-07) repose
sur le YAML, la graine et le commit enregistrés dans `contract.json`.

## D8 — Un « serveur » de nœud est un processus

Un nœud multi-serveurs (fogC : 2, cloud : 8) doit pouvoir traiter plusieurs requêtes en parallèle sans que la charge d'une
requête bloque les autres. Le worker FastAPI délègue donc le calcul à un `ProcessPoolExecutor` dont la taille est le nombre de
serveurs du profil : un serveur = un processus, sans partage du GIL. L'exécuteur « thread » n'existe que pour les tests en mémoire.

**Code.** `fogrl/worker/runtime.py` (machine à états `NodeRuntime`), `fogrl/worker/app.py`.

## D9 — Hétérogénéité : cgroups et `tc netem` sur les VM, émulation applicative en local

Sur les VM, la puissance d'un nœud vient des limites Docker (`cpus`, `mem_limit`) et la latence réseau de `tc netem` sur la
sortie du conteneur (`NETEM_DELAY_MS` = `rtt_ms` du profil, `entrypoint-worker.sh`). Sur un seul poste (essai sans Docker, CI),
`EMULATE_SPEED=1` ralentit le service d'un nœud lent et `SIMULATED_RTT_MS` retarde les réponses. Les deux modes ne sont jamais
cumulés (les fichiers Compose génèrent `EMULATE_SPEED=0` et `SIMULATED_RTT_MS=0` pour les VM).

**Portée.** L'émulation applicative utilise le même modèle que le simulateur : elle valide la chaîne logicielle, **pas** les
VM. Le jalon G4 se mesure sur les VM.

## D10 — Calibration en boucle fermée

La calibration (`fogrl.calibration`) mesure, pour chaque nœud, type de requête et niveau de concurrence, le temps de service
et le RTT, avec la charge définie comme la **fraction de serveurs occupés à l'arrivée** (la même définition que le simulateur).
`fit.py` en déduit `calibration.json` (loi de service par nœud et type de requête), `suggested_nodes.yaml` (`speed`,
`contention`, `service_cv`, `rtt_ms`, `rtt_sigma` mesurés, à reporter dans `pool6.yaml`) et `report.md` (hypothèses du profil
face aux mesures, avec les écarts à vérifier pour OS-2). Le simulateur consomme `calibration.json` via
`CalibratedServiceModel.from_json`, qui utilise le `nominal_work_ms` du fichier lui-même.

## D11 — Tolérance aux pannes de l'orchestrateur

- **Budget de décision** (EF-07) : la politique s'exécute dans un thread sous `asyncio.wait_for` ; au-delà de 20 ms ou en cas
  d'exception, la requête est routée par Least-connections et le repli est compté et journalisé.
- **Disjoncteur de politique** : après 20 échecs consécutifs, la politique est suspendue 5 s (repli permanent), puis réessayée.
- **Disjoncteur de nœud** (EF-09) : 2 échecs consécutifs de sondage ou de requête déclarent le nœud indisponible ; il est
  réintégré dès qu'un sondage réussit.
- **Panne d'agent injectée** (test EF-07) : le modèle vit en mémoire, supprimer ses fichiers en cours de run ne change donc rien ;
  `POST /admin/policy-fault` (`error`, `slow`, `none`) simule la défaillance de l'agent pour rendre le repli observable.
- **Hot-swap** (EF-03) : `POST /policy` charge la nouvelle politique, contrôle son contrat, la fait chauffer
  (5 décisions à blanc) puis la substitue atomiquement ; en cas d'échec, l'ancienne politique reste en service.

**Code.** `fogrl/orchestrator/{router,decision,monitor}.py`, réglages dans `configs/deploy/orchestrator.yaml`.

## D12 — Reproductibilité

Graines d'entraînement 0–99, graines d'évaluation ≥ 1000 (vérifiées par `check_train_seed` / `check_eval_seeds`). Le générateur
de charge rejoue `fogrl.traffic` : même graine = même suite de requêtes en simulation et en réel (EF-06). Chaque modèle
sauvegarde `contract.json` (commit Git, versions des bibliothèques, durée d'entraînement) ; chaque CSV de résultats a une
ligne par (politique, scénario, taille, graine) avec les mêmes colonnes en simulation et en réel (+ `source`).
