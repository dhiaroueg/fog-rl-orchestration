# Générateur de trafic

Le module `generator` produit le même schéma de requête pour le simulateur et
le lanceur réel. Les scénarios sont sélectionnés avec `TrafficConfig.scenario`.
Tous les temps sont en millisecondes, `rho` est la charge de référence et la
seed détermine séparément les arrivées, les types, le travail et les zones.

## Scénarios

| Scénario | Modèle | Valeurs de départ |
| --- | --- | --- |
| `constant` | Processus de Poisson homogène | Débit moyen donné par `rho` et `capacity` |
| `peaks` | MMPP à deux états, durées d'état exponentielles | bas : 30 s × 0,5 ; haut : 10 s × 2,5 |
| `diurnal` | Intensité sinusoïdale jour/nuit | période : 24 h simulées ; amplitude : 0,8 ; maximum à mi-période |
| `flash_crowd` | Niveau de base, rampe linéaire, plateau, récupération linéaire | pic ×5 ; début à 20 % de l'horizon |
| `trace` | Comptages observés transformés en profil d'intensité | CSV configurable, intervalle par défaut : 60 s |

Des profils prêts à l'emploi sont fournis dans `experiments/configs/traffic_*.yaml`.
Le profil `traffic_trace.yaml` est un gabarit : le CSV Azure doit être obtenu,
filtré et vérifié par l'équipe avant son exécution.

Les multiplicateurs du MMPP sont normalisés avec les probabilités stationnaires
du processus de Markov : `rho` garde ainsi le sens de débit moyen à long terme.
Le profil diurne a une moyenne de 1 sur une période complète. Le flash crowd
produit une hausse puis un retour progressif au niveau de base ; il n'est pas
renormalisé en mode durée, car le surcroît de demandes fait partie du scénario.

En mode `duration_ms`, le générateur échantillonne un processus de Poisson
non-homogène et le nombre de requêtes varie. En mode `n_requests`, il conserve
le nombre demandé et conditionne la distribution temporelle sur cet effectif.
L'horizon de référence est alors `n_requests / arrival_rate(cfg)`. Cette
convention permet de comparer les scénarios avec un nombre de tâches fixe ;
pour étudier des durées diurnes réalistes, utiliser une période adaptée à la
durée simulée ou générer un épisode avec `duration_ms`.

## Rejeu d'une trace

Le dépôt ne contient pas la trace Azure. Fournir un CSV local avec une ligne par
intervalle et une colonne `invocations` non négative. Une colonne facultative
`timestamp` peut donner les débuts d'intervalle au format date/heure ; sans
horodatage, chaque ligne représente `trace_interval_ms` (60 000 ms par défaut).
Les intervalles doivent être croissants et le total des comptages doit être
positif. Le générateur conserve la forme temporelle de la trace, ramène son axe
de temps à l'horizon du run et ajuste le débit moyen à la charge `rho` choisie.
Les arrivées sont uniformes à l'intérieur de chaque intervalle. Il s'agit donc
d'un rejeu du profil agrégé, pas d'une reconstitution d'horodatages individuels.

Exemple de CSV :

```csv
timestamp,invocations
2019-01-01T00:00:00Z,125
2019-01-01T00:01:00Z,98
2019-01-01T00:02:00Z,173
```

Lancer le trafic réel avec :

```powershell
python -m orchestrator.run --scenario peaks --requests 2000 --seed 1000
python -m orchestrator.run --scenario trace --trace data/traces/azure_counts.csv --requests 2000
```

Les paramètres de forme (durées MMPP, période diurne, fractions du flash,
colonnes de trace) se règlent dans un YAML au format `TrafficConfig`. Les options
CLI renseignées remplacent les valeurs du YAML ; `--requests` reste le choix le
plus simple pour adapter le volume d'un run.

```yaml
scenario: diurnal
rho: 0.6
duration_ms: 600000
diurnal_period_ms: 600000  # une journée compressée en 10 minutes de simulation
diurnal_amplitude: 0.8
diurnal_peak_phase: 0.5
```

```powershell
python -m orchestrator.run --config experiments/configs/traffic_diurnal.yaml --seed 1000
```

Chaque exécution de `orchestrator.run` écrit aussi un CSV de mesures des nœuds
Fog `fogA`, `fogB` et `fogC`, avec le scénario inscrit sur chaque ligne.
Sans `--output`, chaque scénario produit des fichiers distincts, par exemple
`results/orchestration_flash_crowd.jsonl` et
`results/orchestration_flash_crowd_measurements.csv`.
Avec `--output results/workflow.jsonl`, le CSV est
`results/workflow_flash_crowd_measurements.csv`.
Le CSV contient l'état observé, le RTT, la charge CPU, la file, ainsi que le
temps d'attente et le temps de service du nœud choisi. Un nœud injoignable est
conservé dans le fichier avec son erreur et ses mesures vides.

La licence, la version et la transformation de la trace publique doivent être
consignées avec les résultats. Les fichiers de données ne sont pas téléchargés
automatiquement par le générateur.
