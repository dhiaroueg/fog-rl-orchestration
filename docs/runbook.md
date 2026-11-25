# Runbook : de l'installation à la recette

Chaque étape indique la commande, ce qu'elle produit et ce qu'il faut regarder. `make help` liste toutes les cibles.
Les commandes `make` supposent `python` et `pip` du bon environnement virtuel (`PY=venv/bin/python make …` sinon).

## 0. Installation (poste de développement)

```bash
python -m venv venv && source venv/bin/activate          # Windows : venv\Scripts\activate
# torch en version CPU (la roue PyPI par défaut embarque CUDA, ~2,5 Go) :
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
make install                                              # requirements-dev.txt + pip install -e . --no-deps
make test-fast                                            # tests rapides
make test                                                 # + tests lents : entraînement, processus et HTTP réels
```

Python 3.11 ou plus (3.13 utilisé pour le développement et les images Docker).

## 1. Simulation et apprentissage (Phases 2 et 3)

| Étape | Commande | Produit | À vérifier |
|---|---|---|---|
| Baselines | `make sim-baselines` | `results/sim_baselines/results.csv` | classement des 5 baselines par scénario |
| Test N0 (jalon G2) | `make sanity` | verdict affiché | un PPO privilégie le nœud rapide |
| Niveau N1 | `make train-n1` (`SEEDS=0-4`) | `models/n1/<algo>/seed<k>/` | PPO bat Round-robin et Least-connections |
| Hyperparamètres | `make tune` | `configs/train/tuned_ppo.yaml` | à reporter dans `configs/train/n3.yaml` |
| Ablation | `make ablation` | `results/ablation/` (front de Pareto énergie/latence) | poids `w2`, `w3` retenus |
| Modèles livrés (L3) | `make train-n3` | `models/n3/<algo>/seed<k>/` | courbes TensorBoard : `tensorboard --logdir runs` |
| Matrice complète | `make sim-full` | `results/sim_full/results.csv` | 8 politiques × 6 scénarios × 3 tailles × 5 graines |
| Recette en simulation | `make report` | `results/sim_full/recette_n4.md` | ENF-02, ENF-03, H1, H2, OS-4 : *passé / non atteint* par critère |
| Figures du mémoire | `make figures` | `figures/` | uniquement générées par script (CDC §6.3) |

La durée d'un entraînement (`wall_time_s`) est enregistrée dans `contract.json` : ENF-05 demande < 6 h par run sur CPU.
Les graines d'entraînement vont de 0 à 99 et celles d'évaluation de 1 000 à 1 999.

**Si la recette de simulation n'est pas atteinte (jalon G3)** : remonter au niveau N1, vérifier la normalisation et les poids
de récompense, ajouter un seul terme à la fois (CDC §10.3). Greedy est une baseline forte : le gain attendu du RL se situe
surtout sur l'énergie et le coût (H2), pas sur la latence seule.

## 2. Essai du prototype sans Docker

```bash
make local-smoke      # workers + orchestrateur + générateur de charge, 3 politiques × 2 graines, environ 1,5 minute
python scripts/local_stack.py --policy greedy      # pile en marche : orchestrateur sur http://127.0.0.1:<port>
python scripts/local_stack.py --policy rl:models/n3/ppo/seed0     # politique RL servie par l'orchestrateur
```

C'est une vraie pile (processus et HTTP) mais sur **une seule machine** : vitesses et RTT émulés par l'application. Elle valide
la chaîne logicielle, pas les VM, et ses chiffres ne vont pas dans le mémoire.

## 3. Préparer les 3 VM (Phase 2, étape 7)

Une VM Ubuntu Server 24.04 par PC, réseau en pont (*bridge*), IP fixes, câble Ethernet.

```bash
sudo bash scripts/setup_vm.sh   # Docker, plugin Compose, iproute2, module noyau sch_netem (chrony et iperf3 : à installer aussi)
iperf3 -s                       # sur vm1 ; iperf3 -c <ip vm1> sur les autres : débit entre VM
```

`setup_vm.sh` s'arrête en erreur si `sch_netem` est introuvable (noyaux « virtual » minimaux) : installez
`linux-modules-extra-$(uname -r)`, ou utilisez le repli (section 8).

## 4. Générer et lancer le déploiement

```bash
python scripts/gen_compose.py --hosts vm1=192.168.1.11,vm2=192.168.1.12,vm3=192.168.1.13 \
                              --control-vm vm3 --vm-cpus vm1=2,vm2=4,vm3=8 [--configuration n6] [--worker-mode real]
bash scripts/deploy.sh build           # images fogrl-worker et fogrl (à faire sur chaque VM concernée)
bash scripts/deploy.sh vm1 up          # sur vm1 ; idem vm2 sur vm2, vm3 sur vm3
bash scripts/deploy.sh control up      # sur la VM de contrôle : orchestrateur, Prometheus, Grafana
```

- Les fichiers de `deploy/` sont **générés** et ignorés par Git : régénérez-les sur chaque VM avec la même commande.
- `--control-vm` choisit la VM de l'orchestrateur (par défaut vm3, qui héberge aussi cloud).
- `--vm-cpus` déclenche un avertissement si les limites des profils dépassent les cœurs de la VM : en `n4`, vm3 héberge fogC (2 CPU) et
  fogA + fogB demandent 1,5 CPU à vm1, fogC 2 CPU à vm2 et cloud 4 CPU à vm3. Prévoyez aussi de la marge sur vm3 pour l'orchestrateur,
  Prometheus et Grafana. Un nœud sans ses ressources fausse la mesure ; réduisez `cpus`/`servers` dans `configs/nodes/pool6.yaml`
  (puis recalibrez) plutôt que d'ignorer l'avertissement.
- `--worker-mode real` : vrais calculs, inférence ONNX pour la requête lourde. Créez d'abord le modèle (`scripts/export_mobilenet_onnx.py`,
  non exécuté dans l'environnement de développement) dans `models_onnx/`.
- Politique de départ : `--policy least_connections` (ou `rl:models/n3/ppo/seed0` avec le dossier `models/` présent sur la VM de contrôle).
- Grafana : `http://<vm de contrôle>:3000` (tableau de bord « Orchestration fog par RL » préchargé) ; Prometheus : `:9090` ; orchestrateur : `:9000`.

Vérification rapide : `curl http://<vm de contrôle>:9000/state` doit lister les 4 nœuds disponibles ; `curl :9000/nodes?fresh=true` les interroge en direct.

## 5. Calibration (OS-2, semaine 4 puis recalibration en semaine 15)

Workers démarrés, **sans orchestrateur ni autre charge**, sur le réseau de la VM de contrôle :

```bash
# éditer configs/calibration/plan.yaml : URLs des workers (vm1:8001, …)
make calibrate                              # data/calibration/<date> : concurrence 1…servers+1 × type × nœud
make calibration-fit DATA=data/calibration/<date>
```

Lire `configs/calibration/report.md` (hypothèses du profil face aux mesures) puis :
1. reporter les valeurs de `suggested_nodes.yaml` (`speed`, `contention`, `service_cv`, `rtt_ms`, `rtt_sigma`) dans `configs/nodes/pool6.yaml` ;
2. pour utiliser la loi mesurée, ajouter dans `configs/env/*.yaml` : `service_model: calibrated` et `calibration_file: configs/calibration/calibration.json` ;
3. **réentraîner** (`make train-n3`) si la vitesse ou le RTT d'un nœud a changé sensiblement : un modèle est lié à ses constantes (contrat).

Critère OS-2 : temps de service simulés à ±15 % des mesures (jalon G2).

## 6. Campagne réelle et jalon G4 (Phase 5)

Conditions du CDC §10.3 : PC dédiés pendant les runs, câble Ethernet, **3 répétitions** (graines 1 000 à 1 002), horloge unique (chrony).

```bash
make campaign        # configs/experiments/real_campaign.yaml : 8 politiques × 4 scénarios × 3 graines × 60 s
                     #   → results/real_campaign/results.csv (mêmes colonnes que la simulation, + source: real)
make compare         # jalon G4 : écart de latence moyenne < 20 % et même classement des politiques
```

Depuis un autre poste que la VM de contrôle, ajoutez `--url http://<vm de contrôle>:9000` à la commande
`python -m fogrl.loadgen.campaign` (ou renseignez `orchestrator_url` dans la matrice). Variante dans Docker, sur la VM de contrôle :
`docker compose -f deploy/compose.control.yml --profile tools run --rm loadgen --matrix configs/experiments/real_campaign.yaml --out results/real_campaign`.

Les modèles (`models/n3/<algo>/seed0`) doivent être présents sur la VM de contrôle (volume monté en lecture seule). Le journal
des décisions est dans `logs/` (`DECISION_LOG`, JSONL lisible par pandas : `pd.read_json(path, lines=True)`).

## 7. Tests d'exigences sur le prototype (semaine 15)

| Exigence | Manipulation | Observation attendue |
|---|---|---|
| EF-03 hot-swap | `curl -X POST :9000/policy -H 'content-type: application/json' -d '{"spec":"greedy"}'` pendant une campagne | aucune requête perdue ; `fog_policy_info` change |
| EF-07 repli | `curl -X POST :9000/admin/policy-fault -H 'content-type: application/json' -d '{"mode":"error"}'` (agent en panne ; `{"mode":"slow","delay_ms":80}` : agent lent ; `{"mode":"none"}` : retour à la normale) | les requêtes continuent d'être servies par Least-connections ; `fog_decision_fallback_total` croît, `fallback` et `fallback_reason` dans le journal ; au bout de 20 échecs consécutifs la politique est suspendue 5 s |
| EF-07 modèle introuvable | `POST /policy` avec `rl:` vers un dossier absent | 400, l'ancienne politique reste en service (le modèle chargé vit en mémoire : supprimer ses fichiers en cours de run n'a pas d'effet, d'où `policy-fault`) |
| EF-09 nœud en panne | `docker stop fogc` ou `curl -X POST :9000/admin/nodes/fogC/fail` | le nœud passe indisponible en ≤ 2 sondages et ne reçoit plus de requêtes |
| ENF-01 | `decision_ms` du journal ou histogramme `fog_decision_seconds` | P99 < 50 ms |
| ENF-04 | requêtes sans réponse / total sur la campagne | ≥ 99 % |
| ENF-06 | `--configuration n6` | 6 conteneurs, mêmes politiques (un modèle N3 sert 3, 4 et 6 nœuds) |

## 8. Dépannage et pièges connus

- **`configs/env/` doit être versionné.** Le `.gitignore` Python standard contient la règle `env/`, qui ignorait ce dossier ; la règle
  `!configs/env/` à la fin de `.gitignore` le ré-inclut. Ne la supprimez pas, sinon un clone n'a plus les environnements.
- **Fins de ligne.** Les scripts `*.sh`, les `Dockerfile*`, `*.yml`, `*.yaml`, `*.json` et le `Makefile` sont forcés en LF par
  `.gitattributes` : un `entrypoint-worker.sh` en CRLF échoue dans le conteneur (« bad interpreter »). Sous Windows, lancez
  `make` et les scripts depuis WSL ou Git Bash.
- **`tc netem` indisponible** (pas de `sch_netem`, ou conteneur sans `NET_ADMIN`) : le worker démarre quand même et le signale
  (« NON injectee »). Repli : `python scripts/gen_compose.py --emulate-speed` (vitesse émulée, plus de limites CPU) et
  `SIMULATED_RTT_MS` ; la fidélité baisse, à mentionner dans le mémoire.
- **Mesures instables** : fermer les autres applications, Ethernet plutôt que Wi-Fi, répéter 3 fois, ne pas lancer la calibration
  et une campagne en même temps.
- **Trace Azure** : `bash scripts/download_azure_trace.sh [jour]` télécharge environ 137 Mo ; l'URL du jeu public n'a pas pu être
  vérifiée depuis l'environnement de développement (surchargeable par `AZURE_TRACE_URL`) ; à défaut, téléchargez l'archive depuis
  la page du dépôt Azure/AzurePublicDataset et suivez le message d'erreur. Pointez `trace_file` du scénario `azure` sur le CSV.
- **Modèle refusé au chargement** (`Contrat incompatible sur 'node_ids'` ou `'normalization'`) : le modèle a été entraîné avec un
  autre ordre de nœuds ou d'autres constantes ; réentraînez ou restaurez la configuration d'origine.
