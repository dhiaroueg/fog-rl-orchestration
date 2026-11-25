# Avancement par rapport au cahier des charges v2.0

Légende :
- ✅ **Fait et testé** : implémenté, couvert par des tests automatiques (`make test`).
- 🟡 **Prêt, à mesurer** : le code et les outils existent ; le critère se constate seulement avec les modèles finaux entraînés,
  ou sur les vraies VM (rien de ce qui suit n'a pu être exécuté sur les VM de l'équipe).
- ⬜ **À faire par l'équipe** : travail humain (validation, rédaction, installation des VM, mesures).

## Ce qui a été validé, et ce qui ne l'a pas été

| Validé | Comment |
|---|---|
| Simulateur, politiques, entraînement DQN/PPO/A2C, évaluation, statistiques | tests unitaires et d'intégration (`tests/`), y compris `check_env` de Gymnasium et un entraînement court de chaque algorithme |
| Prototype réel (worker, orchestrateur, générateur de charge, calibration) | tests en mémoire (ASGI) **et** pile de vrais processus en HTTP sur un seul poste (`tests/test_local_stack.py`, `make local-smoke`) |
| Chaîne simulation → réel | campagne locale : écart de latence moyenne simulation/réel de 0,4 à 1,6 % et même classement (Greedy < Least-connections < Round-robin) ; PPO servi par l'orchestrateur, décision P99 ≈ 5 ms, sans repli |
| Fichiers de déploiement | structure testée ; fichiers Compose validés contre le schéma officiel Compose-spec ; `bash -n` des scripts ; métriques du tableau de bord comparées aux expositions réelles |

| **Non** validé ici | Pourquoi / quoi faire |
|---|---|
| Construction des images Docker, `docker compose up`, `tc netem` | pas de démon Docker dans l'environnement de développement ; la CI GitHub (`.github/workflows/ci.yml`) exécute `docker compose config`, construit l'image du worker et le fait démarrer |
| Mesures sur VM, calibration, jalon G4 | la pile locale émule la vitesse et le RTT avec le **même modèle** que le simulateur : elle valide la chaîne logicielle, pas la fidélité physique |
| Modèles finaux (N3, 5 graines × 3 algorithmes) et recette ENF-02 / ENF-03 | plusieurs heures de calcul CPU par run ; lancer `make train-n3` puis `make sim-full` et `make report` |
| Inférence ONNX MobileNetV2 | `scripts/export_mobilenet_onnx.py` n'a pas pu être exécuté (torchvision absent) ; repli sur un calcul matriciel si le modèle manque |
| Téléchargement de la trace Azure | URL non vérifiée depuis l'environnement (`AZURE_TRACE_URL` pour la surcharger) ; une trace synthétique de même format est fournie |

## Exigences fonctionnelles

| Réf. | Exigence | Statut | Implémentation | Tests / vérification |
|---|---|---|---|---|
| EF-01 | Router chaque requête vers un nœud | ✅ | `orchestrator/router.py` (`route`) | `test_orchestrator.py::test_route_end_to_end_with_every_baseline`, journal |
| EF-02 | État : CPU, file, RTT, énergie, requête | ✅ | `state_spec.py` ([MDP](mdp.md)) | `test_state_spec.py` |
| EF-03 | Hot-swap sans redémarrage | ✅ | `Router.set_policy`, `POST /policy` | `test_hot_swap_during_traffic_loses_no_request`, `test_hot_swap_rejects_unknown_policies…` |
| EF-04 | Tableau de bord temps réel par nœud | 🟡 | `deploy/dashboard.py`, `generate.py` (Grafana provisionné) | noms de métriques vérifiés (`test_deploy.py`) ; rendu à constater sur Grafana |
| EF-05 | Journal JSONL de chaque décision | ✅ | `orchestrator/decision_log.py` | `test_decision_log_is_valid_jsonl` |
| EF-06 | Scénarios configurables, même graine = même séquence | ✅ | `traffic.py`, `loadgen/generator.py`, `configs/scenarios.yaml` | `test_traffic.py`, `test_loadgen.py::test_same_seed_replays_the_same_request_sequence` |
| EF-07 | Repli si l'agent échoue ou dépasse 20 ms | ✅ | `orchestrator/decision.py`, `POST /admin/policy-fault` | `test_slow_policy…`, `test_failing_policy…`, `test_policy_breaker…`, `test_injected_agent_fault…` |
| EF-08 | Même interface pour les 8 politiques | ✅ | `policies/base.py`, `factory.py` | baselines et PPO dans l'orchestrateur ; DQN, PPO et A2C via la même classe `RLPolicy` dans `test_training.py` (masque, équivalence avec SB3) |
| EF-09 | Détecter un nœud injoignable | ✅ | `orchestrator/monitor.py`, `Router.route` | `test_failed_node_is_avoided_and_recovers`, `test_unreachable_and_slow_nodes…` ; arrêt d'un conteneur à refaire sur VM |

## Exigences non fonctionnelles

| Réf. | Critère | Statut | Outil | Où se mesure |
|---|---|---|---|---|
| ENF-01 | Décision < 50 ms au P99 | 🟡 | `decision_ms` du journal, `fog_decision_seconds` | localement ≈ 5 ms ; à confirmer sur VM avec les modèles N3 |
| ENF-02 | Latence moyenne −20 % vs Round-robin (charge moyenne/élevée) | 🟡 | `make sim-full`, `make report` | après `make train-n3` ; un premier PPO de niveau N1 (une graine) faisait déjà mieux que Greedy sur la latence et les échéances |
| ENF-03 | Énergie fog −15 % vs Seuils statiques à latence égale | 🟡 | idem (`recette_n4.md`) | après `make train-n3` ; le gain attendu du RL est surtout sur l'énergie et le coût (H2) |
| ENF-04 | Disponibilité ≥ 99 % | 🟡 | taux de requêtes sans réponse du CSV de campagne | `make campaign` sur VM |
| ENF-05 | Convergence < 6 h par run | 🟡 | `wall_time_s` dans `contract.json`, TensorBoard | à relever à chaque entraînement |
| ENF-06 | 3, 4 et 6 nœuds | 🟡 | disposition remplie (D1), configurations `n3`/`n4`/`n6` | modèle N3 testé à 3, 4 et 6 nœuds (`test_padded_n3_model_serves_…`) ; 6 conteneurs à lancer sur VM |
| ENF-07 | Reproductibilité (commit + config + graine) | ✅ | `contract.json`, graines séparées, CI | relance par un autre membre à faire |
| ENF-08 | Écart sim/réel < 20 %, même classement | 🟡 | `make compare` (`evaluation/compare.py`) | local : < 2 % ; jalon G4 sur VM |

## Objectifs spécifiques

| Réf. | Objectif | Statut | Détail |
|---|---|---|---|
| OS-1 | MDP formalisé | ✅ / ⬜ | [`mdp.md`](mdp.md), `state_spec.py` ; **validation par l'encadrant à obtenir** |
| OS-2 | Simulateur Gymnasium calibré | ✅ / 🟡 | `check_env` passe ; calibration (`make calibrate`, `calibration-fit`) à exécuter sur les VM, critère ±15 % |
| OS-3 | DQN / PPO / A2C sur 5 graines | 🟡 | `make train-n1`, `make train-n3` (`SEEDS=0-4`) ; Optuna : `make tune` |
| OS-4 | 5 baselines, gain significatif (p < 0,05) sur 2 objectifs | 🟡 | baselines ✅ ; `fogrl/evaluation/stats.py` (Mann-Whitney) et `make report` |
| OS-5 | Prototype sur 3 VM, 4 à 6 nœuds, décision < 50 ms | 🟡 | code et Compose prêts ; [déploiement](runbook.md) à faire |
| OS-6 | Supervision Prometheus/Grafana | 🟡 | générée par `make compose` ; à constater sur VM |
| OS-7 | Écart simulation/réel | 🟡 | `fogrl/loadgen/campaign.py` + `evaluation/compare.py` |

## Hypothèses de recherche

| Hyp. | Énoncé | Outil |
|---|---|---|
| H1 | −20 % de latence vs Round-robin à charge moyenne/élevée | `make report` (recette ENF-02) |
| H2 | −15 % d'énergie ou de coût à latence comparable à Greedy | `make report` |
| H3 | Gains conservés sur une trace jamais vue et sur le cluster réel | scénario `azure` (absent de l'entraînement N3) + `make campaign` |

## Jalons

| Jalon | Semaine | Critère | État du code |
|---|---|---|---|
| G1 | S3 | MDP validé, CDC approuvé, 3 VM opérationnelles | MDP documenté ; validation et VM ⬜ |
| G2 | S6 | Simulateur calibré (≤ 15 %), test N0 réussi | test N0 ✅ (`make sanity`, `test_n0_consistency_check…`) ; calibration 🟡 |
| G3 | S10 | PPO bat Round-robin et Least-connections, égale Greedy en latence et le bat en énergie/coût | outils prêts 🟡 ; repli prévu : revenir à N1 (voir [runbook](runbook.md) §1) |
| G4 | S15 | Écart sim/réel < 20 %, même classement | 🟡 |
| G5 | S18 | Soutenance | ⬜ |

## Livrables

| N° | Livrable | Statut | Où |
|---|---|---|---|
| L1 | Cahier des charges v2.0 et état de l'art | ⬜ | rédaction de l'équipe |
| L2 | Simulateur documenté, CSV de calibration, tests | ✅ / 🟡 | `fogrl/sim`, `docs/`, `tests/` ; CSV de calibration à produire sur VM |
| L3 | Modèles DQN/PPO/A2C (5 graines), statistiques, courbes | 🟡 | `models/` (ignoré par Git, régénérable) |
| L4 | Rapport d'évaluation en simulation | 🟡 | `results/`, `figures/` (générés par script) |
| L5 | Prototype réel : orchestrateur, workers, Compose, scripts | ✅ | `fogrl/`, `docker/`, `scripts/`, `deploy/` (généré) |
| L6 | Tableaux de bord Grafana (JSON) | ✅ | `make compose` → `deploy/grafana/dashboards/fog-rl.json` |
| L7 | Rapport comparatif sim/réel | 🟡 | `make compare` |
| L8 | Mémoire de PFE | ⬜ | rédaction de l'équipe |
| L9 | Soutenance, démo, vidéo de secours | ⬜ | équipe |

## Critères de recette (CDC §10.2)

| Critère | Statut |
|---|---|
| L'agent converge vers une politique stable sur 5 graines | 🟡 `make train-n3` |
| Amélioration significative sur ≥ 2 des 3 objectifs | 🟡 `make report` |
| Latence −20 % vs Round-robin (ENF-02) | 🟡 |
| Énergie fog −15 % vs Seuils statiques (ENF-03) | 🟡 |
| Décision < 50 ms au P99 sur le prototype réel (ENF-01) | 🟡 (≈ 5 ms mesurés en local) |
| Le prototype réel reproduit les tendances de la simulation | 🟡 (local : oui ; VM à faire) |
| Opérationnel avec repli si l'agent ou un nœud tombe | ✅ en test ; à rejouer sur VM ([runbook](runbook.md) §7) |
| Résultats reproductibles depuis le dépôt | ✅ code, configs, graines, CI ; relance par un autre membre ⬜ |

## Ordre conseillé pour la suite

1. `make install && make test-fast` (vérifie l'installation) ; pousser sur GitHub et vérifier que la CI passe.
2. `make train-n1`, puis relire les résultats N1 : si PPO ne bat pas Round-robin / Least-connections, ne pas passer à N3.
3. Mettre en place les VM ([runbook](runbook.md) §3 et 4), lancer la calibration (§5), reporter les valeurs mesurées dans `pool6.yaml`.
4. `make tune`, `make ablation`, `make train-n3`, `make sim-full`, `make report`.
5. Campagne réelle et jalon G4 (§6), tests d'exigences (§7), figures et mémoire.
