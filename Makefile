# fog-rl-orchestration : commandes courantes (make help).
# Les chemins sont relatifs a la racine du depot. PY=... pour choisir l'interpreteur (ex. PY=venv/bin/python).
PY ?= python
HOSTS ?= vm1=vm1,vm2=vm2,vm3=vm3
SEEDS ?= 0-4

.DEFAULT_GOAL := help
.PHONY: help install test test-fast sanity train-n1 train-n3 tune ablation sim-baselines sim-full report figures \
        compose images local-smoke calibrate calibration-fit campaign compare clean

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Installe les dependances (torch en version CPU recommande, voir README)
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

test: ## Tous les tests (inclut les tests lents : entrainement, processus reels)
	$(PY) -m pytest

test-fast: ## Tests rapides uniquement
	$(PY) -m pytest -m "not slow"

sanity: ## Jalon G2 : un PPO doit privilegier le noeud rapide (N0)
	$(PY) -m fogrl.training.sanity

train-n1: ## Entraine DQN, PPO et A2C au niveau N1 (controle : battre Round-robin / Least-connections)
	$(PY) -m fogrl.training.train --config configs/train/n1.yaml --algo all --seeds $(SEEDS) --n-jobs 2

train-n3: ## Entraine les modeles livres (niveau N3, 3/4/6 noeuds, randomisation de domaine)
	$(PY) -m fogrl.training.train --config configs/train/n3.yaml --algo all --seeds $(SEEDS) --n-jobs 2

tune: ## Optuna : hyperparametres de PPO (30 essais courts)
	$(PY) -m fogrl.training.tune --config configs/train/n3.yaml --trials 30 --out configs/train/tuned_ppo.yaml

ablation: ## Ablation des poids de la recompense et front de Pareto energie / latence
	$(PY) -m fogrl.training.ablation --config configs/experiments/ablation.yaml --out results/ablation

sim-baselines: ## Les 5 baselines sur tous les scenarios et tailles (simulation)
	$(PY) -m fogrl.evaluation.experiments --matrix configs/experiments/sim_baselines.yaml --out results/sim_baselines

sim-full: ## Matrice complete 8 politiques x 6 scenarios x 3 tailles x 5 graines (apres train-n3)
	$(PY) -m fogrl.evaluation.experiments --matrix configs/experiments/sim_full.yaml --out results/sim_full --n-jobs 4

report: ## Criteres de recette (ENF-02, ENF-03, H1, H2, OS-4) depuis results/sim_full
	$(PY) -m fogrl.evaluation.report --results results/sim_full/results.csv --size 4 --out results/sim_full/recette_n4.md

figures: ## Figures du memoire depuis les CSV
	$(PY) -m fogrl.evaluation.figures --results results/sim_full/results.csv --convergence models/n3 --out figures

compose: ## Genere deploy/ (Compose, Prometheus, Grafana) ; HOSTS=vm1=IP1,vm2=IP2,vm3=IP3
	$(PY) scripts/gen_compose.py --hosts $(HOSTS)

images: ## Construit les images Docker fogrl-worker et fogrl
	bash scripts/deploy.sh build

local-smoke: ## Essai du prototype complet sans Docker (workers + orchestrateur + generateur de charge)
	$(PY) scripts/local_stack.py --campaign configs/experiments/local_smoke.yaml --out results/local_smoke

calibrate: ## Mesure les VM (workers en marche, sans orchestrateur) : data/calibration/<date>
	$(PY) -m fogrl.calibration.measure --plan configs/calibration/plan.yaml --out data/calibration/$$(date +%Y-%m-%d)

calibration-fit: ## Reduit les mesures : DATA=data/calibration/<date> make calibration-fit
	$(PY) -m fogrl.calibration.fit --data $(DATA) --out configs/calibration

campaign: ## Campagne sur le prototype deploye (orchestrateur sur localhost:9000)
	$(PY) -m fogrl.loadgen.campaign --matrix configs/experiments/real_campaign.yaml --out results/real_campaign

compare: ## Jalon G4 : simulation contre reel (< 20 % d'ecart, meme classement)
	$(PY) -m fogrl.evaluation.compare --sim results/sim_full/results.csv --real results/real_campaign/results.csv

clean: ## Supprime caches Python et fichiers de test
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
