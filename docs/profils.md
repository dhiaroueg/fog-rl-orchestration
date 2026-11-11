# Documentation : Fichier de Configuration des Profils d'Infrastructure

Ce fichier centralise les paramètres d'émulation du système d'équilibrage de charge. Il permet de s'assurer que le simulateur et l'environnement réel partagent exactement les mêmes métriques.

## 1. Allocation des VM
- **vm1 (PC 1 - Dell i3)** : Héberge les petits nœuds légers (`fogA`, `fogB`).
- **vm2 (PC 2 - Lenovo Ryzen 5)** : Héberge le nœud intermédiaire (`fogC`).
- **vm3 (PC 3 - G5 i7)** : Héberge le nœud le plus puissant (`cloud`) ainsi que l'orchestrateur.

## 2. Typologie des charges
Le système traite trois types de requêtes (`light`, `medium`, `heavy`) caractérisées par leur temps de traitement requis (`work_ms`) et leur contrainte de temps limite (`deadline_ms`).

## 3. Paramètres physiques et réseaux des Nœuds
Chaque nœud émule des conditions matérielles spécifiques :
- **Limitation CPU et mémoire** via Docker (`cpus`, `mem_mb`).
- **Simulation de latence réseau** (`netem_delay_ms`) pour reproduire l'éloignement géographique des serveurs Fog et Cloud.
- **Modélisation énergétique** (`p_idle`, `p_max`, `battery`) pour évaluer l'impact de la consommation électrique des différentes stratégies de routage.