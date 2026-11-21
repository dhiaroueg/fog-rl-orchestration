#!/usr/bin/env bash
# Pilote les fichiers Compose generes (deploy/).
#   bash scripts/deploy.sh build                 construit les images fogrl-worker et fogrl
#   bash scripts/deploy.sh <cible> [up|down|logs|ps]  cible : local | vm1 | vm2 | vm3 | control
# Sur la VM de controle : `bash scripts/deploy.sh vm1 up` puis `bash scripts/deploy.sh control up`.
set -euo pipefail
cd "$(dirname "$0")/.."

target="${1:?usage : bash scripts/deploy.sh <build|local|vm1|vm2|vm3|control> [up|down|logs|ps]}"
action="${2:-up}"

if [ "$target" = "build" ]; then
    docker build -f docker/Dockerfile.worker -t fogrl-worker:latest .
    docker build -f docker/Dockerfile.fogrl -t fogrl:latest .
    exit 0
fi

case "$target" in
    local | vm1 | vm2 | vm3 | control) compose_file="deploy/compose.${target}.yml" ;;
    *) echo "Cible inconnue : $target (local | vm1 | vm2 | vm3 | control)" >&2; exit 2 ;;
esac
if [ ! -f "$compose_file" ]; then
    echo "$compose_file absent : generez-le avec  python scripts/gen_compose.py --hosts vm1=IP1,vm2=IP2,vm3=IP3" >&2
    exit 1
fi
mkdir -p logs results models

case "$action" in
    up) docker compose -f "$compose_file" up -d ;;
    down) docker compose -f "$compose_file" down ;;
    logs) docker compose -f "$compose_file" logs -f --tail=100 ;;
    ps) docker compose -f "$compose_file" ps ;;
    *) echo "Action inconnue : $action (up | down | logs | ps)" >&2; exit 2 ;;
esac
