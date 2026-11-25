#!/usr/bin/env bash
# Prepare une VM Ubuntu 22.04 / 24.04 : Docker Engine, plugin Compose et module noyau sch_netem
# (necessaire a `tc netem`, qui injecte la latence reseau des noeuds).
#   sudo bash scripts/setup_vm.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Lancez ce script avec sudo : sudo $0" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl gnupg iproute2 git make python3-venv
# sch_netem est dans linux-modules-extra sur Ubuntu ; absent des noyaux "virtual" minimaux.
apt-get install -y "linux-modules-extra-$(uname -r)" || echo "linux-modules-extra indisponible pour ce noyau (verifiez sch_netem ci-dessous)"

if ! command -v docker >/dev/null 2>&1; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")"
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${codename} stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker

if modprobe sch_netem; then
    echo sch_netem > /etc/modules-load.d/netem.conf
    echo "sch_netem charge"
else
    echo "ERREUR : module sch_netem indisponible : la latence reseau ne pourra pas etre injectee." >&2
    echo "Solution de repli : EMULATE_SPEED=1 et SIMULATED_RTT_MS dans les workers (voir docs/runbook.md)." >&2
    exit 1
fi

if [ -n "${SUDO_USER:-}" ]; then
    usermod -aG docker "$SUDO_USER"
    echo "Reconnectez-vous pour que le groupe docker soit pris en compte pour $SUDO_USER."
fi
docker --version
docker compose version
