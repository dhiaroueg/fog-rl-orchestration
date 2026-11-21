#!/bin/sh
# Injecte la latence reseau du noeud (tc netem) avant de lancer le worker.
#   NETEM_DELAY_MS   delai ajoute a la SORTIE du conteneur ; l'aller-retour vu par l'orchestrateur vaut ce delai
#   NETEM_JITTER_MS  gigue (loi normale) ; NETEM_IFACE interface (defaut eth0)
# Necessite la capacite NET_ADMIN (cap_add dans Compose). Sans elle, le worker demarre quand meme
# et signale que la latence n'est pas injectee.
set -eu

DELAY="${NETEM_DELAY_MS:-0}"
if [ "$DELAY" != "0" ] && [ "$DELAY" != "0.0" ] && [ -n "$DELAY" ]; then
    IFACE="${NETEM_IFACE:-eth0}"
    JITTER="${NETEM_JITTER_MS:-0}"
    if tc qdisc replace dev "$IFACE" root netem delay "${DELAY}ms" "${JITTER}ms" distribution normal 2>/dev/null \
        || tc qdisc replace dev "$IFACE" root netem delay "${DELAY}ms" "${JITTER}ms" 2>/dev/null; then
        echo "netem actif sur $IFACE : delai ${DELAY} ms, gigue ${JITTER} ms"
    else
        echo "ATTENTION : tc netem impossible (capacite NET_ADMIN ou module sch_netem manquant) : latence NON injectee" >&2
    fi
fi

exec "$@"
