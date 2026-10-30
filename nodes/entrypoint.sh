#!/bin/sh
set -e

if [ -n "${NETEM_DELAY_MS}" ] && [ "${NETEM_DELAY_MS}" != "0" ]; then
  if tc qdisc replace dev eth0 root netem delay "${NETEM_DELAY_MS}ms" "${NETEM_JITTER_MS:-0}ms"; then
    echo "[entrypoint] netem : delai ${NETEM_DELAY_MS}ms +/- ${NETEM_JITTER_MS:-0}ms"
  else
    echo "[entrypoint] ATTENTION : netem indisponible, aucun delai applique"
  fi
fi
exec python -m uvicorn worker:app --host 0.0.0.0 --port 8000