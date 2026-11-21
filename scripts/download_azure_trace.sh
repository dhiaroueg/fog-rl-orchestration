#!/usr/bin/env bash
# Telecharge la trace Azure Functions 2019 (jour 1) au format attendu par `type: trace`
# (colonnes HashOwner, HashApp, HashFunction, Trigger, 1..1440 = invocations par minute).
#   bash scripts/download_azure_trace.sh [jour 01-14]
# Resultat : data/traces/azure_functions_2019_d<jour>.csv ; pointez `trace_file` du scenario `azure` dessus.
# L'URL est celle du jeu de donnees public (page GitHub Azure/AzurePublicDataset, AzureFunctionsDataset2019.md) ;
# surchargez-la avec AZURE_TRACE_URL si elle a change. Le jeu fait ~137 Mo compresse.
set -euo pipefail
cd "$(dirname "$0")/.."

day="${1:-01}"
url="${AZURE_TRACE_URL:-https://azurecloudpublicdataset2.z19.web.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz}"
member="invocations_per_function_md.anon.d${day}.csv"
target="data/traces/azure_functions_2019_d${day}.csv"

mkdir -p data/traces
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "Telechargement : $url"
if ! curl -fL --retry 3 -o "$work/azure.tar.xz" "$url"; then
    echo "Echec du telechargement. Recuperez l'archive depuis la page GitHub Azure/AzurePublicDataset" >&2
    echo "puis : tar -xf azurefunctions-dataset2019.tar.xz $member && mv $member $target" >&2
    exit 1
fi
tar -xf "$work/azure.tar.xz" -C "$work" "$member"
mv "$work/$member" "$target"
echo "Trace ecrite : $target ($(wc -l < "$target") lignes)"
echo "Utilisation : dans configs/scenarios.yaml, scenario azure -> trace_file: $target"
