"""Genere les fichiers de deploiement a partir de configs/nodes/*.yaml (source unique des profils).

    python scripts/gen_compose.py --hosts vm1=192.168.56.11,vm2=192.168.56.12,vm3=192.168.56.13

Fichiers produits dans ``deploy/`` (ne pas les modifier a la main : relancer le generateur) :
    compose.local.yml          tout sur une machine (developpement, reseau Docker commun)
    compose.vm1|vm2|vm3.yml    les workers de chaque VM, avec limites CPU/memoire et ``tc netem``
    compose.control.yml        orchestrateur, Prometheus, Grafana (a lancer sur la VM de controle)
    prometheus/prometheus.yml, grafana/provisioning/*, grafana/dashboards/fog-rl.json

Heterogeneite : sur les VM, ``cpus`` / ``mem_limit`` (limites cgroup) font la difference de puissance
entre noeuds (``EMULATE_SPEED=0``). Sans limites (machines identiques, CI) : ``--emulate-speed``.
Latence : ``NETEM_DELAY_MS`` = ``rtt_ms`` du profil sur la sortie du worker (un seul sens : l'aller-retour
mesure vaut donc ``rtt_ms``) ; la gigue ``rtt_sigma`` x ``rtt_ms`` est ajoutee en loi normale.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import yaml

from ..config import NodeProfile, load_nodes
from ..orchestrator.settings import parse_urls
from .dashboard import build_dashboard

HEADER = "# GENERE par scripts/gen_compose.py a partir de configs/nodes/*.yaml : ne pas modifier a la main.\n"
NETWORK = "fognet"
PROMETHEUS_IMAGE = "prom/prometheus:v3.5.0"
GRAFANA_IMAGE = "grafana/grafana:12.1.0"


@dataclass(frozen=True)
class DeployOptions:
    nodes_file: str = "configs/nodes/pool6.yaml"
    configuration: str = "n4"
    hosts: Mapping[str, str] = field(default_factory=lambda: {"vm1": "vm1", "vm2": "vm2", "vm3": "vm3"})
    control_vm: str = "vm1"
    worker_image: str = "fogrl-worker:latest"
    fogrl_image: str = "fogrl:latest"
    worker_mode: str = "burn"  # burn | real
    emulate_speed: bool = False
    policy: str = "least_connections"
    env_file: str = "configs/env/eval.yaml"
    orchestrator_port: int = 9000
    prometheus_port: int = 9090
    grafana_port: int = 3000
    vm_cpus: Mapping[str, float] = field(default_factory=dict)  # verification des ressources (optionnel)


def load_profiles(opts: DeployOptions) -> list[NodeProfile]:
    profiles, configurations = load_nodes(opts.nodes_file)
    if opts.configuration not in configurations:
        raise ValueError(f"Configuration inconnue : {opts.configuration!r} (disponibles : {sorted(configurations)})")
    active = set(configurations[opts.configuration])
    chosen = [p for p in profiles if p.node_id in active]
    unknown_vms = sorted({p.vm for p in chosen} - set(opts.hosts))
    if unknown_vms:
        raise ValueError(f"VM sans adresse : {unknown_vms} (option --hosts, ex. vm1=192.168.56.11)")
    return chosen


def host_name(profile: NodeProfile) -> str:
    return profile.node_id.lower()


def worker_url(profile: NodeProfile, opts: DeployOptions, local: bool) -> str:
    host = host_name(profile) if local else opts.hosts[profile.vm]
    return f"http://{host}:{profile.port}"


def worker_service(profile: NodeProfile, opts: DeployOptions) -> dict:
    jitter = round(profile.rtt_ms * profile.rtt_sigma, 2)
    environment = {
        "NODE_ID": profile.node_id,
        "NODES_FILE": "configs/nodes/pool6.yaml" if opts.nodes_file.endswith("pool6.yaml") else opts.nodes_file,
        "WORKER_MODE": opts.worker_mode,
        "WORKER_EXECUTOR": "process",
        "EMULATE_SPEED": "1" if opts.emulate_speed else "0",
        "PORT": str(profile.port),
        "NETEM_DELAY_MS": str(profile.rtt_ms),
        "NETEM_JITTER_MS": str(jitter),
        "SIMULATED_RTT_MS": "0",
    }
    service: dict = {
        "image": opts.worker_image,
        "container_name": f"fog-{host_name(profile)}",
        "hostname": host_name(profile),
        "restart": "unless-stopped",
        "cap_add": ["NET_ADMIN"],  # tc netem
        "environment": environment,
        "ports": [f"{profile.port}:{profile.port}"],
        "networks": [NETWORK],
        "healthcheck": {
            "test": ["CMD", "python", "-c",
                     f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{profile.port}/health', timeout=2)"],
            "interval": "10s",
            "timeout": "3s",
            "retries": 3,
        },
    }
    if not opts.emulate_speed:
        service["cpus"] = profile.cpus
        service["mem_limit"] = f"{profile.mem_mb}m"
    if opts.worker_mode == "real":
        environment["ONNX_MODEL"] = "/models/mobilenet_v2.onnx"
        service["volumes"] = ["../models_onnx:/models:ro"]
    return service


def orchestrator_services(opts: DeployOptions, urls: Optional[Mapping[str, str]], local: bool = False) -> dict:
    environment = {
        "ORCH_CONFIG": "configs/deploy/orchestrator.yaml",
        "ORCH_CONFIGURATION": opts.configuration,
        "ORCH_POLICY": opts.policy,
        "DECISION_LOG": "logs/decisions.jsonl",
        "PORT": str(opts.orchestrator_port),
    }
    if urls:
        environment["ORCH_NODE_URLS"] = ",".join(f"{node}={url}" for node, url in urls.items())
    common_volumes = ["../models:/app/models:ro", "../logs:/app/logs", "../results:/app/results"]
    return {
        "orchestrator": {
            "image": opts.fogrl_image,
            "container_name": "fog-orchestrator",
            "restart": "unless-stopped",
            "command": ["python", "-m", "fogrl.orchestrator.app"],
            "environment": environment,
            "ports": [f"{opts.orchestrator_port}:{opts.orchestrator_port}"],
            "volumes": common_volumes,
            "networks": [NETWORK],
        },
        "loadgen": {  # docker compose run --rm loadgen --matrix configs/experiments/real_campaign.yaml --out results/real
            "image": opts.fogrl_image,
            "profiles": ["tools"],
            "entrypoint": ["python", "-m", "fogrl.loadgen.campaign", "--url", f"http://orchestrator:{opts.orchestrator_port}"],
            "volumes": common_volumes,
            "networks": [NETWORK],
            "depends_on": ["orchestrator"],
        },
        "prometheus": {
            "image": PROMETHEUS_IMAGE,
            "container_name": "fog-prometheus",
            "restart": "unless-stopped",
            "command": ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.retention.time=30d"],
            "volumes": [
                f"./prometheus/{'prometheus.local.yml' if local else 'prometheus.yml'}:/etc/prometheus/prometheus.yml:ro",
                "prometheus-data:/prometheus",
            ],
            "ports": [f"{opts.prometheus_port}:9090"],
            "networks": [NETWORK],
        },
        "grafana": {
            "image": GRAFANA_IMAGE,
            "container_name": "fog-grafana",
            "restart": "unless-stopped",
            "environment": {"GF_AUTH_ANONYMOUS_ENABLED": "true", "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
                            "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH": "/var/lib/grafana/dashboards/fog-rl.json"},
            "volumes": [
                "./grafana/provisioning:/etc/grafana/provisioning:ro",
                "./grafana/dashboards:/var/lib/grafana/dashboards:ro",
                "grafana-data:/var/lib/grafana",
            ],
            "ports": [f"{opts.grafana_port}:3000"],
            "networks": [NETWORK],
            "depends_on": ["prometheus"],
        },
    }


def _compose(services: dict, volumes: Sequence[str] = ()) -> dict:
    document: dict = {"name": "fogrl", "services": services, "networks": {NETWORK: {"driver": "bridge"}}}
    if volumes:
        document["volumes"] = {name: {} for name in volumes}
    return document


def build_compose(opts: DeployOptions) -> dict[str, dict]:
    """Retourne {nom de fichier: document Compose}."""

    profiles = load_profiles(opts)
    files: dict[str, dict] = {}
    local_services = {host_name(p): worker_service(p, opts) for p in profiles}
    local_services.update(orchestrator_services(opts, None, local=True))  # DNS Docker : http://<noeud en minuscules>:<port>
    files["compose.local.yml"] = _compose(local_services, ["prometheus-data", "grafana-data"])
    for vm in sorted({p.vm for p in profiles}):
        files[f"compose.{vm}.yml"] = _compose({host_name(p): worker_service(p, opts) for p in profiles if p.vm == vm})
    urls = {p.node_id: worker_url(p, opts, local=False) for p in profiles}
    files["compose.control.yml"] = _compose(orchestrator_services(opts, urls), ["prometheus-data", "grafana-data"])
    return files


def build_prometheus(opts: DeployOptions, local: bool) -> dict:
    profiles = load_profiles(opts)
    orchestrator = f"orchestrator:{opts.orchestrator_port}"
    workers = [f"{host_name(p)}:{p.port}" if local else f"{opts.hosts[p.vm]}:{p.port}" for p in profiles]
    return {
        "global": {"scrape_interval": "5s", "evaluation_interval": "15s"},
        "scrape_configs": [
            {"job_name": "orchestrator", "static_configs": [{"targets": [orchestrator]}]},
            {"job_name": "workers", "static_configs": [
                {"targets": [target], "labels": {"node": p.node_id}} for target, p in zip(workers, profiles)
            ]},
        ],
    }


def grafana_provisioning() -> dict[str, dict]:
    return {
        "grafana/provisioning/datasources/prometheus.yml": {
            "apiVersion": 1,
            "datasources": [{"name": "Prometheus", "uid": "prometheus", "type": "prometheus", "access": "proxy",
                             "url": "http://prometheus:9090", "isDefault": True}],
        },
        "grafana/provisioning/dashboards/dashboards.yml": {
            "apiVersion": 1,
            "providers": [{"name": "fog-rl", "type": "file", "disableDeletion": True, "allowUiUpdates": True,
                           "options": {"path": "/var/lib/grafana/dashboards"}}],
        },
    }


def resource_summary(opts: DeployOptions) -> dict[str, dict]:
    """CPU et memoire demandes par VM (limites des conteneurs de la configuration)."""

    summary: dict[str, dict] = {}
    for profile in load_profiles(opts):
        vm = summary.setdefault(profile.vm, {"cpus": 0.0, "mem_mb": 0, "nodes": []})
        vm["cpus"] += profile.cpus
        vm["mem_mb"] += profile.mem_mb
        vm["nodes"].append(profile.node_id)
    return summary


def resource_warnings(opts: DeployOptions) -> list[str]:
    warnings = []
    for vm, info in resource_summary(opts).items():
        available = opts.vm_cpus.get(vm)
        if available is not None and info["cpus"] > available:
            warnings.append(
                f"{vm} : {info['nodes']} demandent {info['cpus']:g} CPU pour {available:g} disponibles. "
                "Reduisez `cpus` (et `servers`) dans configs/nodes/pool6.yaml, puis recalibrez."
            )
    return warnings


def _dump(document: dict) -> str:
    return HEADER + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=120)


def write_all(opts: DeployOptions, out_dir) -> list[Path]:
    out = Path(out_dir)
    written: list[Path] = []

    def write(relative: str, text: str) -> None:
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)

    for name, document in build_compose(opts).items():
        write(name, _dump(document))
    write("prometheus/prometheus.yml", _dump(build_prometheus(opts, local=False)))
    write("prometheus/prometheus.local.yml", _dump(build_prometheus(opts, local=True)))
    for relative, document in grafana_provisioning().items():
        write(relative, _dump(document))
    write("grafana/dashboards/fog-rl.json", json.dumps(build_dashboard(), indent=2, ensure_ascii=False) + "\n")
    return written


def parse_kv_floats(text: str) -> dict[str, float]:
    return {key: float(value) for key, value in parse_urls(text).items()} if text else {}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes-file", default="configs/nodes/pool6.yaml")
    parser.add_argument("--configuration", default="n4", help="n3, n4 ou n6")
    parser.add_argument("--hosts", default="", help="vm1=10.0.0.1,vm2=10.0.0.2,vm3=10.0.0.3 (defaut : vm1,vm2,vm3)")
    parser.add_argument("--control-vm", default="vm1")
    parser.add_argument("--worker-mode", choices=("burn", "real"), default="burn")
    parser.add_argument("--emulate-speed", action="store_true", help="pas de limites CPU : heterogeneite emulee par l'application")
    parser.add_argument("--policy", default="least_connections")
    parser.add_argument("--vm-cpus", default="", help="vm1=2,vm2=4,vm3=8 : avertit si les limites depassent")
    parser.add_argument("--out", default="deploy")
    args = parser.parse_args(argv)

    hosts = {"vm1": "vm1", "vm2": "vm2", "vm3": "vm3", **parse_urls(args.hosts)}
    opts = DeployOptions(
        nodes_file=args.nodes_file,
        configuration=args.configuration,
        hosts=hosts,
        control_vm=args.control_vm,
        worker_mode=args.worker_mode,
        emulate_speed=args.emulate_speed,
        policy=args.policy,
        vm_cpus=parse_kv_floats(args.vm_cpus),
    )
    for path in write_all(opts, args.out):
        print("ecrit", path)
    for vm, info in sorted(resource_summary(opts).items()):
        print(f"{vm}: {info['cpus']:g} CPU, {info['mem_mb']} Mo pour {', '.join(info['nodes'])}")
    for warning in resource_warnings(opts):
        print("ATTENTION :", warning)


if __name__ == "__main__":
    main()
