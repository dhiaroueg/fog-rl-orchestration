"""Fichiers de deploiement : Compose, Prometheus, Grafana, Dockerfiles et scripts.

Docker n'est pas utilise ici (pas de daemon) : on verifie la structure et la coherence avec
configs/nodes/pool6.yaml. La CI lance `docker compose config` et construit l'image du worker.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import yaml
from helpers_stack import route_body, stack

from fogrl.config import PROJECT_ROOT, load_nodes
from fogrl.deploy.dashboard import build_dashboard
from fogrl.deploy.generate import (
    DeployOptions,
    build_compose,
    build_prometheus,
    load_profiles,
    resource_summary,
    resource_warnings,
    write_all,
)
from fogrl.orchestrator.settings import OrchestratorSettings

warnings.filterwarnings("ignore")

HOSTS = {"vm1": "192.168.56.11", "vm2": "192.168.56.12", "vm3": "192.168.56.13"}
POOL = {p.node_id: p for p in load_nodes("configs/nodes/pool6.yaml")[0]}


def opts(**overrides):
    return DeployOptions(hosts=HOSTS, **overrides)


def test_one_compose_file_per_vm_with_the_workers_of_the_pool_assignment():
    files = build_compose(opts(configuration="n6"))
    assert set(files) == {"compose.local.yml", "compose.vm1.yml", "compose.vm2.yml", "compose.vm3.yml", "compose.control.yml"}
    for vm in ("vm1", "vm2", "vm3"):
        expected = {p.node_id.lower() for p in POOL.values() if p.vm == vm}
        assert set(files[f"compose.{vm}.yml"]["services"]) == expected
    assert set(files["compose.local.yml"]["services"]) >= {p.lower() for p in POOL} | {"orchestrator", "prometheus", "grafana"}
    n4 = build_compose(opts(configuration="n4"))
    assert set(n4["compose.vm1.yml"]["services"]) == {"foga"}  # fogD absent de la configuration n4


def test_workers_get_their_profile_limits_latency_and_ports():
    services = build_compose(opts())["compose.vm3.yml"]["services"]
    fog_c, cloud = services["fogc"], services["cloud"]
    for service, profile in ((fog_c, POOL["fogC"]), (cloud, POOL["cloud"])):
        env = service["environment"]
        assert env["NODE_ID"] == profile.node_id and env["PORT"] == str(profile.port)
        assert env["NETEM_DELAY_MS"] == str(profile.rtt_ms)
        assert float(env["NETEM_JITTER_MS"]) == pytest.approx(profile.rtt_ms * profile.rtt_sigma, abs=0.01)
        assert env["EMULATE_SPEED"] == "0" and env["SIMULATED_RTT_MS"] == "0"  # netem + cgroups font le travail
        assert service["cpus"] == profile.cpus and service["mem_limit"] == f"{profile.mem_mb}m"
        assert service["cap_add"] == ["NET_ADMIN"] and service["ports"] == [f"{profile.port}:{profile.port}"]
        assert service["healthcheck"]["test"][0] == "CMD"


def test_emulate_speed_removes_the_cpu_limits_and_real_mode_mounts_the_onnx_model():
    service = build_compose(opts(emulate_speed=True))["compose.vm1.yml"]["services"]["foga"]
    assert "cpus" not in service and "mem_limit" not in service and service["environment"]["EMULATE_SPEED"] == "1"
    service = build_compose(opts(worker_mode="real"))["compose.vm1.yml"]["services"]["foga"]
    assert service["environment"]["ONNX_MODEL"] == "/models/mobilenet_v2.onnx"
    assert service["volumes"] == ["../models_onnx:/models:ro"]


def test_control_compose_points_the_orchestrator_at_the_vm_addresses():
    control = build_compose(opts(policy="greedy"))["compose.control.yml"]["services"]
    env = control["orchestrator"]["environment"]
    urls = dict(item.split("=", 1) for item in env["ORCH_NODE_URLS"].split(","))
    assert urls == {n: f"http://{HOSTS[POOL[n].vm]}:{POOL[n].port}" for n in ("fogA", "fogB", "fogC", "cloud")}
    assert env["ORCH_POLICY"] == "greedy" and env["ORCH_CONFIGURATION"] == "n4"
    assert set(control) == {"orchestrator", "loadgen", "prometheus", "grafana"}
    assert control["loadgen"]["profiles"] == ["tools"]
    assert "../models:/app/models:ro" in control["orchestrator"]["volumes"]


def test_local_compose_service_names_match_the_default_orchestrator_urls():
    """Sans ORCH_NODE_URLS, l'orchestrateur vise http://<noeud en minuscules>:<port> : ce sont les services Compose."""

    local = build_compose(opts())["compose.local.yml"]["services"]
    assert "ORCH_NODE_URLS" not in local["orchestrator"]["environment"]
    resolved = OrchestratorSettings().resolve()
    for node_id, url in resolved.urls.items():
        host_port = url.removeprefix("http://")
        host, port = host_port.split(":")
        assert host in local and local[host]["environment"]["PORT"] == port and local[host]["hostname"] == host
    assert "prometheus.local.yml" in local["prometheus"]["volumes"][0]


def test_prometheus_scrapes_the_orchestrator_and_every_worker():
    config = build_prometheus(opts(), local=False)
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert jobs["orchestrator"]["static_configs"][0]["targets"] == ["orchestrator:9000"]
    targets = {t["labels"]["node"]: t["targets"][0] for t in jobs["workers"]["static_configs"]}
    assert targets == {n: f"{HOSTS[POOL[n].vm]}:{POOL[n].port}" for n in ("fogA", "fogB", "fogC", "cloud")}
    local = build_prometheus(opts(), local=True)
    assert {t["targets"][0] for t in local["scrape_configs"][1]["static_configs"]} == {
        f"{n.lower()}:{POOL[n].port}" for n in ("fogA", "fogB", "fogC", "cloud")
    }


def test_invalid_deployments_are_refused_and_resources_checked():
    with pytest.raises(ValueError, match="Configuration inconnue"):
        load_profiles(opts(configuration="n9"))
    with pytest.raises(ValueError, match="VM sans adresse"):
        load_profiles(DeployOptions(hosts={"vm1": "a", "vm2": "b"}))
    summary = resource_summary(opts())
    assert summary["vm3"]["cpus"] == POOL["fogC"].cpus + POOL["cloud"].cpus and summary["vm3"]["nodes"] == ["fogC", "cloud"]
    warnings_ = resource_warnings(opts(vm_cpus={"vm1": 2, "vm3": 4}))
    assert len(warnings_) == 1 and "vm3" in warnings_[0] and "recalibrez" in warnings_[0]
    assert resource_warnings(opts(vm_cpus={"vm1": 2, "vm3": 8})) == []


def test_write_all_produces_lf_header_marked_valid_files(tmp_path):
    written = write_all(opts(), tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in written}
    assert {"compose.local.yml", "compose.control.yml", "prometheus/prometheus.yml", "prometheus/prometheus.local.yml",
            "grafana/provisioning/datasources/prometheus.yml", "grafana/provisioning/dashboards/dashboards.yml",
            "grafana/dashboards/fog-rl.json"} <= names
    for path in written:
        raw = path.read_bytes()
        assert b"\r" not in raw, f"{path} : fins de ligne CRLF"
        if path.suffix in (".yml", ".yaml"):
            assert raw.startswith(b"# GENERE par scripts/gen_compose.py")
            assert isinstance(yaml.safe_load(raw), dict)
        else:
            assert json.loads(raw)["uid"] == "fog-rl"
    for name in ("compose.local.yml", "compose.vm3.yml", "compose.control.yml"):  # ports hotes uniques
        services = yaml.safe_load((tmp_path / name).read_text())["services"]
        published = [p.split(":")[0] for s in services.values() for p in s.get("ports", [])]
        assert len(published) == len(set(published)), name


# --------------------------------------------------------------------- tableau de bord


class BrokenPolicy:
    """Politique qui leve une exception : l'orchestrateur doit basculer sur le repli (metrique de repli)."""

    name = "broken"
    needs_budget = False

    def reset(self, seed=None):
        pass

    def select(self, nodes, request, context):
        raise ValueError("politique cassee")


def exposed_metrics():
    """Expositions reelles de l'orchestrateur et d'un worker, apres une requete normale et une requete en repli."""

    async def scenario():
        async with stack("greedy") as client:
            await client.post("/route", json=route_body("medium", 10.0))
            client.router.engine.swap(BrokenPolicy(), client.resolved.active_ids)
            await client.post("/route", json=route_body("light", 5.0))
            orchestrator = (await client.get("/metrics")).text
            worker = (await client.router.clients["fogA"].get("/metrics")).text
            return orchestrator + worker

    return asyncio.run(scenario())


def exposed_names(text):
    """Noms utilisables dans PromQL : echantillons presents + familles declarees (un compteur a labels n'a
    aucun echantillon avant son premier increment, mais sa famille est deja annoncee par # TYPE)."""

    names = set()
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            names.add(name)
            if kind == "counter":
                names.add(name if name.endswith("_total") else f"{name}_total")
            elif kind == "histogram":
                names |= {f"{name}_bucket", f"{name}_sum", f"{name}_count"}
        elif line.strip() and not line.startswith("#"):
            names.add(re.match(r"[A-Za-z_:][A-Za-z0-9_:]*", line).group(0))
    return names


def test_every_metric_used_by_the_dashboard_is_exposed_by_the_applications():
    names = exposed_names(exposed_metrics())
    dashboard = build_dashboard()
    ids = [panel["id"] for panel in dashboard["panels"]]
    assert len(ids) == len(set(ids)) and dashboard["uid"] == "fog-rl"
    for panel in dashboard["panels"]:
        assert panel["gridPos"]["x"] + panel["gridPos"]["w"] <= 24 and panel["datasource"]["uid"] == "prometheus"
        for target in panel["targets"]:
            for metric in set(re.findall(r"\bfog_[a-z_]+", target["expr"])):
                assert metric in names, f"panneau {panel['title']!r} : metrique absente {metric}"


# ------------------------------------------------------------------ Docker et scripts


def test_dockerfiles_copy_existing_paths_and_make_the_entrypoint_executable():
    for dockerfile in (PROJECT_ROOT / "docker").glob("Dockerfile.*"):
        lines = dockerfile.read_text().splitlines()
        assert lines[next(i for i, l in enumerate(lines) if l.startswith("FROM"))].startswith("FROM python:3.13")
        for line in lines:
            if line.startswith("COPY "):
                *sources, _ = line.split()[1:]
                for source in sources:
                    assert (PROJECT_ROOT / source).exists(), f"{dockerfile.name} : COPY {source} introuvable"
    worker = (PROJECT_ROOT / "docker" / "Dockerfile.worker").read_text()
    instructions = "\n".join(line for line in worker.splitlines() if not line.lstrip().startswith("#"))
    assert "iproute2" in instructions and "ENTRYPOINT" in instructions
    # Le bit executable est perdu par un commit depuis Windows : l'image le rend elle-meme executable.
    assert "chmod +x /usr/local/bin/entrypoint-worker.sh" in instructions
    assert "torch" not in instructions and "requirements-worker.txt" in instructions  # image legere


POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="scripts shell : Linux, macOS ou WSL")


@POSIX_ONLY
def test_shell_scripts_have_valid_syntax_and_lf_endings():
    scripts = sorted((PROJECT_ROOT / "scripts").glob("*.sh")) + [PROJECT_ROOT / "docker" / "entrypoint-worker.sh"]
    assert len(scripts) >= 4
    for script in scripts:
        assert b"\r" not in script.read_bytes(), f"{script.name} : CRLF (Docker/Linux refuserait le script)"
        shell = "sh" if script.name == "entrypoint-worker.sh" else "bash"
        result = subprocess.run([shell, "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, f"{script.name} : {result.stderr}"


@POSIX_ONLY
def test_entrypoint_runs_the_command_and_warns_when_netem_is_unavailable(tmp_path):
    entrypoint = ["sh", str(PROJECT_ROOT / "docker" / "entrypoint-worker.sh")]  # sans dependre du bit executable
    env = {**os.environ, "NETEM_DELAY_MS": "20", "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    fake_tc = tmp_path / "tc"
    fake_tc.write_text("#!/bin/sh\nexit 1\n")
    fake_tc.chmod(0o755)
    result = subprocess.run(
        [*entrypoint, "echo", "worker-demarre"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0 and "worker-demarre" in result.stdout
    assert "NON injectee" in result.stderr  # le worker demarre quand meme
    fake_tc.write_text("#!/bin/sh\necho \"tc $*\" >> " + str(tmp_path / "tc.log") + "\nexit 0\n")
    result = subprocess.run(
        [*entrypoint, "true"],
        capture_output=True, text=True, env={**env, "NETEM_JITTER_MS": "3", "NETEM_IFACE": "eth9"},
    )
    assert result.returncode == 0
    assert "qdisc replace dev eth9 root netem delay 20ms 3ms distribution normal" in (tmp_path / "tc.log").read_text()
    quiet = subprocess.run([*entrypoint, "true"], capture_output=True, text=True,
                           env={k: v for k, v in env.items() if k != "NETEM_DELAY_MS"})
    assert quiet.returncode == 0 and "netem" not in quiet.stdout + quiet.stderr  # pas de delai demande : rien a faire
