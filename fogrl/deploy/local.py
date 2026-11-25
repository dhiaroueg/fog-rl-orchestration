"""Pile complete sur une seule machine, SANS Docker : workers + orchestrateur en sous-processus.

Vrai HTTP et vrais processus (un par serveur de noeud), mais une seule machine partage les coeurs :
l'heterogeneite est emulee par l'application (``EMULATE_SPEED=1``) et la latence reseau par
``SIMULATED_RTT_MS`` (pas de ``tc netem``). Sert a essayer le prototype, a la CI et a verifier la
chaine complete avant de deployer sur les VM ; les mesures obtenues ne remplacent pas celles des VM.

    python scripts/local_stack.py --policy greedy                       # pile en marche (Ctrl-C pour arreter)
    python scripts/local_stack.py --campaign configs/experiments/local_smoke.yaml --out results/local_smoke
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import httpx

from ..config import PROJECT_ROOT, load_yaml
from ..orchestrator.settings import OrchestratorSettings


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LocalStack:
    def __init__(
        self,
        configuration: str = "n4",
        policy: str = "least_connections",
        env_file: str = "configs/env/eval.yaml",
        *,
        executor: str = "process",
        emulate_speed: bool = True,
        simulated_rtt: bool = True,
        decision_log: Optional[str] = None,
        log_dir: Optional[str] = None,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if sys.platform == "win32":
            raise RuntimeError("La pile locale exige Linux, macOS ou WSL (groupes de processus POSIX) : lancez-la dans WSL.")
        self.settings = OrchestratorSettings(env_file=env_file, configuration=configuration, policy=policy)
        self.resolved = self.settings.resolve()
        self.nodes_file = load_yaml(env_file).get("nodes_file", "configs/nodes/pool6.yaml")
        self.executor, self.emulate_speed, self.simulated_rtt = executor, emulate_speed, simulated_rtt
        self.decision_log = decision_log
        self.log_dir = Path(log_dir or tempfile.mkdtemp(prefix="fogrl-stack-"))
        self.extra_env = dict(extra_env or {})
        self.ports = {node_id: free_port() for node_id in self.resolved.active_ids}
        self.orchestrator_port = free_port()
        self.processes: dict[str, subprocess.Popen] = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.orchestrator_port}"

    @property
    def worker_urls(self) -> dict[str, str]:
        return {node_id: f"http://127.0.0.1:{port}" for node_id, port in self.ports.items()}

    # ------------------------------------------------------------------ cycle de vie

    def _spawn(self, name: str, module: str, env: Mapping[str, str]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"{name}.log", "w", encoding="utf-8")
        merged = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "HOST": "127.0.0.1", **self.extra_env, **env}
        self.processes[name] = subprocess.Popen(
            [sys.executable, "-m", module],
            env=merged,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # un groupe par processus : on arrete aussi les processus de serveurs
        )

    def start(self, timeout_s: float = 90.0) -> "LocalStack":
        try:
            for node_id, port in self.ports.items():
                profile = self.resolved.profiles[node_id]
                self._spawn(
                    node_id,
                    "fogrl.worker.app",
                    {
                        "NODE_ID": node_id,
                        "NODES_FILE": self.nodes_file,
                        "PORT": str(port),
                        "WORKER_MODE": "burn",
                        "WORKER_EXECUTOR": self.executor,
                        "EMULATE_SPEED": "1" if self.emulate_speed else "0",
                        "SIMULATED_RTT_MS": str(profile.rtt_ms) if self.simulated_rtt else "0",
                    },
                )
            urls = ",".join(f"{node}={url}" for node, url in self.worker_urls.items())
            env = {
                "PORT": str(self.orchestrator_port),
                "ORCH_ENV_FILE": self.settings.env_file,
                "ORCH_CONFIGURATION": self.settings.configuration,
                "ORCH_POLICY": self.settings.policy,
                "ORCH_NODE_URLS": urls,
            }
            if self.decision_log:
                env["DECISION_LOG"] = self.decision_log
            self._spawn("orchestrator", "fogrl.orchestrator.app", env)
            self._wait_ready(timeout_s)
        except BaseException:
            self.stop()
            raise
        return self

    def _wait_ready(self, timeout_s: float) -> None:
        targets = {**self.worker_urls, "orchestrator": self.url}
        deadline = time.monotonic() + timeout_s
        pending = dict(targets)
        while pending:
            for name, url in list(pending.items()):
                process = self.processes[name]
                if process.poll() is not None:
                    raise RuntimeError(f"{name} s'est arrete (code {process.returncode}) :\n{self.tail(name)}")
                try:
                    if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                        del pending[name]
                except httpx.HTTPError:
                    pass
            if pending and time.monotonic() > deadline:
                raise TimeoutError(f"Demarrage trop lent : {sorted(pending)} ; journaux dans {self.log_dir}")
            if pending:
                time.sleep(0.3)

    def tail(self, name: str, lines: int = 25) -> str:
        path = self.log_dir / f"{name}.log"
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) if path.exists() else ""

    def stop(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 8.0
        for process in self.processes.values():
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.processes.clear()

    def __enter__(self) -> "LocalStack":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - lance des processus
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configuration", default="n4")
    parser.add_argument("--policy", default="least_connections")
    parser.add_argument("--env", default="configs/env/eval.yaml")
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--no-emulate-speed", action="store_true")
    parser.add_argument("--no-simulated-rtt", action="store_true")
    parser.add_argument("--campaign", help="matrice de campagne (YAML) : la lance puis arrete la pile")
    parser.add_argument("--out", default="results/local_campaign")
    parser.add_argument("--decision-log", default=None)
    args = parser.parse_args(argv)

    stack = LocalStack(
        args.configuration,
        args.policy,
        args.env,
        executor=args.executor,
        emulate_speed=not args.no_emulate_speed,
        simulated_rtt=not args.no_simulated_rtt,
        decision_log=args.decision_log,
    )
    with stack:
        print(f"Orchestrateur : {stack.url}   (workers : {stack.worker_urls}) ; journaux : {stack.log_dir}", flush=True)
        if args.campaign:
            from ..loadgen.campaign import run_campaign

            matrix = dict(load_yaml(args.campaign))
            matrix["orchestrator_url"] = stack.url
            path = asyncio.run(run_campaign(matrix, args.out))
            print(f"Resultats : {path}")
            return
        print("Ctrl-C pour arreter.", flush=True)
        try:
            signal.pause()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":  # pragma: no cover
    main()
