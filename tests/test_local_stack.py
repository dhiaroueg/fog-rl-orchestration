"""Pile locale sans Docker : vrais sous-processus et vrai HTTP (tests lents)."""

import asyncio
import sys
import warnings

import httpx
import pytest
from helpers_stack import route_body

from fogrl.config import EnvConfig
from fogrl.deploy.local import LocalStack
from fogrl.loadgen.generator import run_scenario

warnings.filterwarnings("ignore")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(sys.platform == "win32", reason="pile locale : Linux, macOS ou WSL")]


def test_stack_serves_requests_over_real_http_and_stops_every_process(tmp_path):
    with LocalStack("n4", "greedy", executor="thread", log_dir=str(tmp_path)) as stack:
        assert httpx.get(f"{stack.url}/health", timeout=5.0).status_code == 200
        assert set(stack.worker_urls) == {"fogA", "fogB", "fogC", "cloud"}
        for node_id, url in stack.worker_urls.items():
            info = httpx.get(f"{url}/info", timeout=5.0).json()
            assert info["node_id"] == node_id and info["emulate_speed"] is True
            assert info["simulated_rtt_ms"] == pytest.approx(stack.resolved.profiles[node_id].rtt_ms)
        reply = httpx.post(f"{stack.url}/route", json=route_body("light", 5.0), timeout=10.0).json()
        assert reply["node"] in stack.worker_urls and reply["rejected"] is False and reply["fallback"] is False
        assert 0.0 < reply["latency_ms"] < 2000.0 and reply["decision_ms"] < 20.0
        metrics = httpx.get(f"{stack.url}/metrics", timeout=5.0).text
        assert "fog_route_requests_total" in metrics and 'fog_monitor_node_up{node="cloud"} 1.0' in metrics
        processes = list(stack.processes.values())
        assert len(processes) == 5 and all(p.poll() is None for p in processes)
    assert all(p.poll() is not None for p in processes) and stack.processes == {}
    assert {p.name for p in tmp_path.glob("*.log")} == {"fogA.log", "fogB.log", "fogC.log", "cloud.log", "orchestrator.log"}


def test_load_generator_replays_a_scenario_against_real_processes(tiny_env, tmp_path):
    config = EnvConfig.from_yaml(tiny_env)
    with LocalStack("n4", "greedy", tiny_env, executor="thread", log_dir=str(tmp_path)) as stack:

        async def scenario():
            async with httpx.AsyncClient(base_url=stack.url, timeout=10.0) as client:
                return await run_scenario(client, config, "n4", "tiny", 1000, duration_s=2.5)

        summary, rows = asyncio.run(scenario())
    assert len(rows) == summary["n_requests"] >= 20
    assert summary["rejection_rate"] == 0.0 and summary["fallback_rate"] == 0.0
    assert summary["decision_ms_p99"] < 20.0 and summary["loadgen_lag_ms_p99"] < 250.0


def test_failed_start_reports_the_log_and_leaves_no_process_behind(tmp_path):
    stack = LocalStack("n4", "no_such_policy", executor="thread", log_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="orchestrator s'est arrete"):
        stack.start(timeout_s=60.0)
    assert stack.processes == {}
    for url in stack.worker_urls.values():  # les workers deja lances ont ete arretes
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"{url}/health", timeout=0.5)
