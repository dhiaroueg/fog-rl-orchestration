"""Generateur de charge et campagnes : memes metriques et memes colonnes qu'en simulation."""

import asyncio
import json
import warnings

import httpx
import numpy as np
import pandas as pd
import pytest
from helpers_stack import stack

from fogrl.config import EnvConfig, RewardConfig, load_yaml
from fogrl.evaluation.compare import compare_sim_real
from fogrl.evaluation.evaluate import evaluate_policy, make_policy
from fogrl.loadgen.campaign import policy_spec, run_campaign
from fogrl.loadgen.generator import run_load, run_scenario, summarize_run
from fogrl.sim.env import FogEnv
from fogrl.traffic import generate_stream_until

warnings.filterwarnings("ignore")


def test_run_scenario_end_to_end_reports_simulator_metrics(tiny_env):
    config = EnvConfig.from_yaml(tiny_env)

    async def scenario():
        async with stack("greedy", env_file=tiny_env) as client:
            return await run_scenario(client, config, "n4", "tiny", 1000, duration_s=1.5)

    summary, rows = asyncio.run(scenario())
    assert len(rows) == summary["n_requests"] >= 15
    assert summary["rejection_rate"] == 0.0 and summary["deadline_rate"] > 0.9
    assert summary["fallback_rate"] == 0.0 and summary["decision_ms_p99"] < 20.0
    assert summary["energy_fog_j_per_req"] > 0 and 0.0 < summary["jain_cpu"] <= 1.0
    assert summary["scenario"] == "tiny" and summary["size"] == 4 and summary["configuration"] == "n4"
    assert summary["loadgen_lag_ms_p99"] < 250.0
    shares = {k: v for k, v in summary.items() if k.startswith("share_")}
    assert set(shares) == {f"share_{n}" for n in ("fogA", "fogB", "fogC", "fogD", "fogE", "cloud")}
    assert sum(shares.values()) == pytest.approx(1.0) and shares["share_fogD"] == 0.0
    assert all(r["latency_ms"] > 0 and r["e2e_ms"] >= r["latency_ms"] for r in rows)


def test_same_seed_replays_the_same_request_sequence(tiny_env):
    config = EnvConfig.from_yaml(tiny_env)

    async def scenario():
        async with stack("least_connections", env_file=tiny_env) as client:
            first = await run_scenario(client, config, "n4", "tiny", 1001, duration_s=1.0)
            second = await run_scenario(client, config, "n4", "tiny", 1001, duration_s=1.0)
            other = await run_scenario(client, config, "n4", "tiny", 1002, duration_s=1.0)
            return first[1], second[1], other[1]

    first, second, other = asyncio.run(scenario())
    key = lambda rows: [(r["type"], round(r["work_ms"], 9), round(r["target_s"], 9)) for r in rows]
    assert key(first) == key(second) != key(other)  # EF-06


def test_loadgen_errors_are_counted_as_rejections(tiny_env):
    config = EnvConfig.from_yaml(tiny_env)
    stream = generate_stream_until(config.scenarios[0], 30.0, 0.3, config.requests, 1000)

    def refuse(request):
        raise httpx.ConnectError("orchestrateur injoignable", request=request)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://orchestrator") as client:
            return await run_load(client, stream, {r.name: r.deadline_ms for r in config.requests})

    rows = asyncio.run(scenario())
    assert len(rows) == len(stream) and all(r["rejected"] and r["reason"] == "loadgen_ConnectError" for r in rows)


def test_summarize_run_follows_the_simulator_definitions():
    profiles = {p.node_id: p for p in EnvConfig.from_yaml("configs/env/eval.yaml").pool}
    slots = ("fogA", "fogB", "fogC", "fogD", "fogE", "cloud")
    base = dict(deadline_ms=100.0, decision_ms=0.5, fallback=False, lag_ms=1.0)
    rows = [
        {**base, "node": "fogA", "rejected": False, "latency_ms": 50.0, "energy_j": 0.4, "cost": 0.0},
        {**base, "node": "fogB", "rejected": False, "latency_ms": 150.0, "energy_j": 1.0, "cost": 0.0},  # echeance manquee
        {**base, "node": "cloud", "rejected": False, "latency_ms": 90.0, "energy_j": 0.0, "cost": 1.0},
        {**base, "node": "fogA", "rejected": True, "latency_ms": None, "energy_j": 0.0, "cost": 0.0, "fallback": True},
    ]
    zero = {n: {"energy_idle_j": 0.0, "busy_time_s": 0.0} for n in ("fogA", "fogB", "fogC", "cloud")}
    after = {
        "fogA": {"energy_idle_j": 27.0, "busy_time_s": 5.0},
        "fogB": {"energy_idle_j": 50.0, "busy_time_s": 5.0},
        "fogC": {"energy_idle_j": 100.0, "busy_time_s": 0.0},
        "cloud": {"energy_idle_j": 0.0, "busy_time_s": 3.0},  # le cloud est exclu de l'energie fog et de Jain
    }
    reward = RewardConfig(w1=1.0, w2=0.5, w3=0.3, w4=2.0, w5=5.0)
    summary = summarize_run(
        rows, slot_ids=slots, profiles=profiles, reward=reward, duration_s=10.0, nodes_before=zero, nodes_after=after
    )
    assert summary["n_requests"] == 4 and summary["n_rejected"] == 1 and summary["rejection_rate"] == 0.25
    assert summary["deadline_rate"] == pytest.approx(0.5)  # 50 ms et 90 ms ; le rejet compte comme manque
    assert summary["latency_mean_ms"] == pytest.approx((50 + 150 + 90) / 3)
    assert summary["energy_fog_total_j"] == pytest.approx(1.4 + 177.0)  # dynamique + a vide, sans le cloud
    assert summary["cloud_cost_per_req"] == pytest.approx(0.25)
    assert summary["share_fogA"] == 0.5 and summary["share_cloud"] == 0.25
    utilization = [5.0 / (1 * 10), 5.0 / (1 * 10), 0.0]  # fogA, fogB, fogC (2 serveurs)
    expected_jain = sum(utilization) ** 2 / (3 * sum(u * u for u in utilization))
    assert summary["jain_cpu"] == pytest.approx(expected_jain)
    assert summary["fallback_rate"] == 0.25 and summary["decision_ms_mean"] == 0.5
    # recompense : 0.5 -> -(50/200 + .5*.4) ; 150 ms -> -(150/200 + .5*1 + 2) ; 90 ms + cout -> -(90/200 + .3) ; rejet -> -(300/100*... plafonne)
    cap = 3 * 100.0
    expected = -(0.25 + 0.2) - (0.75 + 0.5 + 2.0) - (0.45 + 0.3) - (cap / 200 + 2.0 + 5.0)
    assert summary["reward_sum"] == pytest.approx(expected)


def test_campaign_csv_is_comparable_with_the_simulator(tiny_env, tmp_path):
    matrix = {
        "name": "mini",
        "env": tiny_env,
        "configuration": "n4",
        "scenarios": ["tiny"],
        "seeds": [1000, 1001],
        "duration_s": 1.0,
        "policies": [{"name": "greedy", "type": "baseline"}, {"name": "round_robin", "type": "baseline"}],
    }

    async def scenario():
        async with stack("least_connections", env_file=tiny_env) as client:
            return await run_campaign(matrix, tmp_path / "real", client=client, verbose=False, save_requests=True)

    results = asyncio.run(scenario())
    real = pd.read_csv(results)
    assert len(real) == 4 and set(real["policy"]) == {"greedy", "round_robin"} and set(real["source"]) == {"real"}
    assert (tmp_path / "real" / "requests" / "greedy_tiny_1000.csv").exists()
    info = json.loads((tmp_path / "real" / "run_info.json").read_text())
    assert info["commit"] and info["n_rows"] == 4 and info["matrix"]["name"] == "mini" and info["env_digest"]

    config = EnvConfig.from_yaml(tiny_env)
    env, profiles = FogEnv(config), {p.node_id: p for p in config.pool}
    sim_rows = []
    for name in ("greedy", "round_robin"):
        sim_rows += evaluate_policy(
            make_policy(name, profiles), env, [1000, 1001], policy_name=name, policy_type="baseline",
            scenario="tiny", configuration="n4",
        )
    sim = pd.DataFrame(sim_rows)
    missing = set(sim.columns) - set(real.columns)
    assert not missing, f"colonnes du simulateur absentes du CSV reel : {sorted(missing)}"
    report = compare_sim_real(sim, real)  # s'execute sans erreur sur les deux CSV
    assert {pair["policy"] for pair in report["pairs"]} == {"greedy", "round_robin"}


def test_campaign_validates_seeds_and_reports_a_refused_policy(tiny_env, tmp_path):
    base = {"env": tiny_env, "scenarios": ["tiny"], "duration_s": 0.5, "policies": [{"name": "greedy"}]}
    with pytest.raises(ValueError, match="1000"):
        asyncio.run(run_campaign({**base, "seeds": [5]}, tmp_path / "a", verbose=False))

    async def scenario():
        async with stack("greedy", env_file=tiny_env) as client:
            matrix = {**base, "seeds": [1000], "policies": [{"name": "magic"}]}
            await run_campaign(matrix, tmp_path / "b", client=client, verbose=False)

    with pytest.raises(RuntimeError, match="magic"):
        asyncio.run(scenario())


def test_policy_spec_builds_baseline_and_model_specs():
    assert policy_spec({"name": "greedy", "type": "baseline"}) == "greedy"
    assert policy_spec({"name": "ppo", "type": "rl", "model": "models/n3/ppo/seed0"}) == "rl:models/n3/ppo/seed0"
