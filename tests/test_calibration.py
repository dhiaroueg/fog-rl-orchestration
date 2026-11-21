"""Calibration : mesure sur des workers (en memoire), reduction en table, utilisation par le simulateur."""

import asyncio
import dataclasses
import json
import warnings
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
import numpy as np
import pandas as pd
import pytest

from fogrl.calibration.fit import MIN_SAMPLES, build_calibration, fit_service_table, render_report
from fogrl.calibration.measure import CalibrationPlan, run_calibration, save_measurements
from fogrl.config import EnvConfig, load_nodes
from fogrl.evaluation.evaluate import evaluate_policy, make_policy
from fogrl.sim.env import FogEnv
from fogrl.sim.service import CalibratedServiceModel
from fogrl.worker.app import WorkerSettings, create_app

warnings.filterwarnings("ignore")

PROFILES = {p.node_id: p for p in load_nodes("configs/nodes/pool6.yaml")[0]}
NOMINAL = {"light": 8.0, "medium": 16.0, "heavy": 32.0}


@pytest.fixture()
def tiny_requests(tmp_path):
    path = tmp_path / "requests.yaml"
    path.write_text(
        "requests:\n"
        "  light: {work_ms: 8, deadline_ms: 100}\n"
        "  medium: {work_ms: 16, deadline_ms: 300}\n"
        "  heavy: {work_ms: 32, deadline_ms: 1000}\n"
    )
    return str(path)


@asynccontextmanager
async def workers(overrides=None, **worker_settings):
    overrides = overrides or {}
    async with AsyncExitStack() as stack:
        clients = {}
        for node_id in ("fogA", "fogC"):
            profile = dataclasses.replace(PROFILES[node_id], **overrides.get(node_id, {}))
            settings = WorkerSettings(node_id=node_id, executor="thread", **worker_settings)
            app = create_app(settings, profile)
            await stack.enter_async_context(app.router.lifespan_context(app))
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker")
            clients[node_id] = await stack.enter_async_context(client)
        yield clients


def synthetic_raw(speed=0.5, contention=0.4, noise=0.05, n=40, seed=0, node="fogA"):
    rng = np.random.default_rng(seed)
    rows = []
    for req_type, nominal in NOMINAL.items():
        for load in (0.0, 0.25, 0.5, 0.75, 1.0):
            mean = nominal / speed * (1 + contention * load)
            for value in mean * rng.lognormal(0.0, noise, n):
                rows.append(
                    {"node": node, "req_type": req_type, "work_ms": nominal, "concurrency": 1, "load": load, "service_ms": value}
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ fit


def test_fit_recovers_the_known_service_law():
    raw = synthetic_raw(speed=0.5, contention=0.4)
    rtt = pd.DataFrame({"node": "fogA", "rtt_ms": np.random.default_rng(1).normal(30.0, 3.0, 200)})
    calibration, suggestions, warns = build_calibration(raw, rtt, PROFILES, NOMINAL, source="synthetique")
    assert not warns and calibration["version"] == 1 and calibration["source"] == "synthetique"
    entry = next(e for e in calibration["entries"] if e["node"] == "fogA" and e["req_type"] == "medium")
    assert entry["loads"] == [0.0, 0.25, 0.5, 0.75, 1.0] and min(entry["n"]) == 40
    assert entry["mean_ms"][0] == pytest.approx(16 / 0.5, rel=0.03)
    assert entry["mean_ms"][-1] == pytest.approx(16 / 0.5 * 1.4, rel=0.03)
    assert max(entry["cv"]) < 0.08
    s = suggestions["fogA"]
    assert s["speed"] == pytest.approx(0.5, rel=0.03) and s["contention"] == pytest.approx(0.4, abs=0.04)
    assert s["rtt_ms"] == pytest.approx(30.0, abs=1.0) and s["rtt_sigma"] == pytest.approx(0.1, abs=0.02)


def test_fit_warns_about_nodes_missing_from_the_configuration():
    _, _, warns = build_calibration(synthetic_raw(node="fogZ"), None, PROFILES, NOMINAL)
    assert any("absents de la configuration" in w and "fogZ" in w for w in warns)


def test_fit_normalizes_measurements_taken_with_other_request_sizes():
    raw = synthetic_raw(speed=1.0, contention=0.0, noise=0.01)
    raw["work_ms"] = raw["work_ms"] * 2.0
    raw["service_ms"] = raw["service_ms"] * 2.0  # requetes deux fois plus grosses, donc deux fois plus longues
    entries, _ = fit_service_table(raw, NOMINAL)
    assert next(e for e in entries if e["req_type"] == "light")["mean_ms"][0] == pytest.approx(8.0, rel=0.02)


def test_fit_skips_sparse_points_and_warns():
    raw = pd.concat([synthetic_raw(n=40, node="fogA"), synthetic_raw(node="fogY", n=3)])
    entries, warns = fit_service_table(raw, NOMINAL)
    assert {e["node"] for e in entries} == {"fogA"}
    assert any("fogY/light" in w and f">= {MIN_SAMPLES}" in w for w in warns)
    one_level = synthetic_raw(n=40)
    one_level = one_level[one_level["load"] == 0.0]
    entries, warns = fit_service_table(one_level, NOMINAL)
    assert len(entries) == 3 and any("un seul niveau" in w for w in warns)


def test_report_flags_assumptions_that_deviate_by_more_than_30_percent():
    suggestions = {"fogA": {"speed": 0.2, "contention": 0.31, "rtt_ms": 10.5}}
    report = render_report(suggestions, PROFILES, ["avertissement de test"])
    rows = {line.split("|")[2].strip(): line for line in report.splitlines() if line.startswith("| fogA")}
    assert "**A REVISER**" in rows["speed"] and "ok" in rows["contention"] and "ok" in rows["rtt_ms"]
    assert "hypotheses" in report and "avertissement de test" in report


# ------------------------------------------------------- mesure -> table -> simulateur


def test_measure_fit_roundtrip_recovers_node_speeds_and_feeds_the_simulator(tiny_requests, tmp_path):
    plan = CalibrationPlan(
        workers={"fogA": "http://a", "fogC": "http://c"},
        requests_file=tiny_requests,
        req_types=("light", "medium"),
        samples_per_level=12,
        warmup_requests=2,
        rtt_samples=10,
    )

    async def scenario():
        async with workers(emulate_speed=True) as clients:
            return await run_calibration(plan, clients, verbose=False)

    raw, rtt, meta = asyncio.run(scenario())
    out = save_measurements(raw, rtt, meta, tmp_path / "measures")
    assert {"raw.csv", "rtt.csv", "meta.json"} <= {p.name for p in out.iterdir()}
    assert set(raw["node"]) == {"fogA", "fogC"} and set(raw["concurrency"]) == {1, 2, 3}  # servers + 1 niveaux
    assert meta["workers"]["fogC"]["servers"] == 2 and meta["commit"]

    # La charge rapportee est celle de l'ARRIVEE : 0 avec un client, 1 quand tous les serveurs sont pris.
    fog_a = raw[raw["node"] == "fogA"]
    assert set(fog_a.loc[fog_a["concurrency"] == 1, "load"]) == {0.0}
    assert fog_a.loc[fog_a["concurrency"] == 2, "load"].max() == 1.0
    assert set(raw.loc[(raw["node"] == "fogC") & (raw["concurrency"] == 2), "load"]) <= {0.0, 0.5}

    calibration, suggestions, warns = build_calibration(
        raw, rtt, PROFILES, {"light": 8.0, "medium": 16.0}, source=str(out)
    )
    assert suggestions["fogA"]["speed"] == pytest.approx(0.5, rel=0.2)  # vitesse emulee du profil
    assert suggestions["fogC"]["speed"] == pytest.approx(1.0, rel=0.25)
    assert suggestions["fogA"]["speed"] < suggestions["fogC"]["speed"]
    assert suggestions["fogA"]["rtt_ms"] > 0

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    config = EnvConfig.from_yaml(
        "configs/env/eval.yaml",
        {"service_model": "calibrated", "calibration_file": str(path), "episode_requests": 300, "scenarios": ["low"]},
    )
    model = CalibratedServiceModel.from_json(path, config.requests)
    assert model.covers("fogA", "light") and not model.covers("fogB", "light")
    mean_a = model.mean_ms(PROFILES["fogA"], "light", 5.0, 0.0)  # fogA, requete legere nominale (5 ms)
    assert mean_a == pytest.approx(5.0 / suggestions["fogA"]["speed"], rel=0.4)
    rows = evaluate_policy(
        make_policy("greedy", {p.node_id: p for p in config.pool}), FogEnv(config), [1000],
        policy_name="greedy", policy_type="baseline",
    )
    assert np.isfinite(rows[0]["latency_mean_ms"]) and rows[0]["n_requests"] == 300  # le simulateur tourne sur la table


def test_rtt_probes_measure_the_injected_latency(tiny_requests):
    plan = CalibrationPlan(
        workers={"fogA": "http://a", "fogC": "http://c"},
        requests_file=tiny_requests,
        req_types=("light",),
        levels=[1],
        samples_per_level=5,
        warmup_requests=1,
        rtt_samples=12,
    )

    async def scenario():
        flat = {"fogA": {"rtt_sigma": 0.0}, "fogC": {"rtt_sigma": 0.0}}
        async with workers(overrides=flat, simulated_rtt_ms=40.0) as clients:
            return await run_calibration(plan, clients, verbose=False)

    raw, rtt, _ = asyncio.run(scenario())
    assert set(raw["concurrency"]) == {1}
    for node in ("fogA", "fogC"):
        samples = rtt.loc[rtt["node"] == node, "rtt_ms"]
        assert len(samples) == 12 and 38.0 <= samples.mean() <= 55.0


def test_plan_validation(tmp_path):
    path = tmp_path / "plan.yaml"
    path.write_text("workers: {fogA: 'http://a'}\nsamples_per_level: 7\nlevels: [1, 2]\n")
    plan = CalibrationPlan.from_yaml(path)
    assert plan.levels_for(4) == [1, 2] and plan.samples_per_level == 7
    assert CalibrationPlan(workers={}).levels_for(2) == [1, 2, 3]
    path.write_text("workers: {}\nsample: 3\n")
    with pytest.raises(ValueError, match="inconnues"):
        CalibrationPlan.from_yaml(path)
    plan = CalibrationPlan(workers={}, req_types=("gigantesque",))
    with pytest.raises(ValueError, match="inconnus"):
        asyncio.run(run_calibration(plan, {}, verbose=False))
    assert CalibrationPlan.from_yaml("configs/calibration/plan.yaml").req_types == ["light", "medium", "heavy"]


def test_calibrated_model_scales_with_the_nominal_work_used_during_calibration(tmp_path):
    """Si requests.yaml change apres la calibration, la table reste interpretee avec son propre travail nominal."""

    raw = synthetic_raw(speed=0.5, contention=0.0, noise=0.01)  # nominal calibre : light = 8 ms
    calibration, _, _ = build_calibration(raw, None, PROFILES, NOMINAL)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    config = EnvConfig.from_yaml("configs/env/eval.yaml")  # requests.yaml actuel : light = 5 ms
    model = CalibratedServiceModel.from_json(path, config.requests)
    assert model.mean_ms(PROFILES["fogA"], "light", 8.0, 0.0) == pytest.approx(16.0, rel=0.02)
    assert model.mean_ms(PROFILES["fogA"], "light", 4.0, 0.0) == pytest.approx(8.0, rel=0.02)  # proportionnel au travail
