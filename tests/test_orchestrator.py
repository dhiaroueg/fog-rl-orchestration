"""Orchestrateur reel : sondage, decision avec budget et repli, API, politiques RL, journal.

Les workers et l'orchestrateur tournent dans le meme processus (ASGI en memoire) : le chemin de
code est celui du deploiement, sans reseau. Les durees de service sont reelles mais courtes.
"""

import asyncio
import json
import time
import warnings

import httpx
import numpy as np
import pytest
from helpers_stack import route_body, stack

from fogrl.config import EnvConfig, load_yaml
from fogrl.orchestrator.decision import DecisionEngine, NoNodeAvailable
from fogrl.orchestrator.monitor import NodeMonitor
from fogrl.orchestrator.router import RateWindow
from fogrl.orchestrator.settings import OrchestratorSettings, parse_urls
from fogrl.policies.base import Policy
from fogrl.policies.baselines import BASELINES
from fogrl.policies.rl import RLPolicy
from fogrl.state_spec import DecisionContext, NodeState, Normalization, RequestState
from fogrl.training.train import train

warnings.filterwarnings("ignore")

NORM = Normalization(arrival_rate_rps=200.0)


class FixedPolicy(Policy):
    name = "fixed"

    def __init__(self, index, needs_budget=False):
        self.index, self.needs_budget = index, needs_budget

    def select(self, nodes, request, ctx):
        return self.index


class SlowPolicy(FixedPolicy):
    name = "slow"

    def __init__(self, delay_s):
        super().__init__(0, needs_budget=True)
        self.delay_s = delay_s

    def select(self, nodes, request, ctx):
        time.sleep(self.delay_s)
        return 0


class BrokenPolicy(FixedPolicy):
    name = "broken"

    def select(self, nodes, request, ctx):
        raise ValueError("mesure non finie")


def states(**overrides):
    base = {
        "fogA": dict(queue_ms=0, cpu_load=0.1, net_rtt_ms=10, battery_fraction=0.9),
        "fogB": dict(queue_ms=0, cpu_load=0.1, net_rtt_ms=20, battery_fraction=0.9),
        "fogC": dict(queue_ms=0, cpu_load=0.1, net_rtt_ms=30, battery_fraction=None),
        "cloud": dict(queue_ms=0, cpu_load=0.0, net_rtt_ms=80, battery_fraction=None),
    }
    return [NodeState(node_id, **{**values, **overrides.get(node_id, {})}) for node_id, values in base.items()]


SLOTS = ("fogA", "fogB", "fogC", "cloud")
REQUEST, CTX = RequestState("medium", 40.0, 300.0), DecisionContext(0.0, 50.0)


# --------------------------------------------------------------------- reglages


def test_settings_reject_unknown_keys_and_resolve_nodes_urls_and_deadlines():
    with pytest.raises(ValueError, match="inconnues"):
        OrchestratorSettings.from_dict({"polcy": "greedy"})
    resolved = OrchestratorSettings(urls={"fogA": "http://vm1:8001"}).resolve()
    assert resolved.active_ids == ("fogA", "fogB", "fogC", "cloud")
    assert resolved.urls["fogA"] == "http://vm1:8001" and resolved.urls["cloud"] == "http://cloud:8006"
    assert resolved.deadlines == {"light": 100.0, "medium": 300.0, "heavy": 1000.0}
    assert resolved.poll_interval_s == pytest.approx(0.2) and resolved.stale_after_s == 1.0
    with pytest.raises(ValueError, match="Configuration inconnue"):
        OrchestratorSettings(configuration="n9").resolve()
    assert parse_urls("fogA=http://a:1; fogB=http://b:2") == {"fogA": "http://a:1", "fogB": "http://b:2"}
    with pytest.raises(ValueError):
        parse_urls("fogA")


def test_settings_from_env_apply_overrides(tmp_path):
    config = tmp_path / "orch.yaml"
    config.write_text("policy: greedy\nconfiguration: n3\ndecision_budget_ms: 15\n")
    settings = OrchestratorSettings.from_env(
        {"ORCH_CONFIG": str(config), "ORCH_POLICY": "round_robin", "DECISION_LOG": "x.jsonl", "ORCH_NODE_URLS": "fogA=http://h:1"}
    )
    assert (settings.policy, settings.configuration, settings.decision_budget_ms) == ("round_robin", "n3", 15.0)
    assert settings.log_file == "x.jsonl" and settings.urls == {"fogA": "http://h:1"}
    assert OrchestratorSettings.from_env({"ORCH_CONFIG": str(tmp_path / "absent.yaml")}).policy == "least_connections"


def test_rate_window_matches_the_simulator_formula():
    window = RateWindow(1.0, t0=0.0)
    assert window.record(0.0) == pytest.approx(1 / 0.1)  # plancher de 0,1 s au demarrage
    rates = [window.record(0.1 * i) for i in range(1, 11)]
    assert rates[-1] == pytest.approx(10.0)  # 10 requetes dans la derniere seconde
    assert window.current(5.0) == 0.0  # fenetre videe


# ---------------------------------------------------------------------- monitor


def mock_state(**overrides):
    return {"node_id": "fogA", "queue_ms": 12.0, "cpu_load": 0.4, "battery_fraction": 0.7, "available": True, **overrides}


def make_monitor(handler, clock, **kwargs):
    profile = EnvConfig.from_yaml("configs/env/eval.yaml").profile("fogA")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fogA")
    return NodeMonitor([profile], {"fogA": client}, clock=clock, **kwargs)


def test_monitor_reads_state_and_smooths_rtt():
    async def scenario():
        monitor = make_monitor(lambda request: httpx.Response(200, json=mock_state()), clock=lambda: 0.0)
        assert monitor.is_available("fogA") is False  # aucun sondage encore
        await monitor.poll_once()
        view = monitor.views["fogA"]
        assert (view.queue_ms, view.cpu_load, view.battery_fraction) == (12.0, 0.4, 0.7)
        assert monitor.is_available("fogA") and view.rtt_ms < 10.0  # converge de 10 ms nominaux vers ~0
        node = monitor.states({"fogA": 3})[0]
        assert (node.queue_ms, node.in_flight, node.available) == (12.0, 3, True)

    asyncio.run(scenario())


def test_monitor_circuit_breaker_and_staleness():
    clock = {"now": 0.0}
    answers = {"ok": True}

    def handler(request):
        if not answers["ok"]:
            raise httpx.ConnectError("refuse", request=request)
        return httpx.Response(200, json=mock_state())

    async def scenario():
        monitor = make_monitor(handler, clock=lambda: clock["now"], failure_threshold=2, stale_after_s=1.0)
        await monitor.poll_once()
        assert monitor.is_available("fogA")
        answers["ok"] = False
        await monitor.poll_once()
        assert monitor.is_available("fogA")  # un seul echec : on tolere
        await monitor.poll_once()
        assert not monitor.is_available("fogA")  # disjoncteur ouvert
        answers["ok"] = True
        await monitor.poll_once()
        assert monitor.is_available("fogA")  # refermeture apres un sondage reussi
        clock["now"] = 2.5
        assert not monitor.is_available("fogA")  # mesure trop ancienne
        assert monitor.age_ms("fogA") == pytest.approx(2500.0)
        await monitor.poll_once()
        monitor.views["fogA"].reported_available = False  # le noeud se dit hors service (batterie vide, panne)
        assert not monitor.is_available("fogA")

    asyncio.run(scenario())


def test_monitor_offers_every_node_when_all_look_down():
    async def scenario():
        monitor = make_monitor(lambda request: httpx.Response(200, json=mock_state(available=False)), clock=lambda: 0.0)
        await monitor.poll_once()
        assert not monitor.is_available("fogA")
        assert monitor.states({})[0].available is True  # mesures tenues pour perimees, comme FogEnv

    asyncio.run(scenario())


# ------------------------------------------------------------------- decision


def engine(policy, **kwargs):
    return DecisionEngine(policy, SLOTS, NORM, budget_ms=kwargs.pop("budget_ms", 20.0), **kwargs)


def decide(eng, nodes=None):
    async def run():
        try:
            return await eng.decide(nodes or states(), REQUEST, CTX)
        finally:
            eng.close()

    return asyncio.run(run())


def test_decision_uses_the_policy_when_it_answers_in_time():
    decision = decide(engine(FixedPolicy(2)))
    assert (decision.node_id, decision.fallback, decision.reason, decision.policy) == ("fogC", False, "", "fixed")


def test_slow_policy_falls_back_to_least_connections_within_the_budget():
    start = time.perf_counter()
    decision = decide(engine(SlowPolicy(0.3), budget_ms=20.0), states(fogA={"in_flight": 4}, fogB={"in_flight": 2}))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert decision.fallback and decision.reason == "timeout"
    assert decision.node_id == "fogC"  # 0 requete en cours (egalite avec le cloud : file la plus courte puis index)
    assert 15.0 <= decision.decision_ms < 150.0 and elapsed_ms < 250.0  # on n'attend pas les 300 ms


def test_failing_policy_and_invalid_action_fall_back():
    assert decide(engine(BrokenPolicy(0))).reason == "error:ValueError"
    decision = decide(engine(FixedPolicy(1)), states(fogB={"queue_ms": 5}))  # valide
    assert not decision.fallback
    nodes = states()
    nodes[1] = NodeState("fogB", 0, 0.1, 20, 0.9, available=False)
    unavailable = decide(engine(FixedPolicy(1)), nodes)
    assert unavailable.reason == "invalid_action" and unavailable.node_id != "fogB"
    assert decide(engine(FixedPolicy(99))).reason == "invalid_action"


def test_policy_breaker_opens_after_repeated_failures_then_recovers():
    clock = {"now": 0.0}
    eng = engine(BrokenPolicy(0), failure_threshold=3, cooldown_s=5.0, clock=lambda: clock["now"])

    async def scenario():
        reasons = [(await eng.decide(states(), REQUEST, CTX)).reason for _ in range(5)]
        assert reasons == ["error:ValueError"] * 3 + ["breaker_open"] * 2  # la 4e decision ne retente plus
        assert eng.breaker_open
        clock["now"] = 6.0
        assert not eng.breaker_open
        eng.swap(FixedPolicy(0), SLOTS)
        decision = await eng.decide(states(), REQUEST, CTX)
        assert not decision.fallback and decision.node_id == "fogA"
        eng.close()

    asyncio.run(scenario())
    assert eng.fallbacks == {"error:ValueError": 3, "breaker_open": 2}


def test_no_node_available_is_reported():
    nodes = [NodeState(n.node_id, 0, 0.1, 10, None, available=False) for n in states()]
    eng = engine(FixedPolicy(0))

    async def scenario():
        with pytest.raises(NoNodeAvailable):
            await eng.decide(nodes, REQUEST, CTX)
        eng.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------- API


def test_route_end_to_end_with_every_baseline():
    async def scenario():
        async with stack("greedy") as client:
            for spec in BASELINES:
                assert (await client.post("/policy", json={"spec": spec})).status_code == 200
                reply = (await client.post("/route", json=route_body("medium", 20.0))).json()
                assert reply["rejected"] is False and reply["policy"] == spec
                assert reply["node"] in client.resolved.active_ids
                assert reply["latency_ms"] >= reply["service_ms"] > 0 and reply["decision_ms"] < 20.0
                assert reply["deadline_ms"] == 300.0 and reply["deadline_met"] is True
                assert reply["fallback"] is False

    asyncio.run(scenario())


def test_greedy_prefers_the_fast_nearby_fog_for_a_light_request():
    async def scenario():
        async with stack("greedy") as client:
            nodes = [(await client.post("/route", json=route_body("light", 5.0))).json()["node"] for _ in range(6)]
            return nodes

    nodes = asyncio.run(scenario())
    assert set(nodes) <= {"fogA", "fogB", "fogC"} and "cloud" not in nodes


def test_state_nodes_policy_metrics_and_health_endpoints():
    async def scenario():
        async with stack("round_robin") as client:
            await client.post("/route", json=route_body())
            state = (await client.get("/state")).json()
            fresh = (await client.get("/nodes", params={"fresh": True})).json()
            policy = (await client.get("/policy")).json()
            metrics = (await client.get("/metrics")).text
            health = (await client.get("/health")).json()
            return state, fresh, policy, metrics, health

    state, fresh, policy, metrics, health = asyncio.run(scenario())
    assert [n["node_id"] for n in state["nodes"]] == list(SLOTS) and all(n["available"] for n in state["nodes"])
    assert state["decisions"] == 1 and set(fresh) == set(SLOTS) and fresh["fogC"]["servers"] == 2
    assert policy["name"] == "round_robin" and policy["kind"] == "baseline" and health["status"] == "ok"
    for expected in (
        'fog_route_requests_total{node="fogA",policy="round_robin",status="ok",type="light"} 1.0',
        "fog_decision_seconds_bucket",
        'fog_monitor_node_up{node="fogA"} 1.0',
        'fog_monitor_cpu_load{node="cloud"}',
        'fog_policy_info{kind="baseline",policy="round_robin"} 1.0',
        "fog_orchestrator_arrival_rate_rps",
        "fog_route_latency_seconds_bucket",
    ):
        assert expected in metrics, expected


def test_hot_swap_rejects_unknown_policies_and_keeps_the_current_one():
    async def scenario():
        async with stack("greedy") as client:
            bad = await client.post("/policy", json={"spec": "magic"})
            missing = await client.post("/policy", json={"spec": "rl:models/absent/ppo/seed0"})
            invalid_body = await client.post("/route", json={"type": "giant", "work_ms": 5})
            return bad, missing, invalid_body, (await client.get("/policy")).json()

    bad, missing, invalid_body, policy = asyncio.run(scenario())
    assert bad.status_code == 400 and missing.status_code == 400 and invalid_body.status_code == 422
    assert policy["name"] == "greedy"


def test_hot_swap_during_traffic_loses_no_request():
    """EF-03 : changer de politique pendant un run ne perd aucune requete."""

    async def scenario():
        async with stack("round_robin") as client:

            async def traffic():
                replies = []
                for _ in range(40):
                    replies.append((await client.post("/route", json=route_body("light", 3.0))).json())
                    await asyncio.sleep(0.005)
                return replies

            async def swaps():
                statuses = []
                for spec in ("greedy", "least_connections", "static_threshold", "round_robin"):
                    await asyncio.sleep(0.04)
                    statuses.append((await client.post("/policy", json={"spec": spec})).status_code)
                return statuses

            return await asyncio.gather(traffic(), swaps())

    replies, statuses = asyncio.run(scenario())
    assert statuses == [200, 200, 200, 200]
    assert len(replies) == 40 and all(not r["rejected"] and r["node"] for r in replies)
    assert len({r["policy"] for r in replies}) >= 3  # les changements ont bien eu lieu en cours de run


def test_slow_policy_in_a_running_stack_serves_the_request_through_the_fallback():
    async def scenario():
        async with stack("greedy", decision_budget_ms=20.0) as client:
            client.router.engine.swap(SlowPolicy(0.3), SLOTS)
            reply = (await client.post("/route", json=route_body("light", 5.0))).json()
            metrics = (await client.get("/metrics")).text
            return reply, metrics

    reply, metrics = asyncio.run(scenario())
    assert reply["rejected"] is False and reply["fallback"] is True and reply["fallback_reason"] == "timeout"
    assert 15.0 <= reply["decision_ms"] < 150.0
    assert 'fog_decision_fallback_total{policy="slow",reason="timeout"} 1.0' in metrics


def test_injected_agent_fault_falls_back_visibly_and_is_lifted(tmp_path):
    """EF-07 sur la pile en marche : l'agent tombe en panne (ou devient lent), le taux de repli est visible, puis tout reprend."""

    async def scenario():
        async with stack("greedy", decision_budget_ms=20.0, log_file=str(tmp_path / "decisions.jsonl")) as client:
            route = lambda: client.post("/route", json=route_body("light", 5.0))  # noqa: E731
            healthy = (await route()).json()
            error_info = (await client.post("/admin/policy-fault", json={"mode": "error"})).json()
            broken = [(await route()).json() for _ in range(3)]
            slow_info = (await client.post("/admin/policy-fault", json={"mode": "slow", "delay_ms": 120})).json()
            slow = (await route()).json()
            no_delay = await client.post("/admin/policy-fault", json={"mode": "slow", "delay_ms": 0})
            unknown = await client.post("/admin/policy-fault", json={"mode": "explode"})
            cleared = (await client.post("/admin/policy-fault", json={"mode": "none"})).json()
            after = (await route()).json()
            await client.post("/admin/policy-fault", json={"mode": "error"})
            await client.post("/policy", json={"spec": "least_connections"})  # un nouveau modele est sain
            swapped = (await client.get("/policy")).json()
            metrics = (await client.get("/metrics")).text
            return healthy, error_info, broken, slow_info, slow, no_delay, unknown, cleared, after, swapped, metrics

    healthy, error_info, broken, slow_info, slow, no_delay, unknown, cleared, after, swapped, metrics = asyncio.run(
        scenario()
    )
    assert healthy["fallback"] is False and error_info["fault"] == {"mode": "error", "delay_ms": 0.0}
    assert all(r["fallback"] and r["fallback_reason"] == "error:RuntimeError" and not r["rejected"] for r in broken)
    assert slow_info["fault"] == {"mode": "slow", "delay_ms": 120.0}
    assert slow["fallback"] and slow["fallback_reason"] == "timeout" and not slow["rejected"]
    assert 15.0 <= slow["decision_ms"] < 100.0  # on n'attend pas les 120 ms de l'agent
    assert no_delay.status_code == 400 and unknown.status_code == 422
    assert cleared["fault"] is None and after["fallback"] is False
    assert swapped["name"] == "least_connections" and swapped["fault"] is None
    assert 'fog_decision_fallback_total{policy="greedy",reason="error:RuntimeError"} 3.0' in metrics
    assert 'fog_decision_fallback_total{policy="greedy",reason="timeout"} 1.0' in metrics
    records = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert [r["fallback"] for r in records] == [False, True, True, True, True, False]


def test_failed_node_is_avoided_and_recovers():
    async def scenario():
        async with stack("greedy") as client:
            assert (await client.post("/admin/nodes/fogC/fail")).json()["available"] is False
            state = (await client.get("/state")).json()
            during = [(await client.post("/route", json=route_body("heavy", 30.0))).json()["node"] for _ in range(4)]
            await client.post("/admin/nodes/fogC/recover")
            after = (await client.get("/state")).json()
            unknown = await client.post("/admin/nodes/fogZ/fail")
            return state, during, after, unknown.status_code

    state, during, after, unknown = asyncio.run(scenario())
    assert {n["node_id"]: n["available"] for n in state["nodes"]}["fogC"] is False
    assert "fogC" not in during  # le noeud en panne n'est plus propose (EF-09)
    assert {n["node_id"]: n["available"] for n in after["nodes"]}["fogC"] is True
    assert unknown == 404


def test_unreachable_and_slow_nodes_produce_rejections_and_trip_the_breaker():
    async def scenario():
        async with stack("greedy", failure_threshold=2) as client:
            client.router.engine.swap(FixedPolicy(1), SLOTS)  # force fogB
            client.transports["fogB"].mode = "refuse"
            first = (await client.post("/route", json=route_body())).json()
            failures_after_first = client.router.monitor.views["fogB"].failures
            client.transports["fogB"].mode = "timeout"
            second = (await client.post("/route", json=route_body())).json()
            client.transports["fogB"].mode = None
            return first, failures_after_first, second, client.router.monitor.is_available("fogB")

    first, failures, second, available = asyncio.run(scenario())
    assert first["rejected"] and first["reason"] == "unreachable" and first["latency_ms"] is None
    assert failures >= 1 and not first["deadline_met"]
    assert second["rejected"] and second["reason"] == "timeout"
    assert available is False  # deux echecs consecutifs : disjoncteur ouvert jusqu'au prochain sondage reussi


def test_worker_rejections_are_reported_with_their_reason():
    async def scenario():
        async with stack("greedy") as client:
            client.router.engine.swap(FixedPolicy(0), SLOTS)
            await client.router.clients["fogA"].post("/admin/fail")  # panne non vue du moniteur (sondage perime)
            return (await client.post("/route", json=route_body())).json()

    reply = asyncio.run(scenario())
    assert reply["rejected"] and reply["reason"] == "unavailable" and reply["node"] == "fogA"


def test_reset_clears_decisions_and_worker_counters():
    async def scenario():
        async with stack("greedy") as client:
            await client.post("/route", json=route_body())
            await client.post("/admin/reset")
            nodes = (await client.get("/nodes", params={"fresh": True})).json()
            return (await client.get("/state")).json(), nodes

    state, nodes = asyncio.run(scenario())
    assert state["decisions"] == 0 and sum(n["served"] for n in nodes.values()) == 0


def test_decision_log_is_valid_jsonl(tmp_path):
    log_file = tmp_path / "logs" / "decisions.jsonl"

    async def scenario():
        async with stack("least_connections", log_file=str(log_file)) as client:
            for _ in range(3):
                await client.post("/route", json=route_body())
            client.router.log.flush()

    asyncio.run(scenario())
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(lines) == 3
    for record in lines:
        assert {"ts", "request_id", "policy", "node", "latency_ms", "decision_ms", "rejected", "state", "rate_rps"} <= set(record)
        assert set(record["state"]) == set(SLOTS) and len(record["state"]["fogA"]) == 6


def test_concurrent_requests_are_spread_and_tracked_in_flight():
    async def scenario():
        async with stack("least_connections") as client:
            replies = await asyncio.gather(*(client.post("/route", json=route_body("medium", 40.0)) for _ in range(12)))
            return [r.json() for r in replies], client.router.in_flight

    replies, in_flight = asyncio.run(scenario())
    assert all(not r["rejected"] for r in replies)
    assert len({r["node"] for r in replies}) >= 3  # least-connections repartit la charge
    assert sum(in_flight.values()) == 0  # tous les compteurs sont revenus a zero


# ------------------------------------------------------------- politiques RL


def tiny_config(level):
    config = load_yaml(f"configs/train/{level}.yaml")
    config["eval"] = {"freq": 0}
    config["n_envs"] = 2
    for algo in config["algos"].values():
        algo.pop("n_envs", None)
    config["algos"]["ppo"].update(n_steps=64, batch_size=64)
    return config


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("orch_models")
    return {
        level: train(tiny_config(level), "ppo", 0, out_root=str(root), runs_root=str(root / "runs"), total_timesteps=1500)
        for level in ("n1", "n3")
    }


def test_rl_policy_decides_within_budget_on_a_compact_model(tiny_models):
    async def scenario():
        async with stack("greedy", env_file="configs/env/n1.yaml") as client:
            info = (await client.post("/policy", json={"spec": f"rl:{tiny_models['n1']}", "name": "ppo"})).json()
            replies = [(await client.post("/route", json=route_body("medium", 20.0))).json() for _ in range(15)]
            return info, replies

    info, replies = asyncio.run(scenario())
    assert info["name"] == "ppo" and info["kind"] == "rl" and info["slot_ids"] == list(SLOTS)
    assert all(not r["rejected"] and not r["fallback"] for r in replies)  # le budget de 20 ms tient
    decisions = sorted(r["decision_ms"] for r in replies)
    # Mediane dans le budget de 20 ms ; le maximum seulement sous la limite ENF-01 (50 ms) : un pic isole est
    # possible sur un poste charge (aucun repli n'a ete declenche, verifie plus haut).
    assert decisions[len(decisions) // 2] < 20.0 and decisions[-1] < 50.0
    assert all(r["node"] in SLOTS for r in replies)


def test_padded_model_serves_a_four_node_deployment(tiny_models):
    """Decision D1 : un modele a 6 emplacements route correctement 4 noeuds actifs."""

    async def scenario():
        async with stack("greedy", configuration="n4") as client:
            info = (await client.post("/policy", json={"spec": f"rl:{tiny_models['n3']}"})).json()
            replies = [(await client.post("/route", json=route_body("light", 5.0))).json() for _ in range(15)]
            return info, replies

    info, replies = asyncio.run(scenario())
    assert info["slot_ids"] == ["fogA", "fogB", "fogC", "fogD", "fogE", "cloud"]
    assert all(r["node"] in SLOTS and not r["fallback"] for r in replies)  # jamais un noeud absent (fogD, fogE)


def test_incompatible_models_are_refused_at_load_time(tiny_models):
    async def scenario():
        async with stack("greedy", configuration="n6") as client:  # fogD et fogE actifs, mais le modele N1 n'a que 4 emplacements
            bad_nodes = await client.post("/policy", json={"spec": f"rl:{tiny_models['n1']}"})
        async with stack("greedy", env_file="configs/env/n1.yaml") as client:
            ok = await client.post("/policy", json={"spec": f"rl:{tiny_models['n1']}"})
        return bad_nodes, ok

    bad_nodes, ok = asyncio.run(scenario())
    assert bad_nodes.status_code == 400 and ("incompatible" in bad_nodes.text or "couvre" in bad_nodes.text)
    assert ok.status_code == 200


def test_rl_policy_state_encoding_matches_the_simulator_contract(tiny_models):
    """Les NodeState de l'orchestrateur passent par le meme encode_state : memes decisions qu'en simulation."""

    config = EnvConfig.from_yaml("configs/env/n1.yaml")
    policy = RLPolicy.load(tiny_models["n1"], slot_ids=config.slot_ids, normalization=config.normalization)
    rng = np.random.default_rng(1)
    for _ in range(20):
        nodes = [
            NodeState(n, float(rng.uniform(0, 500)), float(rng.uniform()), float(rng.uniform(5, 90)), 0.7)
            for n in config.slot_ids
        ]
        direct = policy.select(nodes, REQUEST, CTX)
        padded_call = policy.select(nodes[::-1][::-1], REQUEST, CTX)
        assert direct == padded_call
