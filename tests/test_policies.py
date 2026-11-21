import numpy as np
import pytest

from fogrl.config import load_nodes
from fogrl.policies import BASELINES, make_baseline
from fogrl.policies.baselines import GreedyPolicy, StaticThresholdPolicy
from fogrl.state_spec import DecisionContext, NodeState, RequestState

PROFILES = {p.node_id: p for p in load_nodes("configs/nodes/pool6.yaml")[0]}
CTX = DecisionContext(now_s=1.0, arrival_rate_rps=50.0)
REQ = RequestState("medium", 40.0, 300.0)


def nodes(**per_node):
    base = {
        "fogA": dict(queue_ms=0, cpu_load=0.1, net_rtt_ms=10, battery_fraction=0.9, in_flight=0),
        "fogB": dict(queue_ms=0, cpu_load=0.1, net_rtt_ms=20, battery_fraction=0.9, in_flight=0),
        "fogC": dict(queue_ms=0, cpu_load=0.1, net_rtt_ms=30, battery_fraction=None, in_flight=0),
        "cloud": dict(queue_ms=0, cpu_load=0.0, net_rtt_ms=80, battery_fraction=None, in_flight=0),
    }
    result = []
    for node_id, values in base.items():
        result.append(NodeState(node_id, **{**values, **per_node.get(node_id, {})}))
    return result


@pytest.mark.parametrize("name", BASELINES)
def test_every_baseline_returns_valid_index_and_skips_unavailable(name):
    policy = make_baseline(name, PROFILES)
    policy.reset(0)
    states = nodes(fogA={"available": False}, fogC={"available": False})
    for _ in range(40):
        choice = policy.select(states, REQ, CTX)
        assert states[choice].available


@pytest.mark.parametrize("name", BASELINES)
def test_baselines_raise_when_nothing_is_available(name):
    policy = make_baseline(name, PROFILES)
    states = [NodeState(n.node_id, 0, 0, 0, available=False) for n in nodes()]
    with pytest.raises(ValueError):
        policy.select(states, REQ, CTX)


def test_unknown_baseline():
    with pytest.raises(ValueError, match="inconnue"):
        make_baseline("magic", PROFILES)


def test_round_robin_cycles_through_available_nodes():
    policy = make_baseline("round_robin", PROFILES)
    states = nodes(fogB={"available": False})
    picks = [policy.select(states, REQ, CTX) for _ in range(6)]
    assert picks == [0, 2, 3, 0, 2, 3]


def test_random_is_reproducible_with_seed_and_covers_nodes():
    policy = make_baseline("random", PROFILES)
    policy.reset(1)
    a = [policy.select(nodes(), REQ, CTX) for _ in range(200)]
    policy.reset(1)
    b = [policy.select(nodes(), REQ, CTX) for _ in range(200)]
    assert a == b and set(a) == {0, 1, 2, 3}


def test_least_connections_picks_fewest_in_flight_then_shortest_queue():
    policy = make_baseline("least_connections", PROFILES)
    states = nodes(fogA={"in_flight": 5}, fogB={"in_flight": 1, "queue_ms": 50},
                   fogC={"in_flight": 1, "queue_ms": 10}, cloud={"in_flight": 3})
    assert policy.select(states, REQ, CTX) == 2


def test_static_threshold_prefers_nearest_fog_then_cloud():
    policy = StaticThresholdPolicy(PROFILES)
    assert policy.select(nodes(), REQ, CTX) == 0  # fogA, le plus proche
    assert policy.select(nodes(fogA={"cpu_load": 0.9}), REQ, CTX) == 1
    assert policy.select(nodes(fogA={"queue_ms": 500}, fogB={"cpu_load": 0.95}), REQ, CTX) == 2
    assert policy.select(nodes(fogA={"battery_fraction": 0.05}), REQ, CTX) == 1  # batterie faible
    saturated = nodes(fogA={"cpu_load": 1}, fogB={"cpu_load": 1}, fogC={"cpu_load": 1})
    assert policy.select(saturated, REQ, CTX) == 3  # cloud


def test_greedy_minimizes_estimated_completion_time():
    policy = GreedyPolicy(PROFILES)
    heavy = RequestState("heavy", 150.0, 1000.0)
    # requete lourde : fogB (vitesse 1, RTT 20) bat fogA (vitesse 0.5) et le cloud (RTT 80)
    assert policy.select(nodes(), heavy, CTX) == 1
    # requete legere : fogA (RTT 10) bat les autres malgre sa vitesse 0.5
    assert policy.select(nodes(), RequestState("light", 5.0, 100.0), CTX) == 0
    # la file compte : fogB encombre -> le cloud (180 ms) passe devant fogC (185 ms)
    assert policy.select(nodes(fogB={"queue_ms": 400}), heavy, CTX) == 3
    # estimation exacte : RTT + file + travail / vitesse * (1 + contention * charge)
    fogA = nodes()[0]
    expected = 10 + 0 + 150 / 0.5 * (1 + 0.3 * 0.1)
    assert policy.estimated_completion_ms(fogA, heavy) == pytest.approx(expected)


def test_greedy_beats_random_and_round_robin_in_simulation():
    from fogrl.evaluation.evaluate import run_episode
    from fogrl.sim.env import FogEnv

    env = FogEnv("configs/env/n1.yaml", {"episode_requests": 1500})
    latency = {
        name: np.mean([run_episode(env, make_baseline(name, PROFILES), seed)["latency_mean_ms"] for seed in (1000, 1001)])
        for name in ("greedy", "round_robin", "random")
    }
    assert latency["greedy"] < latency["round_robin"]
    assert latency["greedy"] < latency["random"]
