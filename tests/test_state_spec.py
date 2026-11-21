import json
import math

import numpy as np
import pytest

from fogrl.state_spec import (
    DEFAULT_NORMALIZATION,
    NodeState,
    Normalization,
    RequestState,
    absent_node,
    check_contract,
    contract_metadata,
    encode_state,
    observation_dim,
    observation_feature_names,
    pad_nodes,
)

NODE_IDS = ["fogA", "fogB", "fogC", "cloud"]


def make_nodes(overrides=None):
    nodes = [
        NodeState("fogA", 100.0, 0.2, 10.0, 0.9),
        NodeState("fogB", 200.0, 0.5, 20.0, None),
        NodeState("fogC", 300.0, 0.8, 30.0, 0.4),
        NodeState("cloud", 50.0, 0.1, 120.0),
    ]
    for index, changes in (overrides or {}).items():
        fields = {**nodes[index].__dict__, **changes}
        nodes[index] = NodeState(**fields)
    return nodes


REQUEST = RequestState("medium", 150.0, 1000.0)


def test_observation_dim_matches_spec():
    assert observation_dim(4) == 22
    assert observation_dim(3) == 18
    with pytest.raises(ValueError):
        observation_dim(0)


def test_feature_names_match_vector_length_and_order():
    names = observation_feature_names(NODE_IDS)
    obs, _ = encode_state(make_nodes(), REQUEST, 50.0)
    assert len(names) == len(obs) == 22
    assert names[0] == "fogA.queue_ms"
    assert names[-1] == "arrival_rate_rps"


def test_encoding_values_and_dtype():
    obs, mask = encode_state(make_nodes(), REQUEST, 50.0)
    assert obs.dtype == np.float32
    assert mask.dtype == np.bool_ and mask.all()
    assert obs[0] == pytest.approx(0.1)  # queue 100/1000
    assert obs[3] == pytest.approx(0.9)  # batterie fogA
    assert obs[7] == pytest.approx(1.0)  # fogB sans batterie -> 1.0
    assert list(obs[16:19]) == [0.0, 1.0, 0.0]  # one-hot medium
    assert obs[19] == pytest.approx(0.5)  # work 150/300
    assert obs[21] == pytest.approx(0.5)  # debit 50/100


def test_values_stay_in_unit_interval_when_saturated():
    nodes = make_nodes({0: {"queue_ms": 1e6, "cpu_load": 3.0}})
    obs, _ = encode_state(nodes, RequestState("heavy", 1e5, 1e6), 1e4)
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    assert obs[0] == 1.0 and obs[1] == 1.0


def test_unavailable_node_is_masked_but_keeps_measurements():
    nodes = make_nodes({2: {"available": False}})
    obs, mask = encode_state(nodes, REQUEST, 10.0)
    assert list(mask) == [True, True, False, True]
    assert obs[8] == pytest.approx(0.3)  # file de fogC conservee


def test_no_available_node_raises():
    nodes = [NodeState(n.node_id, 0, 0, 0, available=False) for n in make_nodes()]
    with pytest.raises(ValueError, match="Aucun noeud"):
        encode_state(nodes, REQUEST, 10.0)


def test_duplicate_ids_and_unknown_type_raise():
    nodes = make_nodes()
    nodes[1] = NodeState("fogA", 0, 0, 0)
    with pytest.raises(ValueError):
        encode_state(nodes, REQUEST, 10.0)
    with pytest.raises(ValueError, match="inconnu"):
        encode_state(make_nodes(), RequestState("giant", 1, 1), 10.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0])
def test_corrupted_measurements_are_rejected(bad):
    with pytest.raises(ValueError):
        encode_state(make_nodes({0: {"queue_ms": bad}}), REQUEST, 10.0)
    with pytest.raises(ValueError):
        encode_state(make_nodes({1: {"cpu_load": bad}}), REQUEST, 10.0)
    with pytest.raises(ValueError):
        encode_state(make_nodes(), REQUEST, bad)


def test_normalization_validation():
    with pytest.raises(ValueError):
        Normalization(queue_ms=0)
    with pytest.raises(ValueError):
        Normalization(deadline_ms=math.nan)


def test_contract_roundtrip_through_json():
    saved = json.loads(json.dumps(contract_metadata(NODE_IDS)))
    check_contract(saved, NODE_IDS)  # ne leve rien


def test_contract_detects_node_order_and_normalization_changes():
    saved = json.loads(json.dumps(contract_metadata(NODE_IDS)))
    with pytest.raises(ValueError, match="node_ids"):
        check_contract(saved, ["fogB", "fogA", "fogC", "cloud"])
    with pytest.raises(ValueError, match="normalization"):
        check_contract(saved, NODE_IDS, Normalization(queue_ms=2000.0))
    tampered = {**saved, "observation_dim": 18}
    with pytest.raises(ValueError, match="observation_dim"):
        check_contract(tampered, NODE_IDS)
    with pytest.raises(ValueError, match="spec_version"):
        check_contract({**saved, "spec_version": 1}, NODE_IDS)


def test_absent_node_is_saturated_dead_and_masked():
    node = absent_node("fogD")
    assert not node.available
    obs, mask = encode_state([make_nodes()[0], node], REQUEST, 10.0)
    assert list(mask) == [True, False]
    assert list(obs[4:8]) == [1.0, 1.0, 1.0, 0.0]  # file, cpu, rtt satures ; batterie vide


def test_pad_nodes_orders_by_slot_and_fills_missing():
    nodes = make_nodes()  # fogA, fogB, fogC, cloud
    slots = ["fogA", "fogB", "fogC", "fogD", "fogE", "cloud"]
    padded = pad_nodes(nodes, slots)
    assert [n.node_id for n in padded] == slots
    assert [n.available for n in padded] == [True, True, True, False, False, True]
    obs, mask = encode_state(padded, REQUEST, 10.0)
    assert len(obs) == observation_dim(6) == 30
    assert list(mask) == [True, True, True, False, False, True]
    with pytest.raises(ValueError, match="hors des emplacements"):
        pad_nodes(nodes, ["fogA", "fogB"])


def test_in_flight_is_not_part_of_the_observation():
    base, _ = encode_state(make_nodes(), REQUEST, 50.0)
    busy_nodes = make_nodes({0: {"in_flight": 42}})
    busy, _ = encode_state(busy_nodes, REQUEST, 50.0)
    assert np.array_equal(base, busy)
