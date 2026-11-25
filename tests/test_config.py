import pytest

from fogrl.config import EnvConfig, NodeProfile, capacity_rps, load_nodes, load_requests


def test_pool_loads_with_expected_configurations():
    profiles, configurations = load_nodes("configs/nodes/pool6.yaml")
    ids = [p.node_id for p in profiles]
    assert ids == ["fogA", "fogB", "fogC", "fogD", "fogE", "cloud"]
    assert profiles[-1].is_cloud and profiles[-1].cost_per_request > 0
    assert configurations["n4"] == ("fogA", "fogB", "fogC", "cloud")
    # fogA et fogB ont une batterie, fogC et le cloud sont sur secteur (cahier 3.4)
    battery = {p.node_id: p.battery is not None for p in profiles}
    assert battery == {
        "fogA": True, "fogB": True, "fogC": False, "fogD": False, "fogE": False, "cloud": False,
    }


def test_energy_profiles_match_cahier():
    profiles = {p.node_id: p for p in load_nodes("configs/nodes/pool6.yaml")[0]}
    assert (profiles["fogA"].p_idle_w, profiles["fogA"].p_max_w) == (2.7, 6.4)
    assert (profiles["fogB"].p_idle_w, profiles["fogB"].p_max_w) == (5.0, 15.0)
    assert (profiles["fogC"].p_idle_w, profiles["fogC"].p_max_w) == (10.0, 35.0)


def test_request_profiles_match_cahier():
    requests = {r.name: r for r in load_requests("configs/requests.yaml")}
    assert (requests["light"].work_ms, requests["light"].deadline_ms) == (5, 100)
    assert (requests["medium"].work_ms, requests["medium"].deadline_ms) == (40, 300)
    assert (requests["heavy"].work_ms, requests["heavy"].deadline_ms) == (150, 1000)


def test_capacity_is_sum_of_server_speeds():
    profiles, _ = load_nodes("configs/nodes/pool6.yaml")
    requests = load_requests("configs/requests.yaml")
    fogA = [p for p in profiles if p.node_id == "fogA"]
    # travail moyen 0.6*5 + 0.3*40 + 0.1*150 = 30 ms ; vitesse 0.5 -> 16.7 req/s
    assert capacity_rps(fogA, requests) == pytest.approx(500 / 30)


def test_compact_layout_slots_follow_configuration():
    cfg = EnvConfig.from_yaml("configs/env/n1.yaml")
    assert cfg.slot_ids == ("fogA", "fogB", "fogC", "cloud")


def test_padded_layout_uses_whole_pool_and_active_subset():
    cfg = EnvConfig.from_yaml("configs/env/n3.yaml")
    assert cfg.slot_ids == ("fogA", "fogB", "fogC", "fogD", "fogE", "cloud")
    assert cfg.active_ids("n3") == ("fogA", "fogB", "cloud")
    assert cfg.randomization.enabled


def test_overrides_are_deep_merged_and_battery_can_be_removed():
    cfg = EnvConfig.from_yaml("configs/env/n0.yaml")
    assert cfg.profile("fogA").battery is None
    assert cfg.profile("fogA").rtt_ms == 20
    assert cfg.profile("fogA").p_max_w == 6.4  # le reste du profil est conserve


def test_invalid_configs_are_rejected():
    with pytest.raises(ValueError, match="inconnues"):
        EnvConfig.from_dict({"typo_key": 1})
    with pytest.raises(ValueError, match="compacte"):
        EnvConfig.from_dict({"configuration": ["n3", "n4"], "layout": "compact"})
    with pytest.raises(ValueError, match="Scenarios inconnus"):
        EnvConfig.from_dict({"scenarios": ["nope"]})
    with pytest.raises(ValueError, match="noeuds inconnus"):
        EnvConfig.from_dict({"node_overrides": {"ghost": {"rtt_ms": 1}}})
    with pytest.raises(ValueError):
        NodeProfile(node_id="x", speed=0)
    with pytest.raises(ValueError):
        NodeProfile(node_id="x", p_idle_w=5, p_max_w=1)


def test_digest_changes_when_config_changes():
    a = EnvConfig.from_yaml("configs/env/n1.yaml")
    b = EnvConfig.from_yaml("configs/env/n1.yaml", {"reward": {"w2": 0.9}})
    assert a.digest() != b.digest()
    assert a.digest() == EnvConfig.from_yaml("configs/env/n1.yaml").digest()
