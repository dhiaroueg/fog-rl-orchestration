import numpy as np
import pytest

from fogrl.config import BatteryConfig, NodeProfile
from fogrl.sim.energy import BatteryModel, dynamic_energy_j, power_w
from fogrl.sim.node import NodeSim
from fogrl.sim.service import AnalyticServiceModel, CalibratedServiceModel


def make_profile(**changes):
    values = dict(
        node_id="n", speed=1.0, servers=1, rtt_ms=0.0, rtt_sigma=0.0, service_cv=0.0,
        contention=0.0, p_idle_w=2.0, p_max_w=6.0, max_queue=3,
    )
    values.update(changes)
    return NodeProfile(**values)


def make_node(**changes):
    return NodeSim(make_profile(**changes), AnalyticServiceModel(), np.random.default_rng(0))


def test_power_is_linear_in_utilization():
    profile = make_profile()
    assert power_w(profile, 0.0) == 2.0
    assert power_w(profile, 0.5) == 4.0
    assert power_w(profile, 1.0) == 6.0
    assert power_w(profile, 7.0) == 6.0  # borne
    # un serveur sur deux occupe pendant 1 s = moitie de la puissance dynamique
    assert dynamic_energy_j(make_profile(servers=2), 1.0) == pytest.approx(2.0)


def test_battery_drains_turns_off_and_restarts_by_recharge_only():
    battery = BatteryModel(BatteryConfig(capacity_j=100, recharge_w=1.0, restart_fraction=0.25))
    battery.step(10, 6.0)  # -5 W net pendant 10 s
    assert battery.energy_j == pytest.approx(50.0)
    battery.step(100, 6.0)  # se vide
    assert not battery.on and battery.energy_j == 0.0
    battery.drain(5.0)  # eteinte : ignore
    assert battery.energy_j == 0.0
    battery.step(20, 6.0)  # recharge seule : 20 J < 25 J
    assert not battery.on
    battery.step(6, 6.0)  # 26 J >= 25 J
    assert battery.on and battery.energy_j == pytest.approx(26.0)
    battery.step(10_000, 0.0)
    assert battery.energy_j == pytest.approx(100.0)  # plafonnee a la capacite


def test_fifo_wait_and_latency_on_single_server():
    node = make_node(speed=0.5)  # 40 ms de travail -> 80 ms de service
    first = node.submit(0.0, "medium", 40.0)
    second = node.submit(0.0, "medium", 40.0)
    assert first.service_ms == pytest.approx(80.0) and first.wait_ms == 0.0
    assert second.wait_ms == pytest.approx(80.0)
    assert second.latency_ms == pytest.approx(160.0)
    assert node.queue_wait_s(0.0) == pytest.approx(0.16)
    assert node.in_flight(0.0) == 2
    assert node.in_flight(0.2) == 0


def test_multi_server_runs_in_parallel():
    node = make_node(servers=2, speed=1.0, max_queue=10)
    results = [node.submit(0.0, "medium", 40.0) for _ in range(3)]
    assert [r.wait_ms for r in results] == pytest.approx([0.0, 0.0, 40.0])
    assert node.busy_fraction(0.01) == 1.0


def test_queue_full_rejects_and_counts():
    node = make_node(max_queue=3)
    outcomes = [node.submit(0.0, "light", 5.0) for _ in range(5)]
    assert [o.accepted for o in outcomes] == [True, True, True, False, False]
    assert outcomes[3].reason == "queue_full"
    assert node.rejected == 2 and node.accepted == 3


def test_down_node_and_empty_battery_reject():
    node = make_node()
    node.down = True
    assert node.submit(0.0, "light", 5.0).reason == "unavailable"
    profile = make_profile(battery=BatteryConfig(capacity_j=10, recharge_w=0.0))
    dead = NodeSim(profile, AnalyticServiceModel(), np.random.default_rng(0), battery_fraction=0.0)
    assert not dead.up
    assert dead.submit(0.0, "light", 5.0).reason == "unavailable"


def test_dynamic_energy_and_battery_drain_on_submit():
    profile = make_profile(battery=BatteryConfig(capacity_j=100, recharge_w=0.0))
    node = NodeSim(profile, AnalyticServiceModel(), np.random.default_rng(0))
    result = node.submit(0.0, "heavy", 1000.0)  # 1 s de service, (6-2) W
    assert result.energy_j == pytest.approx(4.0)
    assert node.battery.energy_j == pytest.approx(96.0)
    node.advance(10.0)  # 10 s de puissance a vide : 2 W
    assert node.battery.energy_j == pytest.approx(76.0)
    assert node.idle_energy_j() == pytest.approx(20.0)
    assert node.utilization(10.0) == pytest.approx(0.1)


def test_cpu_load_is_smoothed_busy_fraction():
    node = make_node()
    node.submit(0.0, "heavy", 5000.0)  # occupe 5 s
    node.advance(0.5)
    low = node.cpu_ema
    node.advance(4.0)
    assert 0 < low < node.cpu_ema <= 1.0


def test_contention_inflates_service_time():
    profile = make_profile(contention=1.0, servers=2, max_queue=10)
    node = NodeSim(profile, AnalyticServiceModel(), np.random.default_rng(0))
    first = node.submit(0.0, "medium", 40.0)  # serveurs libres : charge 0
    second = node.submit(0.0, "medium", 40.0)  # un serveur occupe : charge 0.5
    assert first.service_ms == pytest.approx(40.0)
    assert second.service_ms == pytest.approx(60.0)


def test_calibrated_model_interpolates_and_falls_back():
    table = {
        ("n", "medium"): {"loads": [0.1, 0.9], "mean_ms": [40.0, 80.0], "cv": [0.0, 0.0]},
    }
    model = CalibratedServiceModel(table, {"medium": 40.0, "light": 5.0})
    profile = make_profile()
    rng = np.random.default_rng(0)
    assert model.sample_ms(rng, profile, "medium", 40.0, 0.5) == pytest.approx(60.0)
    assert model.sample_ms(rng, profile, "medium", 80.0, 0.5) == pytest.approx(120.0)  # taille x2
    # type absent de la table -> modele analytique
    assert model.sample_ms(rng, profile, "light", 5.0, 0.0) == pytest.approx(5.0)
    assert model.covers("n", "medium") and not model.covers("n", "light")
