"""Worker reel : machine a etats deterministe (NodeRuntime), API HTTP en memoire, executeur."""

import asyncio
import math
import os
import time
from contextlib import asynccontextmanager

import httpx
import pytest

from fogrl.config import BatteryConfig, NodeProfile
from fogrl.worker.app import WorkerSettings, create_app
from fogrl.worker.runtime import NodeRuntime
from fogrl.worker.workloads import WorkloadExecutor, Workloads, burn_cpu_ms


def profile(**overrides) -> NodeProfile:
    base = dict(
        node_id="n1", kind="fog", speed=1.0, servers=2, rtt_ms=10, p_idle_w=2.0, p_max_w=6.0, max_queue=4
    )
    base.update(overrides)
    return NodeProfile(**base)


# --------------------------------------------------------------- NodeRuntime


def test_queue_is_zero_while_a_server_is_free_then_follows_the_backlog():
    rt = NodeRuntime(profile(servers=2))
    assert rt.admit(0.0, "a", "medium", 100.0) is None
    rt.start(0.0, "a")
    assert rt.queue_wait_ms(0.0) == 0.0  # un serveur libre
    assert rt.admit(0.0, "b", "medium", 100.0) is None
    rt.start(0.0, "b")
    assert rt.queue_wait_ms(0.0) == pytest.approx(100.0)  # (100 + 100) / 2 serveurs
    assert rt.queue_wait_ms(0.04) == pytest.approx(60.0)  # le travail restant diminue
    rt.admit(0.04, "c", "medium", 100.0)  # en attente : 60 + 60 + 100 restants / 2
    assert rt.queue_wait_ms(0.04) == pytest.approx(110.0)


def test_cpu_load_is_the_smoothed_busy_fraction():
    rt = NodeRuntime(profile(servers=2))
    rt.admit(0.0, "a", "light", 10.0)
    rt.start(0.0, "a")
    # 1 serveur sur 2 occupe pendant tau = 1 s : 0.5 x (1 - exp(-1))
    rt.advance(1.0)
    assert rt.cpu_ema == pytest.approx(0.5 * (1 - math.exp(-1)), rel=1e-6)
    rt.finish(1.0, "a", 10.0, 10.0)
    rt.advance(30.0)
    assert rt.cpu_ema < 1e-6  # retombe vers 0


def test_load_reported_at_start_excludes_the_request_itself():
    rt = NodeRuntime(profile(servers=2))
    rt.admit(0.0, "a", "light", 10.0)
    assert rt.start(0.0, "a") == 0.0
    rt.admit(0.0, "b", "light", 10.0)
    assert rt.start(0.0, "b") == 0.5


def test_rejects_when_the_queue_is_full_or_the_node_is_down():
    rt = NodeRuntime(profile(max_queue=2))
    assert rt.admit(0.0, "a", "light", 5.0) is None
    assert rt.admit(0.0, "b", "light", 5.0) is None
    assert rt.admit(0.0, "c", "light", 5.0) == "queue_full"
    rt.down = True
    assert rt.admit(0.0, "d", "light", 5.0) == "unavailable"
    assert (rt.rejected_queue_full, rt.rejected_unavailable) == (1, 1)
    assert rt.snapshot(0.0)["available"] is False


def test_abort_releases_the_slot_without_counting_a_served_request():
    rt = NodeRuntime(profile(max_queue=1))
    rt.admit(0.0, "a", "light", 5.0)
    rt.abort(0.0, "a")
    assert rt.admit(0.0, "b", "light", 5.0) is None
    assert rt.served == 0


def test_duplicate_request_id_is_refused():
    rt = NodeRuntime(profile())
    rt.admit(0.0, "a", "light", 5.0)
    with pytest.raises(ValueError, match="deja en cours"):
        rt.admit(0.0, "a", "light", 5.0)


def test_energy_follows_the_linear_model():
    rt = NodeRuntime(profile(servers=2, p_idle_w=2.0, p_max_w=6.0))
    rt.admit(0.0, "a", "medium", 100.0)
    rt.start(0.0, "a")
    done = rt.finish(0.5, "a", 100.0, 500.0)
    assert done.energy_j == pytest.approx((6.0 - 2.0) / 2 * 0.5)  # (Pmax - Pidle) / servers x duree
    snap = rt.snapshot(2.0)
    assert snap["energy_idle_j"] == pytest.approx(2.0 * 2.0)  # Pidle x temps allume
    assert snap["energy_dynamic_j"] == pytest.approx(1.0)
    assert snap["busy_time_s"] == pytest.approx(0.5)


def test_cost_is_charged_per_request_on_the_cloud():
    rt = NodeRuntime(profile(kind="cloud", p_idle_w=0.0, p_max_w=0.0, cost_per_request=1.0))
    rt.admit(0.0, "a", "light", 5.0)
    rt.start(0.0, "a")
    assert rt.finish(0.01, "a", 5.0, 10.0).cost == 1.0


def test_battery_drains_shuts_the_node_down_and_restarts_after_recharge():
    battery = BatteryConfig(capacity_j=10.0, recharge_w=1.0, restart_fraction=0.5)
    rt = NodeRuntime(profile(servers=1, p_idle_w=2.0, p_max_w=6.0, battery=battery))
    assert rt.snapshot(0.0)["battery_fraction"] == 1.0
    rt.advance(2.0)  # -(2 - 1) W x 2 s a vide
    assert rt.battery.fraction == pytest.approx(0.8)
    rt.admit(2.0, "a", "heavy", 150.0)
    rt.start(2.0, "a")
    rt.finish(2.0, "a", 150.0, 3000.0)  # 4 W x 3 s = 12 J > 8 J disponibles
    assert rt.up is False and rt.battery.fraction == 0.0
    assert rt.admit(2.0, "b", "light", 5.0) == "unavailable"
    rt.advance(2.0 + 4.9)  # 4,9 J rechargees < 50 % de 10 J
    assert rt.up is False
    rt.advance(2.0 + 5.1)
    assert rt.up is True


def test_service_estimate_learns_the_observed_slowdown():
    rt = NodeRuntime(profile())
    assert rt.estimate_ms("medium", 100.0) == 100.0
    for i in range(30):
        rt.admit(float(i), f"r{i}", "medium", 100.0)
        rt.start(float(i), f"r{i}")
        rt.finish(float(i) + 0.2, f"r{i}", 100.0, 200.0)  # deux fois plus lent que prevu
    assert rt.estimate_ms("medium", 100.0) == pytest.approx(200.0, rel=0.02)


def test_reset_restores_counters_battery_and_availability():
    battery = BatteryConfig(capacity_j=10.0, recharge_w=0.0)
    rt = NodeRuntime(profile(battery=battery))
    rt.down = True
    rt.admit(0.0, "a", "light", 5.0)
    rt.advance(50.0)
    rt.reset(50.0)
    snap = rt.snapshot(50.0)
    assert snap["available"] and snap["in_flight"] == 0 and snap["battery_fraction"] == 1.0
    assert snap["rejected_unavailable"] == 0 and snap["uptime_s"] == 0.0


# ------------------------------------------------------------------ workloads


def test_burn_consumes_the_requested_cpu_time():
    start = time.thread_time()
    burn_cpu_ms(30.0)
    assert 0.028 <= time.thread_time() - start <= 0.06


def test_emulate_speed_divides_the_work_by_the_node_speed():
    assert Workloads("burn", speed=0.5, emulate_speed=True).effective_ms(10.0) == 20.0
    assert Workloads("burn", speed=0.5, emulate_speed=False).effective_ms(10.0) == 10.0
    with pytest.raises(ValueError):
        Workloads("gpu")


def test_real_mode_runs_numpy_kernels_without_onnx(caplog):
    workloads = Workloads("real", onnx_path=None)
    for req_type in ("light", "medium", "heavy"):
        start = time.perf_counter()
        workloads.run(req_type, 5.0)
        assert time.perf_counter() - start < 1.0


def test_process_executor_runs_servers_in_parallel_without_sharing_the_gil():
    if (os.cpu_count() or 1) < 2:
        pytest.skip("au moins 2 coeurs requis")
    executor = WorkloadExecutor(2, mode="burn", kind="process")
    try:
        executor.warm_up()
        start = time.perf_counter()
        futures = [executor.submit("light", 400.0) for _ in range(2)]
        for future in futures:
            future.result(timeout=30)
        elapsed = time.perf_counter() - start
    finally:
        executor.close()
    assert elapsed < 0.7  # deux jobs de 400 ms en parallele (pas 800 ms en serie)


# ---------------------------------------------------------------- API en memoire


@asynccontextmanager
async def worker_client(node_profile=None, **settings):
    node_profile = node_profile or profile()
    config = WorkerSettings(node_id=node_profile.node_id, executor="thread", **settings)
    app = create_app(config, node_profile)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
            client.app = app
            yield client


def body(request_id="r1", req_type="light", work_ms=20.0):
    return {"request_id": request_id, "type": req_type, "work_ms": work_ms}


def test_process_returns_durations_energy_and_cost():
    async def scenario():
        async with worker_client(profile(servers=1)) as client:
            reply = await client.post("/process", json=body(work_ms=30.0))
            state = (await client.get("/state")).json()
            return reply, state

    reply, state = asyncio.run(scenario())
    assert reply.status_code == 200
    data = reply.json()
    assert data["node"] == "n1" and data["type"] == "light"
    assert data["service_ms"] >= 28.0 and data["wait_ms"] < 20.0
    assert data["energy_j"] == pytest.approx((6.0 - 2.0) / 1 * data["service_ms"] / 1000.0)
    assert state["served"] == 1 and state["in_flight"] == 0 and state["available"] is True


def test_emulated_speed_slows_a_slow_node():
    async def scenario():
        async with worker_client(profile(speed=0.5, servers=1), emulate_speed=True) as client:
            return (await client.post("/process", json=body(work_ms=30.0))).json()

    assert asyncio.run(scenario())["service_ms"] >= 55.0  # 30 ms / 0.5


def test_requests_wait_for_a_free_server_and_the_state_shows_the_backlog():
    async def scenario():
        async with worker_client(profile(servers=1, max_queue=10)) as client:
            jobs = [asyncio.create_task(client.post("/process", json=body(f"r{i}", work_ms=80.0))) for i in range(3)]
            await asyncio.sleep(0.03)
            during = (await client.get("/state")).json()
            replies = [(await job).json() for job in jobs]
            return during, replies

    during, replies = asyncio.run(scenario())
    assert during["in_flight"] == 3 and during["queue_ms"] > 80.0
    waits = sorted(r["wait_ms"] for r in replies)
    service = min(r["service_ms"] for r in replies)
    # Le 1er n'attend pas ; le 2e attend le 1er, le 3e les deux premiers. Marges larges : avec l'executeur "thread"
    # du test, la boucle brulante se dispute le GIL et les requetes n'arrivent pas toutes au meme instant.
    assert waits[0] < 20.0 and waits[1] > 0.6 * service and waits[2] > 1.2 * service


def test_a_full_queue_answers_429_and_a_failed_node_503():
    async def scenario():
        async with worker_client(profile(servers=1, max_queue=1)) as client:
            first = asyncio.create_task(client.post("/process", json=body("a", work_ms=100.0)))
            await asyncio.sleep(0.02)
            full = await client.post("/process", json=body("b"))
            await first
            await client.post("/admin/fail")
            down = await client.post("/process", json=body("c"))
            state_down = (await client.get("/state")).json()
            await client.post("/admin/recover")
            back = await client.post("/process", json=body("d"))
            return full, down, state_down, back

    full, down, state_down, back = asyncio.run(scenario())
    assert full.status_code == 429 and full.json()["reason"] == "queue_full"
    assert down.status_code == 503 and down.json()["reason"] == "unavailable"
    assert state_down["available"] is False
    assert back.status_code == 200


def test_invalid_requests_are_refused_with_422():
    async def scenario():
        async with worker_client() as client:
            return [
                (await client.post("/process", json={"type": "giant", "work_ms": 5})).status_code,
                (await client.post("/process", json={"type": "light", "work_ms": -1})).status_code,
                (await client.post("/process", json={"type": "light", "work_ms": float("1e9")})).status_code,
            ]

    assert asyncio.run(scenario()) == [422, 422, 422]


def test_metrics_expose_node_gauges_and_request_counters():
    async def scenario():
        async with worker_client(profile(battery=BatteryConfig(capacity_j=100.0, recharge_w=1.0))) as client:
            await client.post("/process", json=body())
            return (await client.get("/metrics")).text

    text = asyncio.run(scenario())
    for expected in (
        'fog_node_queue_ms{node="n1"}',
        'fog_node_cpu_load{node="n1"}',
        'fog_node_up{node="n1"} 1.0',
        'fog_node_battery_fraction{node="n1"}',
        'fog_node_energy_joules_total{kind="dynamic",node="n1"}',
        'fog_worker_requests_total{node="n1",outcome="ok",type="light"} 1.0',
        "fog_worker_service_seconds_bucket",
    ):
        assert expected in text, expected


def test_simulated_rtt_is_added_to_state_calls():
    async def scenario():
        async with worker_client(profile(rtt_sigma=0.0), simulated_rtt_ms=60.0) as client:
            start = time.perf_counter()
            await client.get("/state")
            return (time.perf_counter() - start) * 1000.0

    assert asyncio.run(scenario()) >= 55.0


def test_admin_reset_clears_the_counters():
    async def scenario():
        async with worker_client() as client:
            await client.post("/process", json=body())
            return (await client.post("/admin/reset")).json()

    snap = asyncio.run(scenario())
    assert snap["served"] == 0 and snap["energy_dynamic_j"] == 0.0


def test_settings_from_env_validate_the_node(tmp_path):
    with pytest.raises(ValueError, match="NODE_ID"):
        WorkerSettings.from_env({})
    settings = WorkerSettings.from_env(
        {"NODE_ID": "fogC", "WORKER_MODE": "real", "EMULATE_SPEED": "1", "SIMULATED_RTT_MS": "30", "PORT": "9001"}
    )
    assert settings.mode == "real" and settings.emulate_speed and settings.port == 9001
    assert settings.load_profile().servers == 2  # fogC dans configs/nodes/pool6.yaml
    with pytest.raises(ValueError, match="absent"):
        WorkerSettings.from_env({"NODE_ID": "fogZ"}).load_profile()
