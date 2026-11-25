import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from fogrl.config import EnvConfig
from fogrl.sim.env import FogEnv
from fogrl.state_spec import observation_dim


def run_episode(env, action_fn, seed=0, options=None):
    obs, info = env.reset(seed=seed, options=options)
    total, steps, last = 0.0, 0, info
    while True:
        obs, reward, terminated, truncated, last = env.step(action_fn(env, obs))
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps, last


@pytest.mark.parametrize("name", ["n0", "n1", "n3"])
def test_check_env_passes(name):
    check_env(FogEnv(f"configs/env/{name}.yaml"), skip_render_check=True)


def test_spaces_match_state_spec_contract():
    env = FogEnv("configs/env/n1.yaml")
    assert env.observation_space.shape == (22,)  # 4*4 + 6 : cahier des charges
    assert env.action_space.n == 4
    padded = FogEnv("configs/env/n3.yaml")
    assert padded.observation_space.shape == (observation_dim(6),) == (30,)


def test_observation_is_float32_in_unit_interval_all_episode():
    env = FogEnv("configs/env/n3.yaml")
    obs, _ = env.reset(seed=1)
    for _ in range(300):
        assert obs.dtype == np.float32 and obs.min() >= 0.0 and obs.max() <= 1.0
        obs, *_ = env.step(env.action_space.sample())


def test_same_seed_same_trajectory():
    env = FogEnv("configs/env/n1.yaml")
    policy = lambda e, o: 2  # toujours fogC
    a = run_episode(env, policy, seed=5)
    b = run_episode(env, policy, seed=5)
    c = run_episode(env, policy, seed=6)
    assert a[0] == b[0] and a[2]["episode_metrics"] == b[2]["episode_metrics"]
    assert a[0] != c[0]


def test_episode_length_and_truncation():
    env = FogEnv("configs/env/n1.yaml", {"episode_requests": 50})
    total, steps, info = run_episode(env, lambda e, o: 0)
    assert steps == 50 and "episode_metrics" in info
    with pytest.raises(RuntimeError):
        env.step(0)  # episode termine : reset obligatoire


def test_episode_metrics_are_consistent():
    env = FogEnv("configs/env/n1.yaml")
    _, _, info = run_episode(env, lambda e, o: int(e.np_random.integers(4)))
    m = info["episode_metrics"]
    shares = sum(m[f"share_{n}"] for n in env.slot_ids)
    assert shares == pytest.approx(1.0)
    assert m["n_requests"] == 2000
    assert 0 <= m["deadline_rate"] <= 1 and 0 < m["jain_cpu"] <= 1
    assert m["latency_p99_ms"] >= m["latency_p95_ms"] >= 0
    assert m["energy_fog_total_j"] >= m["energy_fog_dyn_j_per_req"] * m["n_requests"]
    assert m["cloud_cost"] == pytest.approx(m["share_cloud"] * 2000 * 1.0, rel=0.05) or m["n_rejected"] > 0


def test_absent_slots_are_masked_and_choosing_one_is_a_rejection():
    env = FogEnv("configs/env/eval.yaml", {"configuration": "n4"})
    obs, info = env.reset(seed=0)
    assert list(info["action_mask"]) == [True, True, True, False, False, True]
    obs, reward, _, _, step_info = env.step(3)  # fogD absent en n4
    assert step_info["rejected"] and step_info["reason"] == "absent"
    # un rejet coute au moins autant que la pire reponse (latence plafonnee + echeance + rejet)
    reward_cfg = env.config.reward
    assert reward <= -(reward_cfg.w4 + reward_cfg.w5)


def test_state_is_stale_between_polls_but_consequences_are_real():
    env = FogEnv("configs/env/n1.yaml", {"poll_interval_ms": 1e9, "episode_requests": 60})
    obs0, _ = env.reset(seed=0)
    queue_seen, latencies = [obs0[0]], []
    for _ in range(40):
        obs, _, _, _, info = env.step(0)  # tout sur fogA
        queue_seen.append(obs[0])
        if info["latency_ms"] is not None:
            latencies.append(info["latency_ms"])
    assert len(set(queue_seen)) == 1  # l'agent voit toujours la meme file
    assert max(latencies) > 3 * min(latencies)  # alors que la file reelle grossit


def test_polling_refreshes_state_when_interval_is_short():
    env = FogEnv("configs/env/n1.yaml", {"poll_interval_ms": 0, "episode_requests": 60})
    obs0, _ = env.reset(seed=0)
    seen = {obs0[0]}
    for _ in range(40):
        obs, *_ = env.step(0)
        seen.add(obs[0])
    assert len(seen) > 5


def test_outage_masks_node_and_rejects_requests():
    cfg = {"outages": [{"node": "fogB", "start_s": 0.0, "end_s": 1000.0}]}
    env = FogEnv("configs/env/n1.yaml", cfg)
    obs, info = env.reset(seed=0)
    assert list(info["action_mask"]) == [True, False, True, True]
    _, _, _, _, step_info = env.step(1)
    assert step_info["rejected"] and step_info["reason"] == "unavailable"


def test_raw_state_exposes_unnormalized_values_for_baselines():
    env = FogEnv("configs/env/n1.yaml")
    env.reset(seed=0)
    nodes, request, ctx = env.raw_state()
    assert [n.node_id for n in nodes] == ["fogA", "fogB", "fogC", "cloud"]
    assert nodes[3].net_rtt_ms > 20  # RTT du cloud en ms, pas normalise
    assert request.request_type in ("light", "medium", "heavy") and ctx.now_s > 0


def test_domain_randomization_changes_episodes_but_not_with_fixed_seed():
    env = FogEnv("configs/env/n3.yaml")
    rates = []
    for seed in (1, 2, 3):
        _, info = env.reset(seed=seed)
        rates.append((info["configuration"], info["scenario"], round(info["base_rate_rps"], 3)))
    assert len(set(rates)) == 3
    _, again = env.reset(seed=1)
    assert (again["configuration"], again["scenario"], round(again["base_rate_rps"], 3)) == rates[0]


def test_reset_options_force_configuration_and_scenario():
    env = FogEnv("configs/env/n3.yaml")
    _, info = env.reset(seed=0, options={"configuration": "n6", "scenario": "high"})
    assert info["configuration"] == "n6" and info["scenario"] == "high" and info["n_active"] == 6
    with pytest.raises(ValueError):
        env.reset(options={"scenario": "azure"})  # hors de cette config
