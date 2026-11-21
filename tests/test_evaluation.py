import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from fogrl.config import EnvConfig, load_yaml
from fogrl.evaluation import figures
from fogrl.evaluation.compare import compare_sim_real
from fogrl.evaluation.evaluate import evaluate_policy, make_policy
from fogrl.evaluation.experiments import expand_tasks, run_matrix
from fogrl.evaluation.stats import (
    acceptance_report,
    aggregate,
    mann_whitney_less,
    mean_ci,
    relative_reduction,
    render_markdown,
)
from fogrl.sim.env import FogEnv
from fogrl.utils import check_eval_seeds, parse_seeds

REQUIRED_COLUMNS = {
    "policy", "policy_type", "train_seed", "seed", "scenario", "size", "n_requests",
    "latency_mean_ms", "latency_p95_ms", "latency_p99_ms", "deadline_rate", "throughput_rps",
    "energy_fog_total_j", "energy_fog_j_per_req", "cloud_cost", "jain_cpu", "rejection_rate",
    "decision_ms_p99", "fallback_rate", "commit",
}


def test_parse_seeds_and_seed_separation():
    assert parse_seeds("1000-1002") == [1000, 1001, 1002]
    assert parse_seeds("1,5, 9") == [1, 5, 9]
    check_eval_seeds([1000, 2000])
    with pytest.raises(ValueError, match="reservees a l'entrainement"):
        check_eval_seeds([5])
    with pytest.raises(ValueError):
        parse_seeds(" , ")


def test_evaluate_policy_rows_have_all_metrics():
    config = EnvConfig.from_yaml("configs/env/n1.yaml", {"episode_requests": 300})
    env = FogEnv(config)
    profiles = {p.node_id: p for p in config.pool}
    policy = make_policy("greedy", profiles)
    rows = evaluate_policy(policy, env, [1000, 1001], policy_name="greedy", policy_type="baseline")
    assert len(rows) == 2 and REQUIRED_COLUMNS <= set(rows[0])
    assert rows[0]["n_requests"] == 300 and rows[0]["size"] == 4
    assert rows[0]["seed"] == 1000 and rows[0]["decision_ms_p99"] >= 0
    with pytest.raises(ValueError):
        evaluate_policy(policy, env, [3], policy_name="greedy", policy_type="baseline")


def test_same_policy_same_seed_gives_identical_row():
    config = EnvConfig.from_yaml("configs/env/n1.yaml", {"episode_requests": 300})
    env, profiles = FogEnv(config), {p.node_id: p for p in config.pool}
    a = evaluate_policy(make_policy("round_robin", profiles), env, [1000], policy_name="rr", policy_type="baseline")
    b = evaluate_policy(make_policy("round_robin", profiles), env, [1000], policy_name="rr", policy_type="baseline")
    for key in ("latency_mean_ms", "energy_fog_j_per_req", "cloud_cost", "deadline_rate"):
        assert a[0][key] == b[0][key]


def small_matrix(**changes):
    matrix = {
        "name": "t", "env": "configs/env/eval.yaml", "seeds": [1000, 1001], "n_requests": 300,
        "scenarios": ["medium"], "sizes": [4],
        "policies": [{"name": "greedy", "type": "baseline"}, {"name": "random", "type": "baseline"}],
    }
    matrix.update(changes)
    return matrix


def test_run_matrix_writes_csv_and_run_info(tmp_path):
    path = run_matrix(small_matrix(), tmp_path, verbose=False)
    df = pd.read_csv(path)
    assert len(df) == 4 and REQUIRED_COLUMNS <= set(df.columns)
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["n_rows"] == 4 and info["commit"] and info["env_digest"]
    assert info["matrix"]["name"] == "t"  # CSV associe au commit et a la config (regle 6.3)


def test_matrix_runs_in_parallel_with_same_results(tmp_path):
    serial = pd.read_csv(run_matrix(small_matrix(), tmp_path / "a", n_jobs=1, verbose=False))
    parallel = pd.read_csv(run_matrix(small_matrix(), tmp_path / "b", n_jobs=2, verbose=False))
    cols = ["policy", "seed", "latency_mean_ms", "cloud_cost"]
    pd.testing.assert_frame_equal(serial[cols], parallel[cols])


def test_matrix_skips_missing_rl_models_with_warning():
    matrix = small_matrix()
    matrix["policies"].append({"name": "ppo", "type": "rl", "model_glob": "models/none/ppo/seed*"})
    with pytest.warns(UserWarning, match="ignoree"):
        tasks = expand_tasks(matrix)
    assert {t["name"] for t in tasks} == {"greedy", "random"}


def test_matrix_rejects_training_seeds():
    with pytest.raises(ValueError):
        expand_tasks(small_matrix(seeds=[1, 2]))


def test_shipped_experiment_files_are_valid():
    for name in ("sim_baselines", "sim_full"):
        matrix = load_yaml(f"configs/experiments/{name}.yaml")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # modeles RL absents : ignores
            tasks = expand_tasks(matrix)
        assert tasks, name
    full = expand_tasks(load_yaml("configs/experiments/sim_baselines.yaml"))
    assert len(full) == 5 * 6 * 3  # 5 baselines x 6 scenarios x 3 tailles


# ----------------------------------------------------------------- statistiques


def test_mean_ci_and_aggregate():
    mean, half = mean_ci([10, 12, 14, 16, 18])
    assert mean == 14 and half == pytest.approx(3.93, abs=0.01)  # t(0.975, 4) * sd / sqrt(5)
    assert math.isnan(mean_ci([1.0])[1]) and math.isnan(mean_ci([])[0])
    df = pd.DataFrame({"policy": ["a"] * 3 + ["b"] * 3, "scenario": "s", "size": 4, "m": [1, 2, 3, 10, 10, 10]})
    agg = aggregate(df, ["m"]).set_index("policy")
    assert agg.loc["a", "m_mean"] == 2 and agg.loc["b", "m_ci95"] == 0 and agg.loc["a", "n"] == 3


def test_relative_reduction_and_mann_whitney():
    assert relative_reduction(100, 80) == pytest.approx(0.2)
    assert relative_reduction(100, 120) == pytest.approx(-0.2)
    assert math.isnan(relative_reduction(0, 1))
    low, high = np.arange(10, 20.0), np.arange(30, 40.0)
    assert mann_whitney_less(low, high) < 0.001
    assert mann_whitney_less(high, low) > 0.99
    assert mann_whitney_less(low, low) > 0.3
    assert mann_whitney_less(np.ones(5), np.ones(5)) == 1.0  # valeurs identiques
    assert math.isnan(mann_whitney_less(np.array([]), low))


def synthetic_results(rl_latency=70.0, rl_energy=0.40, rl_cost=0.08, with_rl=True):
    rng = np.random.default_rng(0)
    base = {  # latence, energie, cout par politique
        "round_robin": (430.0, 0.55, 0.28),
        "static_threshold": (220.0, 0.55, 0.20),
        "greedy": (80.0, 0.50, 0.10),
        "random": (450.0, 0.52, 0.27),
    }
    rows = []
    entries = {**base, **({"ppo": (rl_latency, rl_energy, rl_cost)} if with_rl else {})}
    for policy, (lat, energy, cost) in entries.items():
        for scenario in ("medium", "high", "spikes"):
            for seed in range(1000, 1005):
                noise = lambda: 1 + rng.normal(0, 0.01)
                rows.append({
                    "policy": policy, "policy_type": "rl" if policy == "ppo" else "baseline",
                    "scenario": scenario, "size": 4, "seed": seed,
                    "latency_mean_ms": lat * noise(), "energy_fog_j_per_req": energy * noise(),
                    "cloud_cost_per_req": cost * noise(),
                })
    return pd.DataFrame(rows)


def test_acceptance_report_passes_when_rl_is_clearly_better():
    report = acceptance_report(synthetic_results())
    assert report["best_rl"] == "ppo"
    results = {c["id"]: c["passed"] for c in report["checks"]}
    assert results == {"ENF-02 / H1": True, "ENF-03": True, "H2": True, "OS-4": True}
    assert "OK" in render_markdown(report)


def test_acceptance_report_flags_failures():
    report = acceptance_report(synthetic_results(rl_latency=350.0, rl_energy=0.54, rl_cost=0.2))
    results = {c["id"]: c["passed"] for c in report["checks"]}
    assert results["ENF-02 / H1"] is False  # 350 vs 430 : seulement -19 %
    assert results["ENF-03"] is False and results["H2"] is False and results["OS-4"] is False
    assert "NON ATTEINT" in render_markdown(report)


def test_acceptance_report_without_rl_or_baselines():
    assert "error" in acceptance_report(synthetic_results(with_rl=False))
    only_rl = synthetic_results()
    only_rl = only_rl[only_rl["policy"] == "ppo"]
    checks = {c["id"]: c["passed"] for c in acceptance_report(only_rl)["checks"]}
    assert checks["ENF-02 / H1"] is None and checks["ENF-03"] is None  # non evaluable


# ----------------------------------------------------------------------- sim/reel


def paired(sim_values, real_values):
    frame = lambda values: pd.DataFrame(
        [{"policy": p, "scenario": "medium", "latency_mean_ms": v} for p, v in values.items()]
    )
    return frame(sim_values), frame(real_values)


def test_compare_sim_real_passes_within_tolerance_and_same_ranking():
    sim, real = paired({"greedy": 80, "static": 220, "rr": 430}, {"greedy": 90, "static": 200, "rr": 480})
    report = compare_sim_real(sim, real)
    assert report["passed"] and report["same_ranking_everywhere"]
    assert report["max_relative_error"] == pytest.approx(0.125)


def test_compare_sim_real_fails_on_error_or_ranking():
    sim, real = paired({"greedy": 80, "static": 220, "rr": 430}, {"greedy": 160, "static": 200, "rr": 430})
    assert not compare_sim_real(sim, real)["all_within_tolerance"]
    sim, real = paired({"greedy": 80, "static": 85, "rr": 430}, {"greedy": 88, "static": 78, "rr": 430})
    report = compare_sim_real(sim, real)
    assert report["all_within_tolerance"] and not report["same_ranking_everywhere"] and not report["passed"]
    with pytest.raises(ValueError):
        compare_sim_real(sim, pd.DataFrame([{"policy": "x", "scenario": "y", "latency_mean_ms": 1}]))


# ---------------------------------------------------------------------- figures


def test_figures_are_generated(tmp_path):
    df = synthetic_results()
    df["latency_p95_ms"] = df["latency_mean_ms"] * 2.5
    assert figures.plot_latency_by_policy(df, tmp_path / "lat.png").stat().st_size > 1000
    assert figures.plot_latency_cdf(
        {"ppo": np.random.default_rng(0).lognormal(4, 0.5, 500), "greedy": np.random.default_rng(1).lognormal(4.3, 0.5, 500)},
        tmp_path / "cdf.png",
    ).exists()
    pareto = pd.DataFrame({"w2": [0, 0.5, 1.0], "latency_mean_ms": [70, 80, 95], "energy_fog_j_per_req": [0.5, 0.4, 0.35]})
    assert figures.plot_pareto(pareto, tmp_path / "pareto.png").exists()
    assert figures.plot_sim_vs_real(df, df, tmp_path / "simreal.png").exists()
    npz = []
    for seed in range(2):
        path = tmp_path / f"e{seed}.npz"
        np.savez(path, timesteps=np.arange(1, 6) * 1000, results=np.random.default_rng(seed).normal(-100, 5, (5, 3)), ep_lengths=np.ones((5, 3)))
        npz.append(path)
    assert figures.plot_convergence({"ppo": npz}, tmp_path / "conv.png").exists()
