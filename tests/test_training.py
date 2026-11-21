import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from fogrl.config import EnvConfig, load_yaml
from fogrl.evaluation.evaluate import make_policy, run_episode
from fogrl.evaluation.experiments import run_matrix
from fogrl.policies.rl import ALGOS, RLPolicy
from fogrl.sim.env import FogEnv
from fogrl.state_spec import DecisionContext, NodeState, RequestState, absent_node
from fogrl.training.train import train

warnings.filterwarnings("ignore")
STEPS = 2500


def tiny_config(level="n1", **extra):
    config = load_yaml(f"configs/train/{level}.yaml")
    config["eval"] = {"freq": 0}
    config["n_envs"] = 2
    for algo in config["algos"].values():
        algo.pop("n_envs", None)
    config["algos"]["ppo"].update(n_steps=64, batch_size=64)
    config["algos"]["dqn"].update(learning_starts=200, buffer_size=5000)
    config.update(extra)
    return config


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    return {
        algo: train(tiny_config(), algo, 0, out_root=str(root), runs_root=str(root / "runs"), total_timesteps=STEPS)
        for algo in ALGOS
    }


def test_train_writes_all_artifacts_and_contract(models):
    for algo, directory in models.items():
        names = {p.name for p in directory.iterdir()}
        assert {"model.zip", "contract.json", "vecnormalize.pkl"} <= names, algo
        meta = json.loads((directory / "contract.json").read_text())
        assert meta["algo"] == algo and meta["train_seed"] == 0 and meta["level"] == "n1"
        assert meta["contract"]["node_ids"] == ["fogA", "fogB", "fogC", "cloud"]
        assert meta["contract"]["observation_dim"] == 22 and meta["contract"]["spec_version"] == 2
        assert meta["commit"] and meta["env_digest"] and meta["wall_time_s"] > 0
        assert meta["obs_normalized"] is False and meta["reward_normalized"] is True


@pytest.mark.parametrize("algo", ALGOS)
def test_rl_policy_matches_sb3_predict_when_all_nodes_available(models, algo):
    """Notre inference (logits / Q-values) doit donner la meme action que model.predict."""

    config = EnvConfig.from_yaml("configs/env/n1.yaml")
    env = FogEnv(config)
    policy = RLPolicy.load(models[algo], slot_ids=config.slot_ids, normalization=config.normalization)
    obs, _ = env.reset(seed=3)
    for _ in range(120):
        nodes, request, ctx = env.raw_state()
        expected, _ = policy.model.predict(obs, deterministic=True)
        assert policy.select(nodes, request, ctx) == int(expected)
        obs, *_ = env.step(int(expected))


@pytest.mark.parametrize("algo", ALGOS)
def test_rl_policy_never_picks_masked_nodes(models, algo):
    config = EnvConfig.from_yaml("configs/env/n1.yaml")
    policy = RLPolicy.load(models[algo], slot_ids=config.slot_ids, normalization=config.normalization)
    rng = np.random.default_rng(0)
    request, ctx = RequestState("medium", 40.0, 300.0), DecisionContext(0.0, 60.0)
    for _ in range(200):
        unavailable = rng.choice(4, size=int(rng.integers(0, 4)), replace=False)  # au moins un reste
        nodes = [
            NodeState(n, float(rng.uniform(0, 800)), float(rng.uniform()), float(rng.uniform(5, 100)),
                      0.8, available=i not in unavailable)
            for i, n in enumerate(config.slot_ids)
        ]
        choice = policy.select(nodes, request, ctx)
        assert nodes[choice].available


def test_contract_mismatch_is_detected_at_load(models):
    directory = models["ppo"]
    with pytest.raises(ValueError, match="node_ids"):
        RLPolicy.load(directory, slot_ids=["fogB", "fogA", "fogC", "cloud"])
    with pytest.raises(ValueError, match="normalization"):
        from fogrl.state_spec import Normalization
        RLPolicy.load(directory, slot_ids=["fogA", "fogB", "fogC", "cloud"], normalization=Normalization(queue_ms=5.0))
    with pytest.raises(ValueError):
        RLPolicy.load(directory, slot_ids=["fogA", "fogB", "cloud"])


def test_policy_runs_in_the_evaluation_pipeline(models):
    config = EnvConfig.from_yaml("configs/env/n1.yaml", {"episode_requests": 300})
    env = FogEnv(config)
    profiles = {p.node_id: p for p in config.pool}
    policy = make_policy(f"rl:{models['ppo']}", profiles, config.slot_ids, config.normalization)
    metrics = run_episode(env, policy, 1000)
    assert metrics["n_requests"] == 300 and metrics["decision_ms_p99"] < 50  # ENF-01 en simulation


def test_train_rejects_bad_seeds_and_algos(tmp_path):
    with pytest.raises(ValueError, match="0-99"):
        train(tiny_config(), "ppo", 100, out_root=str(tmp_path), total_timesteps=10)
    with pytest.raises(ValueError, match="inconnu"):
        train(tiny_config(), "sac", 0, out_root=str(tmp_path), total_timesteps=10)
    bad_eval = tiny_config(eval={"freq": 100, "seed_start": 5})
    with pytest.raises(ValueError, match="reservees"):
        train(bad_eval, "ppo", 0, out_root=str(tmp_path), total_timesteps=200)


def test_observation_normalization_stats_are_applied_at_inference(tmp_path):
    config = tiny_config(normalize={"obs": True, "reward": True})
    directory = train(config, "ppo", 1, out_root=str(tmp_path), runs_root=str(tmp_path / "r"), total_timesteps=STEPS)
    env_cfg = EnvConfig.from_yaml("configs/env/n1.yaml")
    policy = RLPolicy.load(directory, slot_ids=env_cfg.slot_ids, normalization=env_cfg.normalization)
    assert policy.obs_stats is not None
    with open(directory / "vecnormalize.pkl", "rb") as handle:
        vec_normalize = pickle.load(handle)
    obs = FogEnv(env_cfg).reset(seed=0)[0]
    np.testing.assert_allclose(policy._normalize(obs), vec_normalize.normalize_obs(obs), rtol=1e-4, atol=1e-5)
    nodes, request, ctx = (lambda e: (e.reset(seed=0), e.raw_state())[1])(FogEnv(env_cfg))
    expected, _ = policy.model.predict(policy._normalize(obs), deterministic=True)
    assert policy.select(nodes, request, ctx) == int(expected)


def test_padded_n3_model_serves_3_4_and_6_nodes_through_the_matrix(tmp_path):
    config = tiny_config("n3", total_timesteps=STEPS)
    train(config, "ppo", 0, out_root=str(tmp_path / "models"), runs_root=str(tmp_path / "r"), total_timesteps=STEPS)
    matrix = {
        "name": "rl_smoke", "env": "configs/env/eval.yaml", "seeds": [1000], "n_requests": 300,
        "scenarios": ["medium"], "sizes": [3, 4, 6],
        "policies": [
            {"name": "greedy", "type": "baseline"},
            {"name": "ppo", "type": "rl", "model_glob": str(tmp_path / "models/n3/ppo/seed*")},
        ],
    }
    frame = pd.read_csv(run_matrix(matrix, tmp_path / "out", verbose=False))
    rl = frame[frame["policy"] == "ppo"]
    assert sorted(rl["size"]) == [3, 4, 6] and (rl["train_seed"] == 0).all()
    assert (rl["policy_type"] == "rl").all() and rl["rejection_rate"].max() < 1.0
    # un noeud absent ne recoit jamais de requete (masque a l'inference)
    assert rl.loc[rl["size"] == 4, ["share_fogD", "share_fogE"]].to_numpy().sum() == 0
    assert rl.loc[rl["size"] == 3, ["share_fogC", "share_fogD", "share_fogE"]].to_numpy().sum() == 0


def test_optuna_tuning_returns_best_ppo_params():
    from fogrl.training.tune import tune

    result = tune(load_yaml("configs/train/n0.yaml"), n_trials=2, trial_timesteps=600, eval_episodes=1, n_envs=2)
    assert result["n_trials"] == 2 and "learning_rate" in result["best_params"]
    assert len(result["best_params"]["net_arch"]) == 2


@pytest.mark.slow
def test_n0_consistency_check_ppo_prefers_fast_node():
    from fogrl.training.sanity import run_n0_check

    result = run_n0_check(timesteps=60000)
    assert result["passed"], result
    assert result["fast_node_share"] >= 0.8 and result["latency_rl_ms"] < result["latency_random_ms"]
