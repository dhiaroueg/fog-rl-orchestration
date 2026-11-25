"""Reglage des hyperparametres de PPO avec Optuna (cahier, Phase 3 etape 5 : 30 a 50 essais courts).

    python -m fogrl.training.tune --config configs/train/n3.yaml --trials 40 --trial-timesteps 60000 \\
        --out configs/train/tuned_ppo.yaml

Chaque essai entraine un PPO court et retourne la recompense moyenne sur des
episodes d'evaluation fixes (graine 1000). Les meilleurs hyperparametres sont
ecrits dans un YAML a recopier dans ``algos.ppo`` du fichier de niveau.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from ..config import EnvConfig, load_yaml
from .train import build_model, make_eval_env, make_vec_env


def suggest_ppo(trial) -> dict[str, Any]:
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256])
    return {
        "n_steps": n_steps,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs": trial.suggest_int("n_epochs", 4, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.03, log=True),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "net_arch": [trial.suggest_categorical("hidden", [32, 64, 128])] * 2,
    }


def tune(
    config: Mapping[str, Any],
    n_trials: int = 30,
    trial_timesteps: int = 40000,
    eval_episodes: int = 3,
    n_envs: int = 4,
    seed: int = 0,
    storage: Optional[str] = None,
) -> dict[str, Any]:
    import optuna
    from stable_baselines3.common.evaluation import evaluate_policy

    env_cfg = EnvConfig.from_yaml(config["env"], config.get("env_overrides"))
    eval_cfg = EnvConfig.from_yaml(
        config["env"], {**(config.get("env_overrides") or {}), "randomization": {"enabled": False}}
    )
    normalize = config.get("normalize", {})

    def objective(trial) -> float:
        hyper = suggest_ppo(trial)
        # PPO exige que batch_size divise n_steps * n_envs
        while (hyper["n_steps"] * n_envs) % hyper["batch_size"]:
            hyper["batch_size"] //= 2
        vec = make_vec_env(env_cfg, n_envs, "dummy", normalize, gamma=hyper["gamma"])
        model = build_model("ppo", vec, hyper, seed + trial.number, None)
        try:
            model.learn(total_timesteps=trial_timesteps)
            eval_vec = make_eval_env(eval_cfg, normalize)
            if hasattr(vec, "obs_rms") and hasattr(eval_vec, "obs_rms"):
                eval_vec.obs_rms = vec.obs_rms
            eval_vec.seed(1000)
            mean_reward, _ = evaluate_policy(model, eval_vec, n_eval_episodes=eval_episodes, deterministic=True)
            eval_vec.close()
        finally:
            vec.close()
        return float(mean_reward)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        study_name="fogrl_ppo" if storage else None,
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_trial.params)
    hidden = best.pop("hidden")
    best["net_arch"] = [hidden, hidden]
    return {"best_value": study.best_value, "best_params": best, "n_trials": len(study.trials)}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--trial-timesteps", type=int, default=40000)
    parser.add_argument("--storage", default=None, help="ex. sqlite:///runs/optuna.db")
    parser.add_argument("--out", default="configs/train/tuned_ppo.yaml")
    args = parser.parse_args(argv)
    result = tune(load_yaml(args.config), args.trials, args.trial_timesteps, storage=args.storage)
    Path(args.out).write_text(
        "# Hyperparametres PPO trouves par Optuna ; a recopier dans algos.ppo du fichier de niveau.\n"
        + yaml.safe_dump({"best_value": result["best_value"], "ppo": result["best_params"]}, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Meilleure recompense : {result['best_value']:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
