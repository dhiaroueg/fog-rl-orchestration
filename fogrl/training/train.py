"""Entrainement DQN / PPO / A2C (Stable-Baselines3) par niveaux de curriculum N0 a N3.

    python -m fogrl.training.train --config configs/train/n1.yaml --algo ppo --seed 0
    python -m fogrl.training.train --config configs/train/n1.yaml --algo all --seeds 0-4 --n-jobs 2

Sorties (une par algorithme et par graine, cahier 6.2) :
    models/<niveau>/<algo>/seed<k>/model.zip, vecnormalize.pkl, contract.json,
    evaluations.npz (courbe de convergence) ; TensorBoard dans runs/<niveau>/<algo>/.

Les graines d'entrainement sont dans 0-99 ; l'evaluation en cours
d'entrainement utilise une graine >= 1000, re-fixee a chaque evaluation pour que
les points de la courbe soient comparables.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..config import EnvConfig, load_yaml
from ..policies.rl import ALGOS, algo_class
from ..sim.env import FogEnv
from ..state_spec import contract_metadata
from ..utils import check_eval_seeds, check_train_seed, git_commit, library_versions, parse_seeds


def _split_hyper(hyper: Mapping[str, Any]) -> dict[str, Any]:
    """Convertit ``net_arch`` du YAML en ``policy_kwargs`` SB3."""

    hyper = dict(hyper)
    arch = hyper.pop("net_arch", None)
    if arch is not None:
        hyper["policy_kwargs"] = {"net_arch": list(arch)}
    return hyper


def make_vec_env(env_cfg: EnvConfig, n_envs: int, kind: str = "dummy", normalize: Optional[Mapping] = None, gamma: float = 0.99):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    def factory():
        return Monitor(FogEnv(env_cfg))

    vec_cls = SubprocVecEnv if kind == "subproc" and n_envs > 1 else DummyVecEnv
    vec = vec_cls([factory for _ in range(n_envs)])
    normalize = normalize or {}
    if normalize.get("obs") or normalize.get("reward"):
        vec = VecNormalize(
            vec,
            norm_obs=bool(normalize.get("obs")),
            norm_reward=bool(normalize.get("reward")),
            gamma=gamma,
            clip_obs=10.0,
            clip_reward=10.0,
        )
    return vec


def make_eval_env(env_cfg: EnvConfig, normalize: Optional[Mapping] = None):
    """Environnement d'evaluation : sans randomisation de domaine, recompense brute."""

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec = DummyVecEnv([lambda: Monitor(FogEnv(env_cfg))])
    normalize = normalize or {}
    if normalize.get("obs") or normalize.get("reward"):
        # Meme enveloppe que l'entrainement (sync_envs_normalization l'exige),
        # mais gelee et avec la recompense brute pour des courbes comparables.
        vec = VecNormalize(vec, training=False, norm_obs=bool(normalize.get("obs")), norm_reward=False)
    return vec


def build_model(algo: str, vec_env, hyper: Mapping[str, Any], seed: int, tensorboard_log: Optional[str], device: str = "cpu", verbose: int = 0):
    cls = algo_class(algo)
    return cls("MlpPolicy", vec_env, seed=seed, device=device, tensorboard_log=tensorboard_log, verbose=verbose, **_split_hyper(hyper))


def train(
    config: Mapping[str, Any],
    algo: str,
    seed: int,
    out_root: str = "models",
    runs_root: str = "runs",
    total_timesteps: Optional[int] = None,
    device: str = "cpu",
    verbose: int = 0,
) -> Path:
    """Entraine un modele et retourne son repertoire."""

    from stable_baselines3.common.vec_env import VecNormalize

    if algo not in ALGOS:
        raise ValueError(f"Algorithme inconnu : {algo!r} (disponibles : {ALGOS})")
    check_train_seed(seed)
    level = config["level"]
    hyper = dict(config["algos"][algo])
    n_envs = int(hyper.pop("n_envs", config.get("n_envs", 8)))
    steps = int(total_timesteps or config["total_timesteps"])
    normalize = config.get("normalize", {})
    eval_cfg = config.get("eval", {})

    env_cfg = EnvConfig.from_yaml(config["env"], config.get("env_overrides"))
    eval_env_cfg = EnvConfig.from_yaml(
        config["env"],
        {**(config.get("env_overrides") or {}), "randomization": {"enabled": False}, "episode_requests": eval_cfg.get("n_requests", env_cfg.episode_requests)},
    )
    out_dir = Path(out_root) / level / algo / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    vec = make_vec_env(env_cfg, n_envs, config.get("vec_env", "dummy"), normalize, gamma=hyper.get("gamma", 0.99))
    model = build_model(algo, vec, hyper, seed, str(Path(runs_root) / level / algo), device, verbose)

    callback = None
    if eval_cfg.get("freq", 0) > 0:
        eval_seed = int(eval_cfg.get("seed_start", 1000))
        check_eval_seeds([eval_seed])
        eval_vec = make_eval_env(eval_env_cfg, normalize)
        callback = _FixedSeedEvalCallback(
            eval_vec,
            eval_seed,
            n_eval_episodes=int(eval_cfg.get("episodes", 5)),
            eval_freq=max(int(eval_cfg["freq"]) // n_envs, 1),
            log_path=str(out_dir),
            deterministic=True,
            verbose=0,
        )

    started = time.time()
    model.learn(total_timesteps=steps, callback=callback, tb_log_name=f"seed{seed}", progress_bar=False)
    wall = time.time() - started

    model.save(str(out_dir / "model"))
    obs_normalized = bool(normalize.get("obs"))
    if isinstance(vec, VecNormalize):
        vec.save(str(out_dir / "vecnormalize.pkl"))
    metadata = {
        "algo": algo,
        "level": level,
        "train_seed": seed,
        "total_timesteps": steps,
        "n_envs": n_envs,
        "obs_normalized": obs_normalized,
        "reward_normalized": bool(normalize.get("reward")),
        "hyperparameters": dict(config["algos"][algo]),
        "contract": contract_metadata(env_cfg.slot_ids, env_cfg.normalization),
        "env_file": str(config["env"]),
        "env_digest": env_cfg.digest(),
        "env": env_cfg.to_dict(),
        "commit": git_commit(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(wall, 1),
        "steps_per_second": round(steps / wall, 1),
        "versions": library_versions(),
    }
    (out_dir / "contract.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    vec.close()
    return out_dir


def _make_fixed_seed_callback():
    from stable_baselines3.common.callbacks import EvalCallback

    class FixedSeedEvalCallback(EvalCallback):
        """EvalCallback qui rejoue les memes episodes d'evaluation a chaque evaluation."""

        def __init__(self, eval_env, eval_seed: int, **kwargs):
            super().__init__(eval_env, **kwargs)
            self.eval_seed = eval_seed

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                self.eval_env.seed(self.eval_seed)
            return super()._on_step()

    return FixedSeedEvalCallback


def _FixedSeedEvalCallback(*args, **kwargs):  # noqa: N802 - fabrique paresseuse (SB3 importe torch)
    return _make_fixed_seed_callback()(*args, **kwargs)


def _train_job(job: tuple) -> str:
    config, algo, seed, out_root, runs_root, steps, device = job
    return str(train(config, algo, seed, out_root, runs_root, steps, device))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--algo", default="ppo", help=f"{ALGOS} ou 'all'")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="ex. 0-4")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--out", default="models")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    algos = list(ALGOS) if args.algo == "all" else [args.algo]
    seeds = parse_seeds(args.seeds) if args.seeds else [args.seed if args.seed is not None else 0]
    jobs = [(config, a, s, args.out, args.runs, args.timesteps, args.device) for a in algos for s in seeds]
    if args.n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            for path in pool.map(_train_job, jobs):
                print("OK", path, flush=True)
    else:
        for job in jobs:
            print("OK", _train_job(job), flush=True)


if __name__ == "__main__":
    main()
