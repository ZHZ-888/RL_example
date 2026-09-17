# -*- coding: utf-8 -*-
"""Train PPO on Humanoid-v4 and keep every policy paired with VecNormalize."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

ENV_ID = "Humanoid-v4"


def make_env(env_id: str, rank: int, seed: int = 42) -> Callable[[], gym.Env]:
    """Return a factory for one monitored environment process."""
    def _init():
        env = Monitor(gym.make(env_id))
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec_env(
    env_id: str,
    num_envs: int,
    seed: int,
    vecnormalize_path: Path | None = None,
) -> VecNormalize:
    """Create parallel environments and optionally restore normalization state."""
    set_random_seed(seed)
    vec_env = SubprocVecEnv(
        [make_env(env_id, rank, seed) for rank in range(num_envs)]
    )
    if vecnormalize_path is not None:
        print(f"Loading normalization statistics from {vecnormalize_path}")
        return VecNormalize.load(str(vecnormalize_path), vec_env)
    return VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )


def build_ppo(vec_env: VecNormalize, tensorboard_dir: Path, model_path: Path | None) -> PPO:
    """Create a new PPO model or restore a model for continued training."""
    if model_path is not None:
        print(f"Loading model from {model_path}")
        return PPO.load(str(model_path), env=vec_env, device="auto")
    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
    )


class SaveBestVecNormalizeCallback(BaseCallback):
    """Save normalization statistics whenever EvalCallback saves a new best model."""

    def __init__(self, save_path: Path):
        super().__init__(verbose=0)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is None:
            raise RuntimeError("The training environment is not wrapped with VecNormalize")
        vec_normalize.save(str(self.save_path))
        return True


def create_run_dirs(output_dir: Path) -> dict[str, Path]:
    """Create and return the standard directory tree for one training run."""
    paths = {
        "root": output_dir,
        "final": output_dir / "final",
        "best": output_dir / "best",
        "checkpoints": output_dir / "checkpoints",
        "eval": output_dir / "eval",
        "tensorboard": output_dir / "tensorboard",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main_train(
    output_dir: Path,
    total_timesteps: int = 10_000_000,
    seed: int = 42,
    num_envs: int = 28,
    resume_from: Path | None = None,
) -> None:
    """Train PPO and save a self-contained set of run artifacts."""
    paths = create_run_dirs(output_dir)
    resume_model = None
    resume_vecnormalize = None
    if resume_from is not None:
        resume_model = resume_from / "model.zip"
        resume_vecnormalize = resume_from / "vecnormalize.pkl"
        for required_path in (resume_model, resume_vecnormalize):
            if not required_path.is_file():
                raise FileNotFoundError(f"Missing resume artifact: {required_path}")

    config = {
        "run_id": output_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "env_id": ENV_ID,
        "seed": seed,
        "num_envs": num_envs,
        "total_timesteps": total_timesteps,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "resume_from": str(resume_from.resolve()) if resume_from else None,
    }
    (paths["root"] / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    env = build_vec_env(ENV_ID, num_envs, seed, resume_vecnormalize)
    model = build_ppo(env, paths["tensorboard"], resume_model)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // num_envs, 1),
        save_path=str(paths["checkpoints"]),
        name_prefix="model",
        save_vecnormalize=True,
        verbose=1,
    )

    eval_env = build_vec_env(ENV_ID, 1, seed + 10)
    eval_env.training = False
    eval_env.norm_reward = False
    best_vecnormalize_cb = SaveBestVecNormalizeCallback(
        paths["best"] / "vecnormalize.pkl"
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["best"]),
        log_path=str(paths["eval"]),
        eval_freq=max(100_000 // num_envs, 1),
        deterministic=True,
        render=False,
        callback_on_new_best=best_vecnormalize_cb,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=resume_from is None,
            progress_bar=False,
        )
    finally:
        model.save(str(paths["final"] / "model"))
        env.save(str(paths["final"] / "vecnormalize.pkl"))
        env.close()
        eval_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run directory. Defaults to outputs/<timestamp>_seed<seed>.",
    )
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=28)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="A final/ or best/ directory containing model.zip and vecnormalize.pkl.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dir = args.output_dir
    if run_dir is None:
        run_id = f"{datetime.now():%Y%m%d_%H%M%S}_seed{args.seed}"
        run_dir = Path("outputs") / run_id
    main_train(
        output_dir=run_dir,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        num_envs=args.num_envs,
        resume_from=args.resume_from,
    )
