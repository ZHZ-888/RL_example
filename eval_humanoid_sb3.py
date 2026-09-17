# -*- coding: utf-8 -*-
"""Evaluate a saved Humanoid-v4 policy package."""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_eval_env(render: bool = False):
    """Return a factory for one evaluation environment."""
    def _init():
        return gym.make("Humanoid-v4", render_mode="human" if render else None)
    return _init


def main(model_dir: Path, render: bool = False, steps: int = 5000) -> None:
    """Load model.zip and vecnormalize.pkl from one policy directory."""
    model_path = model_dir / "model.zip"
    if not model_path.is_file():
        model_path = model_dir / "best_model.zip"
    vecnormalize_path = model_dir / "vecnormalize.pkl"
    for required_path in (model_path, vecnormalize_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing policy artifact: {required_path}")

    eval_env = DummyVecEnv([make_eval_env(render=render)])
    eval_env = VecNormalize.load(str(vecnormalize_path), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    model = PPO.load(str(model_path), env=eval_env, device="cpu")
    obs = eval_env.reset()
    ep_return, ep_len = 0.0, 0

    try:
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            ep_return += float(reward[0])
            ep_len += 1
            if done[0]:
                print(f"[Eval] Episode return={ep_return:.2f}  length={ep_len}")
                ep_return, ep_len = 0.0, 0
    finally:
        eval_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing model.zip and vecnormalize.pkl.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(model_dir=args.model_dir, render=args.render, steps=args.steps)
