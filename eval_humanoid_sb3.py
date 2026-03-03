# -*- coding: utf-8 -*-
"""
Evaluate a trained PPO on Humanoid-v5 with the saved VecNormalize statistics.
All comments are in English as requested.

Usage:
    python eval_humanoid_sb3.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_eval_env(render: bool = False):
    """
    Create a single evaluation environment.
    Use render_mode="human" if you want to visualize episodes.
    """
    def _init():
        env = gym.make("Humanoid-v4", render_mode="human" if render else None)
        return env
    return _init


def main(render: bool = False, steps: int = 5000):
    """
    Load:
    - VecNormalize stats (must match the ones used during training)
    - PPO policy
    Then run an evaluation loop and print episodic returns/lengths.
    """
    # Wrap a single env in DummyVecEnv to be compatible with SB3 VecEnv interface
    eval_env = DummyVecEnv([make_eval_env(render=render)])

    # Load the saved VecNormalize statistics and freeze them (no updates during eval)
    eval_env = VecNormalize.load("vecnormalize_humanoid.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False  # recommended to keep reward unnormalized in evaluation printouts

    # Load policy (the env passed must be the normalized one)
    model = PPO.load("ppo_humanoid_parallel", env=eval_env, device="cpu")

    # Rollout
    obs = eval_env.reset()
    ep_return, ep_len = 0.0, 0
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = eval_env.step(action)
        ep_return += float(reward)
        ep_len += 1

        if done[0]:
            print(f"[Eval] Episode return={ep_return:.2f}  length={ep_len}")
            ep_return, ep_len = 0.0, 0

    eval_env.close()


if __name__ == "__main__":
    # Set render=True for on-screen visualization (may slow down evaluation)
    main(render=True, steps=5000)