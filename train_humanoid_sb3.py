# -*- coding: utf-8 -*-
"""
Train PPO on Humanoid-v4 with parallel environments (SubprocVecEnv) and observation/reward normalization.
All comments are in English as requested.

Usage:
    python train_humanoid_sb3.py
"""

import os
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

def make_env(env_id: str, rank: int, seed: int = 42) -> Callable[[], gym.Env]:
    """
    Factory that returns a callable to create a single environment instance.
    Using a thunk avoids pickling issues when launching subprocesses.
    :param
        - env_id: the Gym environment ID (e.g., "Humanoid-v4")
        - rank: the index of the environment (used to set different seeds)
        - seed: base random seed for reproducibility
    """
    def _init():
        # Create the Mujoco Humanoid environment
        env = gym.make(env_id)  # "Humanoid-v4"
        # Wrap with Monitor to record episode statistics (length, return)
        env = Monitor(env)
        # Set a different seed for each subprocess to decorrelate rollouts
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec_env(env_id: str = "Humanoid-v4",
                  num_envs: int | None = None,
                  seed: int = 42) -> VecNormalize:
    """
    Build a parallel vectorized environment with SubprocVecEnv and attach VecNormalize.
    VecNormalize is important on high-dimensional continuous control tasks like Humanoid.
    """
    if num_envs is None:
        # A good starting point: half of available CPU cores, but at least 4
        num_envs = max(4, (os.cpu_count() or 8) // 2)

    set_random_seed(seed)
    # env_fns => environment functions
    env_fns = [make_env(env_id, i, seed) for i in range(num_envs)] # ? what's this for?

    # IMPORTANT (Windows/macOS): creating SubprocVecEnv must happen under
    # if __name__ == "__main__": guard in the entry file to avoid deadlocks.
    vec_env = SubprocVecEnv(env_fns)

    # Normalize observations and rewards; clip to avoid extreme values
    vec_env = VecNormalize(vec_env,
                           norm_obs=True,
                           norm_reward=True,
                           clip_obs=10.0,
                           clip_reward=10.0)
    return vec_env


def build_ppo(vec_env: VecNormalize,
              tb_logdir: str = "runs/ppo_humanoid") -> PPO:
    """
    Build a PPO model. Keep n_steps * num_envs around ~2048 as a reasonable default.
    You can increase total_timesteps later for better performance.
    """
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        # Keep total rollout size per update ~2048
        n_steps=256 if vec_env.num_envs == 8 else 128, # vec_env.num_envs
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        tensorboard_log=tb_logdir,
        verbose=1,
    )
    return model


def main_train(total_timesteps: int = 2_000_000, seed: int = 42):
    """
    Main training entrypoint:
    - Build vectorized env
    - Create PPO
    - Configure checkpoint & evaluation callbacks
    - Train and save both model and VecNormalize statistics
    """
    env = build_vec_env("Humanoid-v4", seed=seed)
    model = build_ppo(env)

    # Save checkpoints periodically (by timesteps divided across envs)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // env.num_envs, 1000),
        save_path="./checkpoints",
        name_prefix="ppo_humanoid"
    )

    # Build a separate evaluation env sharing the same VecNormalize structure.
    # We will freeze normalization stats during evaluation (training=False).
    eval_env = build_vec_env("Humanoid-v4", seed=seed + 10)
    eval_env.training = False
    eval_env.norm_reward = False  # usually we only normalize observations at eval time

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./eval_logs",
        eval_freq=max(50_000 // env.num_envs, 2000),
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])

    # Save the policy and VecNormalize stats for future evaluation
    model.save("ppo_humanoid_parallel")
    env.save("vecnormalize_humanoid.pkl")

    # Clean up
    env.close()
    eval_env.close()

# >>> ADD this function into your train_humanoid_sb3.py

def resume_train(
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    model_path: str = "ppo_humanoid_parallel.zip",
    vecnorm_path: str = "vecnormalize_humanoid.pkl",
):
    """
    Continue training from a previously saved PPO model.
    This loads both the model weights and VecNormalize statistics.
    """

    # Rebuild vectorized training environment
    env = build_vec_env("Humanoid-v4", seed=seed)

    # Load existing VecNormalize statistics into the environment
    env = VecNormalize.load(vecnorm_path, env)
    env.training = True          # Continue updating normalization stats
    env.norm_reward = True       # Keep reward normalization enabled

    # Load the PPO model (weights + optimizer state)
    model = PPO.load(model_path, env=env, device="auto")

    # Build evaluation environment with same VecNormalize stats
    eval_env = build_vec_env("Humanoid-v4", seed=seed + 100)
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False    # Do not update stats during evaluation
    eval_env.norm_reward = False # Show raw rewards

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_resume",
        log_path="./eval_logs_resume",
        eval_freq=max(50_000 // env.num_envs, 2000),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Continue training WITHOUT resetting global timestep counter
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb],
        reset_num_timesteps=False,
        progress_bar=True,
    )

    # Save updated model and normalization stats
    model.save("ppo_humanoid_parallel")
    env.save("vecnormalize_humanoid.pkl")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    # IMPORTANT on Windows/macOS: SubprocVecEnv must be created under this guard.
    main_train()

    # To continue training instead of starting fresh:
    # resume_train(total_timesteps=1_000_000)
