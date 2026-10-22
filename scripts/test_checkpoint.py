
"""Evaluate a trained policy checkpoint without human intervention."""

import argparse
from pathlib import Path

import torch
import yaml

from dagger_lh.config import EnvConfig
from dagger_lh.evaluate import evaluate_policy, sample_init_states
from dagger_lh.envs import make_env
from dagger_lh.policies import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoints/pickplace/ladder/bc_rnn_PickPlaceCan_p063.pth",
    )
    parser.add_argument(
        "--config",
        default="configs/study_pickplace.yaml",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.episodes < 1 or args.horizon < 1:
        parser.error("--episodes and --horizon must be positive")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    env_cfg = EnvConfig(**config_data["env"])

    print(f"Checkpoint: {checkpoint}")
    print(f"Environment: {env_cfg.env_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.horizon}")
    print(f"Seed: {args.seed}")

    device = torch.device("cpu")
    policy, extra = load_policy(str(checkpoint), device=device)
    policy.eval()

    env = make_env(env_cfg)

    try:
        init_states = sample_init_states(
            env,
            n=args.episodes,
            seed=args.seed + 9901,
        )

        with torch.no_grad():
            result = evaluate_policy(
                policy=policy,
                env_cfg=env_cfg,
                n_episodes=args.episodes,
                horizon=args.horizon,
                seed=args.seed + 2,
                env=env,
                init_states=init_states,
                progress=True,
            )
    finally:
        env.close()

    successes = result["successes"]
    total = result["n_episodes"]

    print("\n========== EVALUATION RESULT ==========")
    print(f"Successes: {successes}/{total}")
    print(f"Failures: {total - successes}/{total}")
    print(f"Success rate: {result['success_rate']:.1%}")
    print(f"Mean steps: {result['mean_steps']:.1f}")
    print("=======================================")


if __name__ == "__main__":
    main()
