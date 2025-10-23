#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gym_render import ENV_CONFIGS_BY_ID, ENV_GROUPS, RenderRunner
from gym_render.config import available_groups, iter_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Gymnasium environments and persist the outputs to disk.",
    )
    parser.add_argument(
        "--group",
        choices=(*available_groups(), "all"),
        default="all",
        help="Subset of environments to render.",
    )
    parser.add_argument(
        "--env-id",
        help="Render a single Gymnasium environment id. Overrides --group when provided.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record per environment.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional step cap per episode. Defaults to the config default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional base seed to make runs reproducible. Each episode increments the seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders"),
        help="Directory where rendered artifacts are written.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available environment ids and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        for group_name, configs in ENV_GROUPS.items():
            print(f"[{group_name}]")
            for cfg in configs:
                print(f"  - {cfg.env_id}")
        return 0

    runner = RenderRunner(output_dir=args.output_dir, seed=args.seed)

    if args.env_id:
        config = ENV_CONFIGS_BY_ID.get(args.env_id)
        if config is None:
            available = ", ".join(sorted(ENV_CONFIGS_BY_ID))
            raise SystemExit(f"Unknown env-id '{args.env_id}'. Available: {available}")
        runner.run_single(config, episodes=args.episodes, max_steps=args.max_steps)
        return 0

    configs = iter_configs(None if args.group == "all" else args.group)
    runner.run_many(configs, episodes=args.episodes, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
