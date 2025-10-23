from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class EnvConfig:
    """Immutable description of an environment to render."""

    name: str
    env_id: str
    group: str
    render_mode: str = "rgb_array"
    default_max_steps: int = 500
    description: str | None = None

    @property
    def slug(self) -> str:
        """Filesystem-friendly identifier."""
        return self.env_id.replace("-", "_").lower()


def _classic_control() -> List[EnvConfig]:
    return [
        EnvConfig("Acrobot", "Acrobot-v1", "classic_control", default_max_steps=500),
        EnvConfig("CartPole", "CartPole-v1", "classic_control", default_max_steps=500),
        EnvConfig(
            "MountainCarContinuous",
            "MountainCarContinuous-v0",
            "classic_control",
            default_max_steps=500,
        ),
        EnvConfig("MountainCar", "MountainCar-v0", "classic_control", default_max_steps=500),
        EnvConfig("Pendulum", "Pendulum-v1", "classic_control", default_max_steps=500),
    ]


def _box2d() -> List[EnvConfig]:
    return [
        EnvConfig("BipedalWalker", "BipedalWalker-v3", "box2d", default_max_steps=1600),
        EnvConfig("CarRacing", "CarRacing-v2", "box2d", default_max_steps=1000),
        EnvConfig("LunarLander", "LunarLander-v3", "box2d", default_max_steps=1000),
    ]


def _toy_text() -> List[EnvConfig]:
    return [
        EnvConfig(
            "Blackjack",
            "Blackjack-v1",
            "toy_text",
            render_mode="ansi",
            default_max_steps=200,
        ),
        EnvConfig("Taxi", "Taxi-v3", "toy_text", render_mode="ansi", default_max_steps=200),
        EnvConfig(
            "CliffWalking",
            "CliffWalking-v0",
            "toy_text",
            render_mode="ansi",
            default_max_steps=200,
        ),
        EnvConfig(
            "FrozenLake",
            "FrozenLake-v1",
            "toy_text",
            render_mode="ansi",
            default_max_steps=200,
        ),
    ]


def _mujoco() -> List[EnvConfig]:
    return [
        EnvConfig("Ant", "Ant-v4", "mujoco", default_max_steps=1000),
        EnvConfig("HalfCheetah", "HalfCheetah-v4", "mujoco", default_max_steps=1000),
        EnvConfig("Hopper", "Hopper-v4", "mujoco", default_max_steps=1000),
        EnvConfig("Humanoid", "Humanoid-v4", "mujoco", default_max_steps=1000),
        EnvConfig("HumanoidStandup", "HumanoidStandup-v4", "mujoco", default_max_steps=1000),
        EnvConfig(
            "InvertedDoublePendulum",
            "InvertedDoublePendulum-v4",
            "mujoco",
            default_max_steps=1000,
        ),
        EnvConfig("InvertedPendulum", "InvertedPendulum-v4", "mujoco", default_max_steps=1000),
        EnvConfig("Pusher", "Pusher-v4", "mujoco", default_max_steps=1000),
        EnvConfig("Reacher", "Reacher-v4", "mujoco", default_max_steps=200),
        EnvConfig("Swimmer", "Swimmer-v4", "mujoco", default_max_steps=1000),
        EnvConfig("Walker2d", "Walker2d-v4", "mujoco", default_max_steps=1000),
    ]


ENV_GROUPS: Dict[str, List[EnvConfig]] = {
    "classic_control": _classic_control(),
    "box2d": _box2d(),
    "toy_text": _toy_text(),
    "mujoco": _mujoco(),
}

ENV_CONFIGS_BY_ID: Dict[str, EnvConfig] = {
    config.env_id: config for configs in ENV_GROUPS.values() for config in configs
}


def iter_configs(group: str | None = None) -> Iterable[EnvConfig]:
    """Yield configs for the specified group or all groups."""
    if group is None or group == "all":
        for configs in ENV_GROUPS.values():
            yield from configs
    else:
        yield from ENV_GROUPS[group]


def available_groups() -> Tuple[str, ...]:
    return tuple(ENV_GROUPS.keys())
