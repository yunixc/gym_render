#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

import gymnasium as gym  # noqa: E402
from gymnasium.envs.box2d import lunar_lander  # noqa: E402


@dataclass(frozen=True)
class LanderSnapshot:
    x: float
    y: float
    angle: float


@dataclass
class PixelTrajectory:
    xs: np.ndarray
    ys: np.ndarray
    steps: np.ndarray
    final_snapshot: LanderSnapshot
    final_center: Tuple[float, float]


WorldTrajectory = List[LanderSnapshot]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an overlay image of multiple random LunarLander trajectories "
            "for presentation-ready visuals."
        )
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to sample.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Max steps per episode before truncating the rollout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional base seed for reproducible trajectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("renders/lunarlander_overlay.png"),
        help="Path to the generated overlay image.",
    )
    parser.add_argument(
        "--episode-colormap",
        default="Grays_r",
        help="Colormap used to differentiate episodes.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.2,
        help="Width of trajectory lines in the overlay.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Global opacity scale (multiplies the per-step alpha gradient).",
    )
    parser.add_argument(
        "--terrain-seed",
        type=int,
        help="Fix the terrain across episodes by resetting with this seed each time.",
    )
    parser.add_argument(
        "--action-seed",
        type=int,
        help="Seed for the action sampler to keep behavior reproducible while terrain stays fixed.",
    )
    return parser.parse_args()


def collect_trajectories(
    env_id: str,
    episodes: int,
    max_steps: int,
    seed: int | None,
    terrain_seed: int | None = None,
    action_seed: int | None = None,
) -> Tuple[np.ndarray, Sequence[WorldTrajectory]]:
    trajectories: List[WorldTrajectory] = []
    background: np.ndarray | None = None

    base_action_rng = np.random.default_rng(action_seed)

    if terrain_seed is None:
        env = gym.make(env_id, render_mode="rgb_array")
        for episode_idx in range(episodes):
            env.reset(seed=(seed + episode_idx) if seed is not None else None)

            frame = env.render()
            if background is None and frame is not None:
                background = _prepare_background(frame)

            trajectory: WorldTrajectory = [_lander_snapshot(env)]

            episode_rng_seed = base_action_rng.integers(0, 2**32) if action_seed is not None else None
            action_rng = np.random.default_rng(episode_rng_seed)

            for _ in range(max_steps):
                action = _sample_action(env.action_space, action_rng)
                _, _, terminated, truncated, _ = env.step(action)
                trajectory.append(_lander_snapshot(env))

                frame = env.render()
                if background is None and frame is not None:
                    background = _prepare_background(frame)
                if terminated or truncated:
                    break

            trajectories.append(trajectory)

        env.close()
    else:
        for episode_idx in range(episodes):
            env = gym.make(env_id, render_mode="rgb_array")
            env.reset(seed=terrain_seed)

            frame = env.render()
            if background is None and frame is not None:
                background = _prepare_background(frame)

            trajectory: WorldTrajectory = [_lander_snapshot(env)]

            episode_rng_seed = base_action_rng.integers(0, 2**32) if action_seed is not None else None
            action_rng = np.random.default_rng(episode_rng_seed)

            for _ in range(max_steps):
                action = _sample_action(env.action_space, action_rng)
                _, _, terminated, truncated, _ = env.step(action)
                trajectory.append(_lander_snapshot(env))

                if terminated or truncated:
                    break

            trajectories.append(trajectory)
            env.close()

    if background is None:
        background = np.zeros((lunar_lander.VIEWPORT_H, lunar_lander.VIEWPORT_W, 3), dtype=np.uint8)

    return background, trajectories


def _lander_snapshot(env: gym.Env) -> LanderSnapshot:
    lander = env.unwrapped.lander
    return LanderSnapshot(
        x=float(lander.position.x),
        y=float(lander.position.y),
        angle=float(lander.angle),
    )


def _prepare_background(frame: np.ndarray) -> np.ndarray:
    img = frame.astype(np.float32)
    img *= 0.45
    return np.clip(img, 0, 255).astype(np.uint8)


def _segment_colours(
    base_rgba: np.ndarray,
    progress: np.ndarray,
    alpha_scale: float,
    shade_strength: float = 0.55,
    alpha_range: Tuple[float, float] = (0.05, 0.5),
) -> np.ndarray:
    """Compute RGBA for each segment by shading and fading based on progress."""

    p = np.clip(progress, 0.0, 1.0)

    alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * p
    alpha = np.clip(alpha * alpha_scale, 0.0, 1.0)

    base_rgb = np.broadcast_to(base_rgba[:3], (p.shape[0], 3))
    shade = 1.0 - shade_strength * (1.0 - p[:, None])
    rgb = np.clip(base_rgb * shade, 0.0, 1.0)

    return np.concatenate([rgb, alpha[:, None]], axis=1)


def _smooth_series(data: np.ndarray, window: int = 5) -> np.ndarray:
    if data.size <= 2 or window <= 1:
        return data
    window = max(1, window | 1)  # ensure odd
    pad = window // 2
    padded = np.pad(data, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: data.size]


def _densify_path(
    xs: np.ndarray,
    ys: np.ndarray,
    steps: np.ndarray,
    factor: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xs.size < 2 or factor < 1:
        return xs, ys, steps
    new_xs = [xs[0]]
    new_ys = [ys[0]]
    new_steps = [steps[0]]

    for idx in range(1, xs.size):
        prev_x, prev_y, prev_step = xs[idx - 1], ys[idx - 1], steps[idx - 1]
        curr_x, curr_y, curr_step = xs[idx], ys[idx], steps[idx]
        for sub in range(1, factor + 1):
            weight = sub / (factor + 1)
            new_xs.append(prev_x + weight * (curr_x - prev_x))
            new_ys.append(prev_y + weight * (curr_y - prev_y))
            new_steps.append(prev_step + weight * (curr_step - prev_step))
        new_xs.append(curr_x)
        new_ys.append(curr_y)
        new_steps.append(curr_step)

    return np.array(new_xs), np.array(new_ys), np.array(new_steps)


def _sample_action(space, rng: np.random.Generator):
    try:
        seed = int(rng.integers(0, 2**32))
        return space.sample(seed=seed)
    except Exception:
        return space.sample()


def trajectories_to_pixels(
    trajectories: Sequence[WorldTrajectory],
) -> List[PixelTrajectory]:
    pixel_trajs: List[PixelTrajectory] = []
    height = lunar_lander.VIEWPORT_H

    for trajectory in trajectories:
        world_x = np.array([snap.x for snap in trajectory], dtype=np.float32)
        world_y = np.array([snap.y for snap in trajectory], dtype=np.float32)
        steps = np.linspace(0.0, 1.0, len(world_x))

        world_x = _smooth_series(world_x)
        world_y = _smooth_series(world_y)
        world_x, world_y, steps = _densify_path(world_x, world_y, steps)

        pixels_x = world_x * lunar_lander.SCALE
        pixels_y = _smooth_series(height - world_y * lunar_lander.SCALE)

        if len(steps) > 1:
            steps = (steps - steps.min()) / (steps.max() - steps.min())

        final_snapshot = trajectory[-1]
        final_center = (
            final_snapshot.x * lunar_lander.SCALE,
            height - final_snapshot.y * lunar_lander.SCALE,
        )

        pixel_trajs.append(
            PixelTrajectory(
                xs=pixels_x,
                ys=pixels_y,
                steps=steps,
                final_snapshot=final_snapshot,
                final_center=final_center,
            )
        )

    return pixel_trajs


def _pad_canvas_and_offset(
    background: np.ndarray,
    pixel_trajs: Sequence[PixelTrajectory],
) -> Tuple[np.ndarray, List[PixelTrajectory]]:
    """Pad background to fit all trajectories and offset coordinates accordingly.

    Pads with black outside the original frame.
    """
    h, w = background.shape[:2]
    all_x = np.concatenate([traj.xs for traj in pixel_trajs]) if pixel_trajs else np.array([0])
    all_y = np.concatenate([traj.ys for traj in pixel_trajs]) if pixel_trajs else np.array([0])

    min_x = float(np.floor(all_x.min()))
    max_x = float(np.ceil(all_x.max()))
    min_y = float(np.floor(all_y.min()))
    max_y = float(np.ceil(all_y.max()))

    left_pad = int(max(0.0, -min_x))
    top_pad = int(max(0.0, -min_y))
    right_pad = int(max(0.0, max_x - (w - 1)))
    bottom_pad = int(max(0.0, max_y - (h - 1)))

    if left_pad == 0 and right_pad == 0 and top_pad == 0 and bottom_pad == 0:
        return background, [
            PixelTrajectory(
                xs=traj.xs.copy(),
                ys=traj.ys.copy(),
                steps=traj.steps.copy(),
                final_snapshot=traj.final_snapshot,
                final_center=traj.final_center,
            )
            for traj in pixel_trajs
        ]

    new_h = h + top_pad + bottom_pad
    new_w = w + left_pad + right_pad
    padded = np.zeros((new_h, new_w, 3), dtype=background.dtype)
    padded[top_pad : top_pad + h, left_pad : left_pad + w] = background

    adjusted: List[PixelTrajectory] = []
    for traj in pixel_trajs:
        adjusted.append(
            PixelTrajectory(
                xs=traj.xs + left_pad,
                ys=traj.ys + top_pad,
                steps=traj.steps,
                final_snapshot=traj.final_snapshot,
                final_center=(traj.final_center[0] + left_pad, traj.final_center[1] + top_pad),
            )
        )

    return padded, adjusted


def _draw_final_lander(
    ax: matplotlib.axes.Axes,
    traj: PixelTrajectory,
    base_rgb: np.ndarray,
    alpha_scale: float,
) -> None:
    snapshot = traj.final_snapshot
    center_x, center_y = traj.final_center
    angle = snapshot.angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    # LANDER_POLY is defined in pixel units; convert to world space, rotate, then map back to pixels.
    local_world = np.array(lunar_lander.LANDER_POLY, dtype=np.float32) / lunar_lander.SCALE
    rotated_world = local_world @ rotation.T
    rotated_pixels = rotated_world * lunar_lander.SCALE
    polygon_xy = np.column_stack(
        [
            center_x + rotated_pixels[:, 0],
            center_y - rotated_pixels[:, 1],
        ]
    )

    patch = Polygon(
        polygon_xy,
        closed=True,
        facecolor=base_rgb,
        edgecolor="white",
        linewidth=0.6,
        alpha=min(1.0, max(0.0, alpha_scale)),
        zorder=4,
    )
    ax.add_patch(patch)


def save_overlay(
    background: np.ndarray,
    pixel_trajs: Sequence[PixelTrajectory],
    output_path: Path,
    episode_colormap: str,
    line_width: float,
    alpha: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = background.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=220)
    ax.imshow(background, origin="upper")

    episodes_cmap = plt.get_cmap(episode_colormap)
    num = max(len(pixel_trajs) - 1, 1)
    last_traj = pixel_trajs[-1] if pixel_trajs else None
    last_color = None

    for idx, traj in enumerate(pixel_trajs):
        xs, ys, steps = traj.xs, traj.ys, traj.steps
        if len(xs) < 2:
            continue
        episode_norm = idx / num
        base_color = np.array(episodes_cmap(episode_norm))
        base_color[:3] = np.clip(base_color[:3], 0.2, 1.0)

        segments = np.stack([np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1)
        segment_progress = 0.5 * (steps[:-1] + steps[1:])
        colors = _segment_colours(base_color, segment_progress, alpha_scale=alpha)
        colors = np.asarray(colors, dtype=float)
        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=line_width,
            capstyle="round",
            joinstyle="round",
        )
        # Ensure per-segment RGBA is respected (do not override with a scalar alpha)
        lc.set_alpha(None)
        lc.set_zorder(3)
        ax.add_collection(lc)

        if traj is last_traj:
            last_color = base_color
    ax.set_axis_off()

    if last_traj is not None and last_color is not None:
        _draw_final_lander(ax, last_traj, last_color[:3], alpha)

    # Episode colorbar (vertical on the right), matched to the padded axis height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    episode_count = max(len(pixel_trajs), 1)
    episode_sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=1, vmax=episode_count),
        cmap=episodes_cmap,
    )
    episode_cbar = fig.colorbar(episode_sm, cax=cax, orientation="vertical")
    episode_cbar.set_label("Episode index", fontsize=8)
    episode_cbar.ax.tick_params(labelsize=7)
    if len(pixel_trajs) <= 1:
        episode_cbar.set_ticks([1])

    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    background, trajectories = collect_trajectories(
        env_id="LunarLander-v3",
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        terrain_seed=args.terrain_seed,
        action_seed=args.action_seed,
    )
    pixel_trajs = trajectories_to_pixels(trajectories)
    background, pixel_trajs = _pad_canvas_and_offset(background, pixel_trajs)
    save_overlay(
        background=background,
        pixel_trajs=pixel_trajs,
        output_path=args.output,
        episode_colormap=args.episode_colormap,
        line_width=args.line_width,
        alpha=args.alpha,
    )
    print(f"Overlay saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
