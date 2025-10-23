from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, List, Optional

import imageio.v2 as imageio
import numpy as np

import gymnasium as gym
from gymnasium import Env

from .config import EnvConfig


class RenderRunner:
    """Run Gymnasium environments and persist render outputs."""

    def __init__(
        self,
        output_dir: Path | str = "renders",
        seed: Optional[int] = None,
        default_fps: int = 30,
        default_max_steps: int = 500,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.default_fps = default_fps
        self.default_max_steps = default_max_steps

    def run_many(
        self,
        configs: Iterable[EnvConfig],
        episodes: int = 1,
        max_steps: Optional[int] = None,
    ) -> None:
        for config in configs:
            self.run_single(config, episodes=episodes, max_steps=max_steps)

    def run_single(
        self,
        config: EnvConfig,
        episodes: int = 1,
        max_steps: Optional[int] = None,
    ) -> None:
        env = gym.make(config.env_id, render_mode=config.render_mode)
        try:
            fps = int(env.metadata.get("render_fps", self.default_fps))  # type: ignore[arg-type]
        except Exception:
            fps = self.default_fps

        capped_steps = max_steps or config.default_max_steps or self.default_max_steps
        episode_length = max(capped_steps, 1)

        for episode_idx in range(episodes):
            frames_rgb: List[np.ndarray] = []
            frames_text: List[str] = []
            episode_seed = self._resolve_episode_seed(episode_idx)
            step_count = 0

            env.reset(seed=episode_seed)
            self._collect_render(env, config.render_mode, frames_rgb, frames_text)

            for step in range(episode_length):
                action = self._sample_action(env)
                _, _, terminated, truncated, _ = env.step(action)
                self._collect_render(env, config.render_mode, frames_rgb, frames_text)
                step_count += 1

                if terminated or truncated:
                    break

            self._persist_episode(
                config=config,
                episode_index=episode_idx,
                fps=fps,
                rgb_frames=frames_rgb,
                text_frames=frames_text,
                steps=step_count,
            )

        env.close()

    def _collect_render(
        self,
        env: Env,
        render_mode: str,
        rgb_frames: List[np.ndarray],
        text_frames: List[str],
    ) -> None:
        render_output = env.render()
        if render_output is None:
            return

        if render_mode == "rgb_array":
            frame = self._prepare_frame(render_output)
            rgb_frames.append(frame)
        elif render_mode == "ansi":
            text_frames.append(self._stringify_ansi(render_output))
        else:
            raise ValueError(f"Unsupported render_mode '{render_mode}' for persistence.")

    def _persist_episode(
        self,
        config: EnvConfig,
        episode_index: int,
        fps: int,
        rgb_frames: List[np.ndarray],
        text_frames: List[str],
        steps: int,
    ) -> None:
        group_dir = self.output_dir / config.group / config.slug
        group_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = group_dir / f"episode_{episode_index + 1:03d}_metadata.json"
        metadata = asdict(config) | {
            "episode_index": episode_index,
            "episode_number": episode_index + 1,
            "frame_count": len(rgb_frames) + len(text_frames),
            "fps": fps,
            "steps": steps,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        if config.render_mode == "rgb_array":
            if not rgb_frames:
                return
            video_path = group_dir / f"episode_{episode_index + 1:03d}.mp4"
            imageio.mimwrite(video_path, rgb_frames, fps=fps)
        elif config.render_mode == "ansi":
            if not text_frames:
                return
            text_path = group_dir / f"episode_{episode_index + 1:03d}.txt"
            formatted = self._format_ansi_frames(text_frames)
            text_path.write_text(formatted)

    def _sample_action(self, env: Env) -> Any:
        return env.action_space.sample()

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_uint8 = self._ensure_uint8(frame)
        return self._pad_to_macro_block(frame_uint8)

    def _ensure_uint8(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        clipped = np.clip(frame, 0, 255)
        return clipped.astype(np.uint8)

    def _pad_to_macro_block(self, frame: np.ndarray, block_size: int = 16) -> np.ndarray:
        height, width = frame.shape[:2]
        pad_h = (-height) % block_size
        pad_w = (-width) % block_size

        if pad_h == 0 and pad_w == 0:
            return frame

        if frame.ndim == 2:
            pad_widths = ((0, pad_h), (0, pad_w))
        else:
            pad_widths = ((0, pad_h), (0, pad_w), (0, 0))

        return np.pad(frame, pad_widths, mode="edge")

    def _stringify_ansi(self, payload: object) -> str:
        if isinstance(payload, str):
            return payload
        if hasattr(payload, "getvalue"):
            return payload.getvalue()
        return str(payload)

    def _format_ansi_frames(self, frames: List[str]) -> str:
        lines = []
        for idx, frame in enumerate(frames):
            lines.append(f"=== Frame {idx:04d} ===")
            lines.append(frame.rstrip())
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _resolve_episode_seed(self, episode_index: int) -> Optional[int]:
        if self.seed is None:
            return None
        return self.seed + episode_index
