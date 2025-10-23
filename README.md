# Gymnasium Rendering Toolkit

A ready-to-use Conda environment and a small modular codebase designed to capture rendered 
rollouts for a curated list of Gymnasium environments spanning
Classic Control, Box2D, Toy Text, and MuJoCo tasks.

## 1. Conda environment

```bash
conda env create -f environment.yml
conda activate gymnasium
python -m pip install -r requirements.txt
```

The environment installs Gymnasium with all required extras, MuJoCo, Box2D tooling, and
ImageIO for recording RGB videos. The `swig` dependency is included so that Box2D builds
work reliably on fresh systems.

### Alternative: bare pip install

If you prefer to manage Python yourself, install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -m pip install -r requirements.txt
```

## 2. MuJoCo / GLFW notes

- MuJoCo requires an OpenGL context. On Linux, ensure that `libglfw` (GLFW3) is present. For
  example, `sudo apt install libglfw3` (Ubuntu/Debian). On macOS use `brew install glfw`.
- When running headless (e.g., via SSH), set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` before
  launching the renderer: `MUJOCO_GL=egl python scripts/render_envs.py --env-id Ant-v4`.
- Gymnasium may print a warning about older environment versions (e.g., `Ant-v4`). The runner
  still works, but consider upgrading to the latest IDs if you need parity with upstream.

## 3. Rendering script

Use the `render_envs.py` helper to capture episodes to disk. Artifacts end up under
`renders/<group>/<env_id_slug>/`.

```bash
# Render every environment in the classic_control catalog for 2 episodes each
python scripts/render_envs.py --group classic_control --episodes 2

# Capture a Box2D run of LunarLander with short clips
python scripts/render_envs.py --env-id LunarLander-v3 --episodes 3 --max-steps 300

# Render a single environment id with a deterministic seed and custom step cap
python scripts/render_envs.py --env-id Ant-v4 --seed 42 --max-steps 750

# List all known environment ids
python scripts/render_envs.py --list
```

RGB environments are saved as `.mp4` videos alongside a JSON metadata file describing each
episode. Toy Text environments render textual frames to `.txt` files.

### LunarLander trajectory overlay

Generate a presentation-friendly overlay of multiple LunarLander trajectories:

```bash
python scripts/lunarlander_overlay.py --episodes 20 --max-steps 1000 \
    --terrain-seed 2 --episode-colormap Grays_r --alpha 0.4 \
    --output renders/lunarlander_overlay.png
```

The overlay records the centre of the lander, renders every trail with a
transparent → opaque gradient, and draws the final spacecraft hull for the last episode.
Passing `--terrain-seed` recreates the exact same landscape for each run; omit it to let
Gymnasium randomise the terrain every episode. If you also set `--action-seed`, the sampled
actions become reproducible while the terrain stays fixed. Adjust `--line-width`,
`--alpha`, or the colormap to taste. To visualise a trained policy, adapt the script to
draw actions from your controller instead of `action_space.sample()`.

### Rendering with a trained controller

You can hook up your own policy or controller by subclassing `RenderRunner` and overriding
`_sample_action`, or by providing a wrapper that feeds actions from your model. For example:

```python
from gym_render.runner import RenderRunner
from my_agent import load_agent
import numpy as np

class PolicyRunner(RenderRunner):
    def __init__(self, policy, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy

    def _sample_action(self, env):
        obs = env.unwrapped.state if hasattr(env.unwrapped, "state") else None
        if obs is None:
            obs, _ = env.last_observation  # adjust for your API
        return self.policy(obs)

agent = load_agent("models/lunarlander_policy.pt")
runner = PolicyRunner(agent, output_dir="renders/agent_runs")
config = ENV_CONFIGS_BY_ID["LunarLander-v3"]
runner.run_single(config, episodes=5)
```

Adapt the observation retrieval to your environment; the runner only expects `_sample_action`
to return a valid action for the underlying Gymnasium space.

## 4. Environment catalog

| Group            | Environments                                                                 |
| ---------------- | ----------------------------------------------------------------------------- |
| `classic_control`| Acrobot-v1, CartPole-v1, MountainCarContinuous-v0, MountainCar-v0, Pendulum-v1 |
| `box2d`          | BipedalWalker-v3, CarRacing-v2, LunarLander-v3                                |
| `toy_text`       | Blackjack-v1, Taxi-v3, CliffWalking-v0, FrozenLake-v1                         |
| `mujoco`         | Ant-v4, HalfCheetah-v4, Hopper-v4, Humanoid-v4, HumanoidStandup-v4, InvertedDoublePendulum-v4, InvertedPendulum-v4, Pusher-v4, Reacher-v4, Swimmer-v4, Walker2d-v4 |

## 5. Notes

- Videos are encoded with `imageio-ffmpeg`. Ensure `ffmpeg` binaries are discoverable on your
  system (installed automatically when using the provided Conda environment).
- MuJoCo environments require hardware acceleration for smooth rendering. Consider lowering
  `--max-steps` if you only need short clips.
- Frames are zero-padded to the nearest 16×16 macro block before encoding so that FFMPEG
  produces standards-compliant videos without resizing warnings.
- The trajectory overlay helper relies on Matplotlib. If you are using a custom environment,
  set `MPLBACKEND=Agg` or adjust the script if you need interactive figures.
