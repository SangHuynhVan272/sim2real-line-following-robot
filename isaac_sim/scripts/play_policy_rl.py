#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Huynh Van Sang.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a trained checkpoint drive, and export it for firmware.

The GUI here shows the training lane, which has no tape prims -- the robot
follows a line that exists only as maths.  That is expected.  To see it drive on
the rendered track, use ``run_line_following.py``.

    conda activate env_isaaclab
    python isaac_sim/scripts/play_policy_rl.py --num_envs 16
    python isaac_sim/scripts/play_policy_rl.py --num_envs 1 --export-onnx policy.onnx

``--export-onnx`` writes the actor as a plain sequential MLP. The matching
``export_policy.py`` command validates that graph and turns it into a C header
for the ESP32-S3.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=Path, default=None,
                    help="Checkpoint to load; defaults to the newest under the experiment log dir.")
parser.add_argument("--export-onnx", type=Path, default=None)
parser.add_argument("--steps", type=int, default=4000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.line_following_rl.agents.rsl_rl_ppo_cfg import LineFollowingPPORunnerCfg  # noqa: E402
from envs.line_following_rl.env_cfg import LineFollowingEnvCfg, project_root  # noqa: E402
from checkpoint_contract import validate_checkpoint_config  # noqa: E402


def newest_checkpoint(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("model_*.pt"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"no checkpoint found under {log_dir}; train first")
    return candidates[-1]


def main() -> None:
    env_cfg = LineFollowingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = LineFollowingPPORunnerCfg()
    log_dir = project_root() / "isaac_sim/output/rl" / agent_cfg.experiment_name
    checkpoint = args_cli.checkpoint or newest_checkpoint(log_dir)

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_data, dict):
        raise ValueError("RL checkpoint must contain a dictionary payload.")
    validate_checkpoint_config(
        checkpoint_data,
        project_root() / "isaac_sim/config/default.json",
    )

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"loaded {checkpoint}")

    if args_cli.export_onnx is not None:
        # Export a CPU copy.  Moving the live actor would leave the inference
        # policy on CPU while the Isaac Lab observations remain on CUDA.
        actor = copy.deepcopy(runner.alg.policy.actor).to("cpu").eval()
        dummy = torch.zeros(1, 7)
        args_cli.export_onnx.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            actor, dummy, str(args_cli.export_onnx),
            input_names=["observation"], output_names=["duty"], opset_version=13,
        )
        print(f"exported actor to {args_cli.export_onnx} (7 -> 2, sequential MLP)")

    observation = env.get_observations()
    for _ in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            observation, _, _, _ = env.step(policy(observation))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
