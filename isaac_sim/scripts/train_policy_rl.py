#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Huynh Van Sang.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the direct-duty line-following policy with PPO.

This follows the standard Isaac Lab training layout: ``AppLauncher`` at the very
top of the file, ``--headless`` to turn the GUI off, and a single process that
owns every environment at once. Contrast it with ``evaluate_line_following.py``,
which forks one process per episode -- that lane rebuilds the *scene geometry*
for every episode, while this one keeps one scene and only resets state, so it
has nothing to fork for.

PPO here is warm-started from a behaviour-cloned actor (``--init-actor``). That
is not a shortcut around RL; it is what makes the problem learnable. Turning
this chassis requires reversing one side, which is a discrete regime change
rather than a smooth control input, and random exploration almost never
discovers it in the states where it helps. Starting from a cloned analytic
controller places the policy where that behaviour already exists, after which
PPO can optimize completion, robustness and smooth control under randomization.

    conda activate env_isaaclab
    python isaac_sim/scripts/train_policy_rl.py --headless --num_envs 1024
    python isaac_sim/scripts/train_policy_rl.py --num_envs 16   # watch it in the GUI
    tensorboard --logdir isaac_sim/output/rl/line_following

A trained checkpoint is not a result.  Validate it in the lane that has a real
camera before believing anything:

    python isaac_sim/scripts/run_line_following.py --policy-backend rl --checkpoint ...
    python isaac_sim/scripts/evaluate_line_following.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import types

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--random-init-episode-age",
    action="store_true",
    help=(
        "Start each parallel environment at a random episode age. Disabled by default "
        "because this task requires a complete traversal from the start pose; use only "
        "to reproduce a legacy/tutorial rollout."
    ),
)
parser.add_argument("--blind-duty-scale", type=float, default=None,
                    help="Duty fraction allowed before the first detection; 0.0 restores the "
                         "original 'never drive blind' firmware rule.")
parser.add_argument(
    "--init-actor",
    type=Path,
    default=None,
    help="Actor-only warm start from a BC/RSL-RL checkpoint; critic and optimizer stay fresh.",
)
parser.add_argument(
    "--bc-anchor-weight",
    type=float,
    default=None,
    help=(
        "Override rl.ppo.bc_anchor_weight for this run. Use 0 only when the warm-start "
        "actor is deliberately untrusted and PPO must be free to leave it; rendered "
        "camera evaluation still measures the resulting checkpoint before export."
    ),
)
parser.add_argument("--output-dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab modules may only be imported once SimulationApp exists.
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.line_following_rl.agents.rsl_rl_ppo_cfg import LineFollowingPPORunnerCfg  # noqa: E402
from envs.line_following_rl.bc_anchored_ppo import BCAnchoredPPO  # noqa: E402
from envs.line_following_rl.env_cfg import LineFollowingEnvCfg, project_root  # noqa: E402
from checkpoint_contract import config_contract, validate_checkpoint_config  # noqa: E402


def load_actor_warm_start(
    actor: torch.nn.Module,
    checkpoint_path: Path,
    config_path: Path,
) -> None:
    """Load only the frozen actor layout, never BC's critic or optimizer state."""
    if not checkpoint_path.is_file():
        raise ValueError(f"Warm-start checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Warm-start checkpoint does not contain an RSL-RL model_state_dict.")
    validate_checkpoint_config(checkpoint, config_path, label="Warm-start checkpoint")
    actor_state = {
        name.removeprefix("actor."): value
        for name, value in checkpoint["model_state_dict"].items()
        if name.startswith("actor.")
    }
    expected = set(actor.state_dict())
    if set(actor_state) != expected:
        raise ValueError(
            "Warm-start actor is not the frozen 7->64(ELU)->64(ELU)->2 layout: "
            f"expected {sorted(expected)}, got {sorted(actor_state)}."
        )
    if not all(torch.isfinite(value).all() for value in actor_state.values()):
        raise ValueError("Warm-start actor contains non-finite weights.")
    actor.load_state_dict(actor_state, strict=True)


def main() -> None:
    config_path = project_root() / "isaac_sim/config/default.json"
    env_cfg = LineFollowingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.blind_duty_scale is not None:
        env_cfg.actions.duty.blind_duty_scale = args_cli.blind_duty_scale

    agent_cfg = LineFollowingPPORunnerCfg()
    ppo_config = env_cfg.line_config["rl"]["ppo"]
    agent_cfg.policy.init_noise_std = float(ppo_config["init_noise_std"])
    agent_cfg.algorithm.entropy_coef = float(ppo_config["entropy_coef"])
    agent_cfg.algorithm.learning_rate = float(ppo_config["learning_rate"])
    agent_cfg.algorithm.schedule = str(ppo_config["schedule"])
    # gamma lives in the config because the terminal weights are derived from it:
    # see the discounted-ceiling guard in LineFollowingEnvCfg.__post_init__.
    agent_cfg.algorithm.gamma = float(ppo_config["gamma"])
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    log_dir = args_cli.output_dir or project_root() / "isaac_sim/output/rl" / agent_cfg.experiment_name
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("=" * 70)
    print(f"environments      : {env.num_envs}")
    print(f"observation space : {env.observation_space}   (frozen 7-value ABI)")
    print(f"action space      : {env.action_space}        (duty left/right)")
    print(f"physics rate      : {1.0 / env_cfg.sim.dt:.0f} Hz")
    print(f"policy rate       : {1.0 / (env_cfg.sim.dt * env_cfg.decimation):.0f} Hz")
    print(f"decimation        : {env_cfg.decimation} physics ticks / policy tick")
    print(f"camera rate       : {env_cfg.line_config['physics']['camera_hz']:.0f} Hz "
          f"(stride {env_cfg.camera_stride} policy ticks)")
    print(f"blind duty scale  : {env_cfg.actions.duty.blind_duty_scale}")
    print(f"steps per update  : {agent_cfg.num_steps_per_env} x {env.num_envs} = "
          f"{agent_cfg.num_steps_per_env * env.num_envs}")
    print(f"random episode age: {args_cli.random_init_episode_age}")
    print(f"log directory     : {log_dir}")
    print("=" * 70)

    env = RslRlVecEnvWrapper(env)
    if args_cli.init_actor is not None:
        # OnPolicyRunner resolves algorithm names in its own module namespace.
        # Register the local class there rather than modifying site-packages.
        import rsl_rl.runners.on_policy_runner as on_policy_runner

        on_policy_runner.BCAnchoredPPO = BCAnchoredPPO
        agent_cfg.algorithm.class_name = "BCAnchoredPPO"
        anchor_weight = (
            float(ppo_config["bc_anchor_weight"])
            if args_cli.bc_anchor_weight is None
            else float(args_cli.bc_anchor_weight)
        )
        if anchor_weight < 0.0:
            raise ValueError("--bc-anchor-weight must be non-negative.")
        agent_cfg.algorithm.bc_anchor_weight = anchor_weight
        agent_cfg.algorithm.bc_anchor_batch_size = int(ppo_config["bc_anchor_batch_size"])
        agent_cfg.algorithm.bc_anchor_epochs = int(ppo_config["bc_anchor_epochs"])
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    checkpoint_infos: dict[str, object] = {
        "kind": "ppo",
        **config_contract(config_path),
        "seed": args_cli.seed,
        "blind_duty_scale": float(env_cfg.actions.duty.blind_duty_scale),
        "init_actor": None if args_cli.init_actor is None else str(args_cli.init_actor.resolve()),
        "init_actor_sha256": None if args_cli.init_actor is None else hashlib.sha256(
            args_cli.init_actor.read_bytes()
        ).hexdigest(),
        "bc_anchor_weight": None if args_cli.init_actor is None else float(anchor_weight),
        "bc_anchor_batch_size": None if args_cli.init_actor is None else int(
            ppo_config["bc_anchor_batch_size"]
        ),
        "bc_anchor_epochs": None if args_cli.init_actor is None else int(
            ppo_config["bc_anchor_epochs"]
        ),
    }
    original_save = runner.save

    def save_with_contract(self, path: str, infos: dict | None = None) -> None:
        merged = dict(checkpoint_infos)
        if infos:
            merged.update(infos)
        original_save(path, merged)

    # RSL-RL calls self.save() for periodic and final checkpoints. Bind the
    # wrapper on this runner so interrupted runs also leave traceable models.
    runner.save = types.MethodType(save_with_contract, runner)
    if args_cli.init_actor is not None:
        load_actor_warm_start(runner.alg.policy.actor, args_cli.init_actor, config_path)
        if not isinstance(runner.alg, BCAnchoredPPO):
            raise RuntimeError("PPO warm start did not construct BCAnchoredPPO.")
        runner.alg.set_actor_anchor()
        print(f"actor warm start  : {args_cli.init_actor.resolve()}")
        print(f"BC anchor         : weight={runner.alg.bc_anchor_weight:g}, "
              f"batch={runner.alg.bc_anchor_batch_size}, epochs={runner.alg.bc_anchor_epochs}")
    # Random initial episode ages are useful for steady-state tasks such as
    # balancing.  Line following is a start-to-finish task: an env placed near
    # the configured timeout cannot complete the full route under a correct
    # policy, which turns valid completion trajectories into artificial
    # timeouts.  Resets during training are still naturally asynchronous.
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=args_cli.random_init_episode_age,
    )
    env.close()
    print(f"\ntraining complete; checkpoints in {log_dir}")
    print("Training is complete. Run the rendered-camera evaluation next to measure checkpoint performance.")


if __name__ == "__main__":
    main()
    simulation_app.close()
