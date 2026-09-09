#!/usr/bin/env python3
"""Collect analytical-teacher rollouts and behavior-clone the deployable actor.

The teacher is ``teacher_batch.analytic_duty`` -- the batched controller whose
parity with the analytical camera lane is already checked. Most examples are
collected by stepping the real Isaac Lab environment. An explicit uniform-ABI
supplement covers corner errors that a mostly straight rollout under-samples;
without it the dataset is almost all straight track and the cloned actor never
learns the corners.

**The regression happens before ``tanh``.** The deployed actor computes
``tanh(net(obs))``, so a tempting loss is
``mse(tanh(net(obs)), teacher_duty)``. Near saturation, however, the derivative
through ``tanh`` becomes small and optimization can stall with poorly fitted
steering states. Severe saturation can also round the float32 output to exactly
+/-1, eliminating useful local gradient information.

Regressing ``atanh(teacher_duty)`` against the raw network output keeps every
gradient alive. The actor itself is unchanged -- still ``tanh(net(obs))`` -- only
the regression target moves. Validation is still reported in duty space so the
numbers stay comparable.

A useful warning sign: a validation max-absolute-error near 2.0 means a **sign
flip**, not blur. Two duties in [-1, 1] can only differ by 2.0 if they point in
opposite directions.

The resulting checkpoint has normal RSL-RL keys.  It can therefore be used by
the camera evaluator immediately and passed to ``train_policy_rl.py
--init-actor`` without loading a BC critic or optimizer.

    conda activate env_isaaclab
    python isaac_sim/scripts/train_policy_bc.py --headless --num-envs 1024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--rollout-steps", type=int, default=512,
                    help="Teacher policy ticks to collect from every environment.")
parser.add_argument(
    "--uniform-samples", type=int, default=0,
    help=(
        "Additional synthetic ABI samples spanning the full steering domain. "
        "These prevent straight-track rollout frames from hiding large corner errors."
    ),
)
parser.add_argument(
    "--uniform-heading-max-rad", type=float, default=0.8,
    help="Absolute e_theta bound used by --uniform-samples.",
)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--learning-rate", type=float, default=1.0e-3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--blind-duty-scale", type=float, default=None)
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
from envs.line_following_rl.env_cfg import LineFollowingEnvCfg, project_root  # noqa: E402
from envs.line_following_rl.mdp.scene_state import get_scenario  # noqa: E402
from envs.line_following_rl.teacher_batch import analytic_duty  # noqa: E402
from checkpoint_contract import config_contract  # noqa: E402


def policy_observation(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the one frozen policy group and reject accidental ABI changes."""
    values = observation.get("policy")
    if values is None or values.ndim != 2 or values.shape[1] != 7:
        actual = None if values is None else tuple(values.shape)
        raise RuntimeError(f"Expected frozen policy observation shape [N, 7], got {actual}.")
    return values


def actor_layout_is_frozen(actor: torch.nn.Module) -> None:
    """Fail before training if a config edit changed the exportable network."""
    expected = {
        "0.weight": (64, 7), "0.bias": (64,),
        "2.weight": (64, 64), "2.bias": (64,),
        "4.weight": (2, 64), "4.bias": (2,),
    }
    actual = {name: tuple(value.shape) for name, value in actor.state_dict().items()}
    if actual != expected:
        raise RuntimeError(
            "BC requires the deployable 7->64(ELU)->64(ELU)->2 actor; "
            f"got {actual}."
        )


def collect_teacher_rollout(
    env: ManagerBasedRLEnv,
    config: dict,
    rollout_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step the physical batched environment under the analytical teacher."""
    if rollout_steps < 1:
        raise ValueError("--rollout-steps must be positive.")
    observation, _ = env.reset()
    sample_count = env.num_envs * rollout_steps
    observations = torch.empty(sample_count, 7, device=env.device)
    targets = torch.empty(sample_count, 2, device=env.device)
    cursor = 0
    # tanh is the action map in DutyAction.  The tiny margin makes atanh finite
    # if a future teacher asks for exactly +/- max duty.
    action_limit = 1.0 - 1.0e-6
    # Do not use inference_mode here: Isaac Lab keeps mutable state tensors
    # created during env.step(), and its next reset must be allowed to update
    # them.  no_grad saves the rollout graph without turning those buffers into
    # immutable inference tensors.
    with torch.no_grad():
        for step in range(rollout_steps):
            values = policy_observation(observation)
            scenario = get_scenario(env)
            target_duty = analytic_duty(
                values,
                config,
                scenario["wheel_radius_scale"],
                scenario["track_width_scale"],
            )
            if not torch.isfinite(values).all() or not torch.isfinite(target_duty).all():
                raise RuntimeError(f"Non-finite teacher sample at rollout step {step}.")
            next_cursor = cursor + env.num_envs
            observations[cursor:next_cursor].copy_(values)
            targets[cursor:next_cursor].copy_(target_duty)
            # The environment applies its identical blind-duty and slew limits.
            teacher_raw = torch.atanh(target_duty.clamp(-action_limit, action_limit))
            observation, _, _, _, _ = env.step(teacher_raw)
            cursor = next_cursor
    return observations, targets


def collect_uniform_teacher_samples(
    config: dict,
    sample_count: int,
    heading_max_rad: float,
    device: torch.device | str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cover the deployable ABI domain instead of over-sampling straight driving.

    The analytical teacher uses only ``e_y``, ``e_theta`` and confidence.  RPM
    and previous duty are nevertheless randomized across their firmware ranges
    so supervised training learns that those nuisance columns do not change the
    teacher action.  Scenario geometry is intentionally nominal: randomized
    wheel radius and track width are hidden variables that the seven-value ABI
    cannot observe, so fitting per-episode privileged targets would inject
    irreducible label noise.
    """
    if sample_count < 0 or heading_max_rad <= 0.0:
        raise ValueError("--uniform-samples must be non-negative and heading bound positive.")
    if sample_count == 0:
        return (
            torch.empty(0, 7, device=device),
            torch.empty(0, 2, device=device),
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    unit = torch.rand(sample_count, 7, device=device, generator=generator)
    observations = torch.empty_like(unit)
    observations[:, 0] = unit[:, 0] * 2.0 - 1.0
    observations[:, 1] = (unit[:, 1] * 2.0 - 1.0) * float(heading_max_rad)
    observations[:, 2] = unit[:, 2]
    observations[:, 3:5] = unit[:, 3:5] * 500.0 - 250.0
    observations[:, 5:7] = unit[:, 5:7] * 2.0 - 1.0
    with torch.no_grad():
        targets = analytic_duty(observations, config)
    return observations, targets


def train_actor(
    actor: torch.nn.Module,
    observations: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    """Fit teacher duties; keep validation separate from the optimizer samples."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("--epochs, --batch-size and --learning-rate must be positive.")
    sample_count = observations.shape[0]
    if sample_count < 20:
        raise ValueError("Behavior cloning needs at least 20 rollout samples.")
    generator = torch.Generator(device=observations.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(sample_count, device=observations.device, generator=generator)
    validation_count = max(1, sample_count // 10)
    validation = permutation[:validation_count]
    training = permutation[validation_count:]
    # Raw RPM is two orders of magnitude larger than the image errors.  Optimize
    # in standardized coordinates, then fold that affine transform into the
    # first Linear layer.  The saved actor still consumes the exact raw firmware
    # ABI and needs no normalization code or extra exported parameters.
    first = actor[0]
    if not isinstance(first, torch.nn.Linear) or first.in_features != 7:
        raise RuntimeError("BC expected the frozen actor's 7-input first Linear layer.")
    input_mean = observations[training].mean(dim=0)
    input_std = observations[training].std(dim=0).clamp(min=1.0e-3)
    normalized_inputs = (observations - input_mean) / input_std
    # Fit the pre-tanh activation, not the duty. The deployed actor is still
    # ``tanh(actor(obs))``; only the supervised target changes. Optimizing
    # through a saturated tanh gives very small gradients in the steering
    # states that most need correction. Applying atanh to the clipped teacher
    # duty makes the regression target explicit and keeps the fit well scaled.
    squash_limit = 1.0 - 1.0e-3
    pre_tanh_targets = torch.atanh(targets.clamp(-squash_limit, squash_limit))
    optimizer = torch.optim.AdamW(actor.parameters(), lr=learning_rate, weight_decay=1.0e-6)
    # Cosine decay gives the final epochs smaller correction steps. This matters
    # near the centreline, where steering is a small left/right differential on
    # top of a much larger common drive command and sign errors are costly.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 1.0e-2,
    )
    actor.train()
    for epoch in range(epochs):
        order = training[torch.randperm(len(training), device=observations.device, generator=generator)]
        cumulative_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            prediction = actor(normalized_inputs[indices])
            loss = torch.nn.functional.mse_loss(prediction, pre_tanh_targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()
            cumulative_loss += float(loss.detach()) * len(indices)
        schedule.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            mean_loss = cumulative_loss / len(training)
            print(f"BC epoch {epoch + 1:3d}/{epochs}: pre-tanh MSE={mean_loss:.8f}")
    # If y = W*((x-mean)/std)+b, then y = (W/std)*x +
    # (b-W*mean/std).  Apply it once so inference sees raw ABI values.
    with torch.no_grad():
        normalized_weight = first.weight.clone()
        first.weight.copy_(normalized_weight / input_std[None, :])
        first.bias.sub_((normalized_weight * (input_mean / input_std)[None, :]).sum(dim=1))
    actor.eval()
    with torch.inference_mode():
        # Validate after folding, on the original raw observations used by the
        # firmware and rendered-camera lane.
        predicted = torch.tanh(actor(observations[validation]))
        error = predicted - targets[validation]
        validation_mse = float((error.square()).mean())
        validation_mae = float(error.abs().mean())
        validation_max_abs = float(error.abs().max())
    return {
        "validation_samples": float(validation_count),
        "validation_duty_mse": validation_mse,
        "validation_duty_mae": validation_mae,
        "validation_max_abs_error": validation_max_abs,
    }


def main() -> None:
    torch.manual_seed(args_cli.seed)
    env_cfg = LineFollowingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.blind_duty_scale is not None:
        env_cfg.actions.duty.blind_duty_scale = args_cli.blind_duty_scale

    config = env_cfg.line_config
    blind_duty_scale = float(env_cfg.actions.duty.blind_duty_scale)
    if not 0.0 <= blind_duty_scale <= 1.0:
        raise ValueError("blind-duty-scale must be in [0, 1].")
    log_dir = args_cli.output_dir or project_root() / "isaac_sim/output/rl" / "behavior_cloning"
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("=" * 70)
    print(f"BC environments   : {env.num_envs}")
    print(f"rollout samples   : {env.num_envs} x {args_cli.rollout_steps} = "
          f"{env.num_envs * args_cli.rollout_steps}")
    print("teacher            : teacher_batch.analytic_duty")
    print("actor              : frozen 7 -> 64 -> 64 -> 2 (ELU, direct duty)")
    print(f"blind duty scale   : {blind_duty_scale}")
    print(f"output directory   : {log_dir}")
    print("=" * 70)

    try:
        observations, targets = collect_teacher_rollout(env, config, args_cli.rollout_steps)
        uniform_observations, uniform_targets = collect_uniform_teacher_samples(
            config,
            args_cli.uniform_samples,
            args_cli.uniform_heading_max_rad,
            env.device,
            args_cli.seed + 1,
        )
        if args_cli.uniform_samples:
            observations = torch.cat([observations, uniform_observations], dim=0)
            targets = torch.cat([targets, uniform_targets], dim=0)
            print(f"uniform ABI samples: {args_cli.uniform_samples}")
        # Construct the actor through RSL-RL so the saved checkpoint is directly
        # loadable by play_policy_rl.py and has the same state_dict keys as PPO.
        wrapped_env = RslRlVecEnvWrapper(env)
        runner_cfg = LineFollowingPPORunnerCfg()
        if args_cli.device is not None:
            runner_cfg.device = args_cli.device
        runner = OnPolicyRunner(wrapped_env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
        actor = runner.alg.policy.actor
        actor_layout_is_frozen(actor)
        metrics = train_actor(
            actor, observations, targets, args_cli.epochs, args_cli.batch_size,
            args_cli.learning_rate, args_cli.seed,
        )
        config_path = project_root() / "isaac_sim/config/default.json"
        metadata: dict[str, object] = {
            "kind": "behavior_cloning",
            "teacher": "teacher_batch.analytic_duty",
            "input_normalization": "folded_into_first_linear_layer",
            "architecture": [7, 64, 64, 2],
            "activation": "elu",
            "abi": [
                "e_y", "e_theta_rad", "line_confidence", "rpm_left", "rpm_right",
                "duty_left_prev", "duty_right_prev",
            ],
            "samples": int(observations.shape[0]),
            "num_envs": env.num_envs,
            "rollout_steps": args_cli.rollout_steps,
            "uniform_samples": args_cli.uniform_samples,
            "uniform_heading_max_rad": args_cli.uniform_heading_max_rad,
            "epochs": args_cli.epochs,
            "batch_size": args_cli.batch_size,
            "learning_rate": args_cli.learning_rate,
            "seed": args_cli.seed,
            "blind_duty_scale": blind_duty_scale,
            **config_contract(config_path),
            **metrics,
        }
        checkpoint_path = log_dir / "model_bc.pt"
        torch.save({
            "model_state_dict": runner.alg.policy.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "iter": 0,
            "infos": metadata,
        }, checkpoint_path)
        (log_dir / "bc_summary.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(f"BC checkpoint      : {checkpoint_path}")
        print(f"validation duty MAE: {metrics['validation_duty_mae']:.8f}")
        print("Behavior cloning complete. Validate it in the rendered camera lane before PPO.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
