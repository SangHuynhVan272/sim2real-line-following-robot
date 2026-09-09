"""Reward terms: pay for finishing the route, not for sitting on the tape.

These read privileged simulator state -- track progress and distance to the
tape -- that the observation deliberately does not carry. The actor still
receives only the frozen firmware ABI, so nothing here can leak into what the
robot needs at runtime.

Two design decisions are worth understanding before changing anything.

**Centring is a penalty, never a per-step bonus.** Paying a positive reward for
"camera sees the tape and the robot is centred" makes the cheapest policy
obvious: stop on the tape and collect it until the episode times out. Progress
is therefore the only positive per-step term, and it is normalized by one step
at the target speed so its magnitude is comparable with the error penalties --
a policy cannot outscore driving by parking on a perfectly visible line.

**Terminal weights are sized against the *discounted* return.** PPO optimises a
discounted objective, so the ceiling a terminal bonus must beat is

    progress_target_step * progress_max_fraction / (1 - gamma)

and not the undiscounted route return. Sizing the bonus against the wrong
ceiling can drown out shaping and over-reward collecting the terminal bonus as
quickly as possible. Re-derive ``finish`` and ``failure`` whenever ``gamma`` or
the progress terms change.
"""

from __future__ import annotations

import torch

from .scene_state import get_line_state
from .terminations import line_lost, off_track, reached_finish, stalled, unstable_physics


def normalized_track_progress(env, max_fraction: float) -> torch.Tensor:
    """Return new best-route distance as a fraction of one target-speed step.

    At the configured target speed the term is 1.0 rather than a small number
    of metres (about 8.3e-4 at 60 Hz).  That makes its scale commensurate with
    the error and time terms, so a policy cannot outscore driving by parking on
    a perfectly visible line.
    """
    state = get_line_state(env)
    step = state.progress - state.previous_progress
    config = env.cfg.line_config
    target_step_m = float(config["controller"]["target_speed_mps"]) / float(
        config["physics"]["policy_hz"]
    )
    if target_step_m <= 0.0:
        raise ValueError("controller.target_speed_mps and physics.policy_hz must define a positive target step.")
    return (step / target_step_m).clamp(0.0, float(max_fraction))


def lateral_error_penalty(env, scale_m: float) -> torch.Tensor:
    """Negative squared distance from the tape centreline."""
    if scale_m <= 0.0:
        raise ValueError("lateral reward scale must be positive.")
    state = get_line_state(env)
    return -(state.distance_to_track / float(scale_m)).square()


def heading_alignment(env) -> torch.Tensor:
    """Penalty on the observed heading error, in radians."""
    return -get_line_state(env).observation()[:, 1].abs()


def action_rate_penalty(env) -> torch.Tensor:
    """Penalty on how far the duty moved this step.

    Measured against the slew limit so the term is scale-free: 1.0 means the
    policy asked for the largest step the hardware can take.
    """
    state = get_line_state(env)
    slew = float(env.cfg.line_config["motor"]["duty_rate_limit_per_s"]) / float(
        env.cfg.line_config["physics"]["policy_hz"])
    return -(state.duty_delta.abs() / slew).sum(dim=-1)


def duty_excess_penalty(env) -> torch.Tensor:
    """Charge duty the policy asked for above the loaded minimum.

    The dead band puts a floor under duty, so ``minimum_loaded_command_duty`` is
    the cheapest command the plant will honour and everything above it is a
    choice. Without this term, the finish bonus and per-tick time charge can
    make full duty artificially attractive and produce bang-bang control that
    depends on an unrealistically forgiving simulated plant.

    Charged on the *common mode* -- the mean of the two duties -- not per wheel.
    A skid-steer chassis this size only changes heading by reversing a side, and
    the envelope lifts a pivot command to well above the floor, so a per-wheel
    charge taxes the one manoeuvre the plant has for steering.  The common mode
    is the forward speed the policy chose; the differential is the steering it
    needed.  Only the first is a free choice, so only the first is charged.

    Normalised by the usable span, so 1.0 means full forward duty regardless of
    where the measured dead band happens to sit.
    """
    motor = env.cfg.line_config["motor"]
    floor = float(motor["minimum_loaded_command_duty"])
    span = max(1.0e-6, float(motor["max_duty"]) - floor)
    common_mode = get_line_state(env).duty_prev.mean(dim=-1).abs()
    excess = (common_mode - floor).clamp(min=0.0)
    return -(excess / span)


def time_penalty(env) -> torch.Tensor:
    """Charge every policy tick so completion is better than waiting."""
    return -torch.ones_like(get_line_state(env).progress)


def reached_finish_bonus(env) -> torch.Tensor:
    """One-step success reward, matching the existing finish termination."""
    return reached_finish(env).to(dtype=torch.float32)


def terminal_failure_penalty(env) -> torch.Tensor:
    """Penalize every non-success terminal, including timeout and deadlock."""
    timed_out_without_finish = env.termination_manager.time_outs & ~reached_finish(env)
    failed = line_lost(env) | off_track(env) | unstable_physics(env) | stalled(env) | timed_out_without_finish
    return -failed.to(dtype=torch.float32)
