"""Termination terms.

``line_lost`` uses the same threshold as the acceptance gate,
``evaluation.max_line_loss_s``.  Do not loosen it here to make training easier:
the number is what the gate measures, and a policy trained against a softer
limit is being graded on a different task than the one it will be tested on.

Every term refreshes state through ``update_line_state`` rather than reading a
stale snapshot, and every failure term excludes the tick on which the robot
finished.  Isaac Lab evaluates all termination terms on the same step, so
without that exclusion one tick could report ``finished`` *and* ``line_lost``
together -- paying the finish bonus and the failure penalty at once, on exactly
the episodes the policy most needs reinforced.
"""

from __future__ import annotations

import torch

from .observations import OFF_TRACK_LIMIT_M, update_line_state


def _finished_from_state(env, state) -> torch.Tensor:
    """Completion predicate shared by the mutually exclusive terminal terms."""
    tolerance = float(env.cfg.line_config["evaluation"]["completion_tolerance_m"])
    return state.progress >= state.track.total_length - tolerance


def line_lost(env) -> torch.Tensor:
    """True once the tape has been missing for longer than the gate allows."""
    state = update_line_state(env)
    lost = state.line_loss_s > float(env.cfg.line_config["evaluation"]["max_line_loss_s"])
    return lost & ~_finished_from_state(env, state)


def off_track(env, limit_m: float = OFF_TRACK_LIMIT_M) -> torch.Tensor:
    """True when the robot has wandered far enough that recovery is not the task."""
    state = update_line_state(env)
    return (state.distance_to_track > limit_m) & ~_finished_from_state(env, state)


def unstable_physics(env) -> torch.Tensor:
    """True on a tip-over or a robot that has left the floor."""
    from isaaclab.utils.math import euler_xyz_from_quat

    state = update_line_state(env)
    robot = env.scene["robot"]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    limit = torch.deg2rad(torch.tensor(
        float(env.cfg.line_config["evaluation"]["max_roll_pitch_deg"]), device=env.device))
    wrapped = lambda angle: torch.atan2(torch.sin(angle), torch.cos(angle))  # noqa: E731
    tilted = (wrapped(roll).abs() > limit) | (wrapped(pitch).abs() > limit)
    failed = tilted | (robot.data.root_pos_w[:, 2] < -0.01)
    return failed & ~_finished_from_state(env, state)


def stalled(env) -> torch.Tensor:
    """End training episodes that make no useful route progress for too long.

    This is deliberately a training-only failure, not a relaxed evaluation
    gate. It prevents a visible-line episode from consuming the full timeout
    when commanded duty remains below the loaded breakaway requirement.
    """
    config = env.cfg.line_config["rl"]["terminations"]
    timeout_s = float(config["stall_timeout_s"])
    if timeout_s <= 0.0:
        raise ValueError("rl.terminations.stall_timeout_s must be positive.")
    state = update_line_state(env)
    return (state.stall_duration_s >= timeout_s) & ~_finished_from_state(env, state)


def reached_finish(env) -> torch.Tensor:
    """True when the robot has run the whole polyline.

    Reported separately from the failure terminations so the success rate can be
    read directly from the training logs rather than inferred from return.
    """
    return _finished_from_state(env, update_line_state(env))
