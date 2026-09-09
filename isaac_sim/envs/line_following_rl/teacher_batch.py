"""The analytic controller, batched, as a behaviour-cloning teacher.

``policy()`` in ``run_line_following.py`` is pure pursuit plus an open-loop
motor inverse.  It is the current baseline, it already passes nominal, and it
costs nothing to imitate -- so PPO starts from it instead of from noise.

Keeping the teacher here also keeps the comparison honest: the same three-way
evaluation (analytic / behaviour-cloned / PPO) runs against one implementation
of the baseline rather than two.

Note what the teacher does *not* use: ``rpm_left``, ``rpm_right``,
``duty_left_prev`` and ``duty_right_prev`` are in the frozen ABI but the
analytic controller ignores them.  Those four inputs are the headroom an RL
policy has over this baseline.
"""

from __future__ import annotations

import torch

from .motor_batch import enforce_minimum_loaded_command


def wheel_speed_to_duty(radps: torch.Tensor, config: dict) -> torch.Tensor:
    """Open-loop inverse of the motor line -- the feedforward that replaces a PID.

    It uses the *nominal* motor constants on purpose, so a randomized episode is
    driven with the wrong map and the outer loop has to absorb the error, just
    as it must on the real robot.
    """
    motor = config["motor"]
    reference = float(motor["measured_at_volts"])
    radps_per_volt = float(motor["no_load_wheel_radps"]) / reference
    viscous = float(motor["viscous_friction_nm_per_radps"])
    damping = float(motor["stall_torque_nm"]) / reference / radps_per_volt + viscous
    load = float(motor["coulomb_friction_nm"]) + viscous * radps.abs()
    volts = (radps.abs() + load / damping) / radps_per_volt
    duty = (volts + float(motor["driver_drop_v"])) / float(motor["supply_voltage_v"])
    moving = radps.abs() > 1.0e-9
    duty = torch.where(moving, duty, torch.zeros_like(duty))
    maximum = float(motor["max_duty"])
    return torch.copysign(duty, radps).clamp(-maximum, maximum)


def analytic_duty(
    observation: torch.Tensor,
    config: dict,
    wheel_radius_scale: torch.Tensor | None = None,
    track_width_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[N, 2]`` left/right duty for the frozen ``[N, 7]`` observation.

    Mirrors ``policy()`` term for term, including a detail that is easy to miss:
    the differential-drive inverse uses the *scenario* wheel radius and track
    width, not the nominal ones, while ``wheel_speed_to_duty`` below uses the
    nominal motor constants.  Only the motor map is deliberately wrong on a
    randomized episode.  Passing ``None`` for either scale means 1.0.
    """
    e_y, e_theta, confidence = observation[:, 0], observation[:, 1], observation[:, 2]
    controller, robot, vision = config["controller"], config["robot"], config["vision"]
    yaw_rate = float(controller["steering_sign"]) * torch.clamp(
        float(controller["kp_lateral"]) * e_y + float(controller["kp_heading"]) * e_theta,
        -float(controller["max_yaw_rate_radps"]), float(controller["max_yaw_rate_radps"]),
    )
    speed = torch.full_like(e_y, float(controller["target_speed_mps"]))
    if controller.get("algorithm") == "centerline_lookahead":
        speed = torch.clamp(
            speed / (1.0 + float(vision["speed_slowdown_gain"]) * e_theta.abs()),
            min=float(vision["minimum_speed_mps"]),
        )
    speed = speed * torch.clamp(confidence, min=float(controller["minimum_confidence_speed_scale"]))
    ones = torch.ones_like(e_y)
    radius = float(robot["wheel_radius_m"]) * (ones if wheel_radius_scale is None else wheel_radius_scale)
    track = float(robot["track_width_m"]) * (ones if track_width_scale is None else track_width_scale)
    left = (speed - yaw_rate * track / 2.0) / radius
    right = (speed + yaw_rate * track / 2.0) / radius
    raw_duty = torch.stack([wheel_speed_to_duty(left, config), wheel_speed_to_duty(right, config)], dim=-1)
    return enforce_minimum_loaded_command(
        raw_duty, float(config["motor"]["minimum_loaded_command_duty"]),
        float(config["motor"]["scrub_command_margin"]),
        float(config["motor"]["minimum_command_release_fraction"]),
    ).clamp(-float(config["motor"]["max_duty"]), float(config["motor"]["max_duty"]))
