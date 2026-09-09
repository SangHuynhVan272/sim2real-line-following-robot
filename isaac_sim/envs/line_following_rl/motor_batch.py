"""Batched DC-motor line and the firmware safety envelope.

These are the formulas from ``run_line_following.py`` expressed over a batch.
They are reproduced rather than imported because the training lane runs them on
1024 environments per tick in torch; ``check_rl_parity.py`` keeps them numerically
identical to the scalar originals.

The no-PID contract lives here.  A brushed motor obeys ``tau = Kt*(V - Ke*w)/R``,
which is a velocity drive whose damping is the constant stall/no-load slope and
whose target is the no-load speed for the applied volts.  Nothing in this file
is a servo, and nothing may become one.
"""

from __future__ import annotations

import torch


def motor_curve(
    no_load_wheel_radps: torch.Tensor,
    stall_torque_nm: torch.Tensor,
    measured_at_volts: float,
    viscous_friction_nm_per_radps: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(radps_per_volt, damping)`` for the PhysX drive.

    ``damping`` is constant with speed -- that constancy is what makes the drive
    a torque/speed line instead of a velocity servo.
    """
    radps_per_volt = no_load_wheel_radps / measured_at_volts
    torque_per_volt = stall_torque_nm / measured_at_volts
    damping = torque_per_volt / radps_per_volt.clamp(min=1e-9) + viscous_friction_nm_per_radps
    return radps_per_volt, damping


def scrub_severity(duty: torch.Tensor) -> torch.Tensor:
    """How tight a turn a ``[..., 2]`` left/right duty pair asks for, on [0, 1].

    A skid-steer with no steering geometry can only change heading by sliding
    all four tyres sideways, so the load a command puts on the motors is set by
    the *curvature* it asks for, not by its magnitude.  This is that curvature,
    normalised: 0 for a straight run, 0.5 for turning about one stopped wheel,
    1.0 for a pivot in place.  It is scale free on purpose -- stiction is a
    threshold on duty, and the threshold must not move when the robot is simply
    asked to go faster along the same arc.
    """
    if duty.shape[-1] != 2:
        raise ValueError("scrub severity expects [..., 2] left/right duty")
    difference = (duty[..., 0] - duty[..., 1]).abs()
    common = (duty[..., 0] + duty[..., 1]).abs()
    return difference / (difference + common).clamp(min=1.0e-9)


def effective_breakaway_duty(
    loaded_breakaway_duty: torch.Tensor,
    scrub_breakaway_gain: torch.Tensor,
    severity: torch.Tensor,
) -> torch.Tensor:
    """Duty a *turning* command must clear before the wheels move at all.

    ``loaded_breakaway_duty`` alone is the straight-line number, and using it
    for every command is what let a trained policy believe a weak pivot works.
    Measured in this simulator, 0.58/-0.60 crawls at 7 rpm and 0.5 deg/s -- on
    the robot the same command holds the encoders at a hard zero, and the
    controller deadlocks because a frozen observation reproduces the same
    output forever.  Charging the scrub load against the breakaway duty removes
    that phantom crawl, so the only pivots the policy can learn are the strong
    ones the hardware can actually execute.
    """
    return loaded_breakaway_duty + scrub_breakaway_gain * severity


def duty_to_volts(
    duty: torch.Tensor,
    supply_voltage_v: torch.Tensor,
    driver_drop_v: torch.Tensor,
    loaded_breakaway_duty: torch.Tensor,
) -> torch.Tensor:
    """Effective armature volts for a PWM duty.

    The H-bridge drop is what creates the dead zone.  Because the drop is fixed
    while the pack sags, the dead zone widens as the battery drains, exactly as
    on hardware -- it is never a separate tuning fudge.
    """
    magnitude = (duty.abs() * supply_voltage_v - driver_drop_v).clamp(min=0.0)
    magnitude = torch.where(duty.abs() >= loaded_breakaway_duty, magnitude, torch.zeros_like(magnitude))
    return torch.copysign(magnitude, duty)


def duty_to_no_load_radps(
    duty: torch.Tensor,
    supply_voltage_v: torch.Tensor,
    driver_drop_v: torch.Tensor,
    loaded_breakaway_duty: torch.Tensor,
    radps_per_volt: torch.Tensor,
) -> torch.Tensor:
    """PhysX drive target: the no-load wheel speed for the applied volts."""
    return duty_to_volts(
        duty, supply_voltage_v, driver_drop_v, loaded_breakaway_duty,
    ) * radps_per_volt


def expand_side_duty(duty: torch.Tensor, imbalance: torch.Tensor) -> torch.Tensor:
    """Expand left/right duty to four wheels with a true side-to-side mismatch.

    Wheel order is ``left-front, left-rear, right-front, right-rear``. Applying
    the gain to only rear motors changes axle balance, not the left/right bias
    modeled by the rendered simulator and observed on hardware.
    """
    return torch.stack([
        duty[:, 0] * imbalance,
        duty[:, 0] * imbalance,
        duty[:, 1] / imbalance,
        duty[:, 1] / imbalance,
    ], dim=-1)


def release_below_minimum(duty: torch.Tensor, release_duty: float) -> torch.Tensor:
    """Zero a per-wheel command too small to be worth lifting.

    Without this the lift has no lower bound on what it will honour: any
    magnitude above 1e-9 was raised to the full loaded minimum, so a request of
    -0.02 on one wheel came back as -0.90 and a near-zero steering error became
    a full-rate pivot.  The analytic teacher never lands there -- measured over
    400k samples, it emits no raw per-wheel duty in (0, half the minimum) at
    all -- but a network approximating it does, on about 1.8% of centre-band
    samples, and each one is an unintended pivot.

    Releasing to zero is also the physically honest reading.  The plant delivers
    no motion anywhere below the breakaway duty, so zero and any small command
    produce the same wheel behaviour; what differs is only whether the envelope
    then *invents* a large command on the policy's behalf.
    """
    return torch.where(duty.abs() >= release_duty, duty, torch.zeros_like(duty))


def enforce_minimum_loaded_command(
    duty: torch.Tensor, minimum_duty: float, scrub_command_margin: float = 0.0,
    release_fraction: float = 0.0,
) -> torch.Tensor:
    """Lift a command clear of the loaded dead band, scrub included.

    Applied twice.  The plant charges scrub against the duty it is *given*, but
    a single pass has to read severity from the request, and lifting an
    asymmetric request makes it more of a pivot than it was -- so one pass
    consistently under-lifts exactly the hard turns that must not stall.  The
    second pass sees the severity the plant will see.  Symmetric pairs are a
    fixed point after it, and the worst asymmetric case converges to within
    0.014, which the envelope margin covers.
    """
    released = release_below_minimum(duty, release_fraction * minimum_duty)
    return _minimum_loaded_pass(
        _minimum_loaded_pass(released, minimum_duty, scrub_command_margin),
        minimum_duty, scrub_command_margin,
    )


def _minimum_loaded_pass(
    duty: torch.Tensor, minimum_duty: float, scrub_command_margin: float,
) -> torch.Tensor:
    """Clear the loaded dead zone without erasing left/right steering.

    Lifting the two sides independently maps every small positive pair to
    ``[minimum, minimum]``.  On this robot that erased the entire differential
    command at 0.05 m/s, so the teacher drove straight until it had to pivot.
    For two co-rotating wheels, add one common-mode offset instead: the smaller
    magnitude reaches the measured minimum and their difference is preserved.
    Opposite-sign or single-wheel commands are lifted independently.
    """
    if duty.shape[-1] != 2:
        raise ValueError("loaded-command compensation expects [..., 2] left/right duty")
    epsilon = 1.0e-9
    # A turning command has to clear the scrub-loaded dead band, not the
    # straight-line one. Otherwise a motionless robot can repeatedly emit the
    # same ineffective pivot because its encoder observation never changes.
    # Severity comes from the requested pair before any lift, so the rule
    # cannot chase its own output.
    minimum = minimum_duty + scrub_command_margin * scrub_severity(duty)
    magnitude = duty.abs()
    individually_lifted = torch.where(
        magnitude > epsilon,
        torch.copysign(torch.maximum(magnitude, minimum[..., None].expand_as(magnitude)), duty),
        torch.zeros_like(duty),
    )
    same_sign = (duty[..., 0] * duty[..., 1] > 0.0) & (magnitude.min(dim=-1).values > epsilon)
    common_offset = (minimum - magnitude.min(dim=-1).values).clamp(min=0.0)
    pair_lifted = duty + torch.sign(duty) * common_offset[..., None]
    return torch.where(same_sign[..., None], pair_lifted, individually_lifted)


class SafetyEnvelope:
    """The firmware limits, applied during training and not only at deploy time.

    A policy trained without these learns duty steps the hardware cannot
    deliver, so the envelope is part of the environment rather than a wrapper
    bolted on afterwards.  Every limit here has a counterpart in
    ``export_policy.py``'s generated C and must stay in step with it.
    """

    def __init__(self, config: dict, blind_duty_scale: float, policy_hz: float) -> None:
        motor = config["motor"]
        self.max_duty = float(motor["max_duty"])
        self.minimum_loaded_command_duty = float(motor["minimum_loaded_command_duty"])
        self.scrub_command_margin = float(motor["scrub_command_margin"])
        self.minimum_command_release_fraction = float(
            motor["minimum_command_release_fraction"])
        self.slew_per_step = float(motor["duty_rate_limit_per_s"]) / float(policy_hz)
        self.blind_duty_scale = float(blind_duty_scale)

    def apply(
        self, target_duty: torch.Tensor, duty_prev: torch.Tensor, confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp a raw policy action into what the H-bridge can actually be given.

        ``confidence`` gates the blind cap. At the deployed value 0.0, a robot
        that has never seen tape is held stopped; a non-zero experimental value
        would cap it to that fraction of full duty.
        """
        duty = target_duty.clamp(-self.max_duty, self.max_duty)
        blind = confidence <= 0.0
        capped = duty.clamp(
            -self.max_duty * self.blind_duty_scale, self.max_duty * self.blind_duty_scale,
        )
        duty = torch.where(blind[:, None], capped, duty)
        duty = enforce_minimum_loaded_command(
            duty, self.minimum_loaded_command_duty, self.scrub_command_margin,
            self.minimum_command_release_fraction,
        )
        duty = duty.clamp(-self.max_duty, self.max_duty)
        return torch.clamp(duty, duty_prev - self.slew_per_step, duty_prev + self.slew_per_step)
