"""The action term: a raw policy output becomes H-bridge duty and wheel targets.

This is where the no-PID contract is enforced.  The policy emits duty in
[-1, 1] per side and it is written straight through to the motor line.  There is
no velocity servo and no inner loop; the only things between the network and
PhysX are the firmware safety limits, which exist here precisely so the policy
cannot learn to rely on steps the hardware will not deliver.
"""

from __future__ import annotations

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..motor_batch import (
    duty_to_no_load_radps,
    effective_breakaway_duty,
    expand_side_duty,
    motor_curve,
    scrub_severity,
    SafetyEnvelope,
)
from .scene_state import get_line_state, get_scenario, wheel_signs


class DutyAction(ActionTerm):
    """Two signed PWM duties in [-1, 1], one per side.

    ``process_actions`` runs before physics, so it sees the confidence produced
    by the previous step's camera update -- the same ordering the real loop has,
    where the policy always acts on the most recent completed camera sample.
    """

    cfg: DutyActionCfg

    def __init__(self, cfg: DutyActionCfg, env) -> None:
        super().__init__(cfg, env)
        config = env.cfg.line_config
        self._config = config
        expected_joints = list(config["robot"]["wheel_joints"])
        wheel_ids, wheel_names = self._asset.find_joints(expected_joints, preserve_order=True)
        if wheel_names != expected_joints:
            raise RuntimeError(
                "Could not resolve wheel joints in frozen ABI order: "
                f"expected {expected_joints}, resolved {wheel_names}."
            )
        if len(wheel_ids) != len(config["robot"]["wheel_joint_velocity_signs"]):
            raise RuntimeError(
                "wheel_joint_velocity_signs must contain one sign for every resolved wheel DOF."
            )
        # Physics handles only exist after scene creation, so DOF IDs cannot be
        # resolved in LineFollowingEnvCfg.__post_init__.
        env.cfg.wheel_dof_indices = wheel_ids
        self._wheel_dof_indices = wheel_ids
        encoder_slots = env.cfg.encoder_dof_slots
        if encoder_slots is None or any(slot < 0 or slot >= len(wheel_ids) for slot in encoder_slots):
            raise RuntimeError("encoder_dof_slots must index the resolved wheel-joint order.")
        self._envelope = SafetyEnvelope(
            config, cfg.blind_duty_scale, float(config["physics"]["policy_hz"]),
        )
        self._raw = torch.zeros(self.num_envs, 2, device=self.device)
        self._duty = torch.zeros(self.num_envs, 2, device=self.device)
        self._applied_duty = torch.zeros(self.num_envs, 2, device=self.device)
        self._signs = wheel_signs(env)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._duty

    def process_actions(self, actions: torch.Tensor) -> None:
        state = get_line_state(self._env)
        self._raw[:] = actions
        # tanh keeps the network output inside the ABI without a hard clip that
        # would zero the gradient the moment the policy saturates.
        target = torch.tanh(actions)
        duty_before_step = state.duty_prev.clone()
        self._duty[:] = self._envelope.apply(target, duty_before_step, state.confidence)
        self._applied_duty[:] = state.delay_actuation(
            self._duty, get_scenario(self._env)["actuation_delay_ticks"].round(),
        )
        state.duty_delta[:] = self._duty - duty_before_step
        state.duty_prev[:] = self._duty

    def apply_actions(self) -> None:
        """Write the motor line's no-load target to the PhysX velocity drive.

        The drive damping is configured once as the constant stall/no-load
        slope, so setting the no-load speed for the applied volts reproduces the
        real torque/speed line.  This is not a speed command being tracked.
        """
        scenario = get_scenario(self._env)
        radps_per_volt, _ = motor_curve(
            scenario["no_load_wheel_radps"], scenario["stall_torque_nm"],
            float(self._config["motor"]["measured_at_volts"]),
            scenario["viscous_friction_nm_per_radps"],
        )
        side_duty = expand_side_duty(
            self._applied_duty, scenario["motor_imbalance"],
        ).clamp(-float(self._config["motor"]["max_duty"]), float(self._config["motor"]["max_duty"]))
        # The dead zone is charged the scrub load of the commanded turn, so a
        # pivot has to clear a higher duty than a straight run.  Severity comes
        # from the left/right pair before it is expanded, because it describes
        # the chassis, not one wheel.
        breakaway = effective_breakaway_duty(
            scenario["loaded_breakaway_duty"], scenario["scrub_breakaway_gain"],
            scrub_severity(self._applied_duty),
        )
        # A 2S pack droops under load and over a run.  Scale the pack voltage
        # down across the episode so the dead band the policy meets at the end
        # is not the one it met at the start -- that shift is real, and a policy
        # that only ever saw a fresh battery has not been trained for it.
        age = (self._env.episode_length_buf.to(torch.float32)
               / max(1.0, float(self._env.max_episode_length))).clamp(0.0, 1.0)
        sagged_volts = scenario["supply_voltage_v"] * (
            1.0 - scenario["supply_sag_fraction"] * age
        )
        no_load = duty_to_no_load_radps(
            side_duty, sagged_volts[:, None],
            scenario["driver_drop_v"][:, None],
            breakaway[:, None], radps_per_volt[:, None],
        )
        self._env.scene["robot"].set_joint_velocity_target(
            no_load * self._signs, joint_ids=self._wheel_dof_indices,
        )


@configclass
class DutyActionCfg(ActionTermCfg):
    """Configuration for :class:`DutyAction`."""

    class_type: type = DutyAction

    blind_duty_scale: float = 0.0
    """Fraction of ``max_duty`` the policy may use before its first detection.

    Set to 0.0 to restore the original firmware rule that the wheels stay stopped
    until the camera has seen the tape once.  Any non-zero value here is a
    behaviour the firmware must also implement, and it needs a safety review
    before it ships -- it is not a simulation convenience.
    """
