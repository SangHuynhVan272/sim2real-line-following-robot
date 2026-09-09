"""Shared accessors for the per-environment line state and camera pose.

Manager terms are free functions, so the state they share has to hang off the
environment.  Everything that needs ``LineState`` goes through
``get_line_state`` so there is exactly one buffer per environment.
"""

from __future__ import annotations

import math

import torch

from ..line_state import LineState
from ..timing import integer_rate_stride


def get_line_state(env) -> LineState:
    """Return the environment's ``LineState``, creating it on first use."""
    state = getattr(env, "_line_state", None)
    if state is None:
        config = env.cfg.line_config
        camera_stride = integer_rate_stride(
            float(config["physics"]["policy_hz"]),
            float(config["physics"]["camera_hz"]),
            label="policy-to-camera",
        )
        state = LineState(env.num_envs, config, env.device, camera_stride)
        env._line_state = state
    return state


def get_scenario(env) -> dict[str, torch.Tensor]:
    """Return the per-environment randomized scenario tensors.

    Populated by ``events.randomize_scenario`` on reset.  Falls back to the
    nominal values so a term can never silently read an empty buffer.
    """
    scenario = getattr(env, "_line_scenario", None)
    if scenario is None:
        from .events import nominal_scenario

        scenario = nominal_scenario(env)
        env._line_scenario = scenario
    return scenario


def local_xy(
    env,
    world_xy: torch.Tensor,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Strip the per-environment origin so coordinates are track-relative.

    Isaac Lab clones environments across a grid, but there is only one track
    polyline and it lives at the local origin.  Every geometric query -- camera
    projection, progress, distance to the tape -- must therefore be done in the
    environment's own frame.  Mixing the two frames puts every cloned robot
    kilometres from the track and is silent until the rewards look absurd.
    """
    origins = env.scene.env_origins if env_ids is None else env.scene.env_origins[env_ids]
    if origins.shape[0] != world_xy.shape[0]:
        raise ValueError(
            "local_xy needs env_ids when world_xy contains only a subset of environments."
        )
    return world_xy - origins[:, :world_xy.shape[-1]]


def camera_pose(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(eye, forward, robot_xy)`` in the environment's local frame.

    Mirrors ``run_line_following.py``: the camera sits at the mount offset in
    the robot frame and looks forward and down by the scenario tilt.  Camera
    geometry -- not controller gain -- is the binding constraint on how tight a
    curve the robot can follow, so this must stay in step with the simulator.
    """
    from isaaclab.utils.math import matrix_from_quat

    robot = env.scene["robot"]
    config = env.cfg.line_config
    position = robot.data.root_pos_w.clone()
    position[:, :2] = local_xy(env, position[:, :2])
    rotation = matrix_from_quat(robot.data.root_quat_w)
    scenario = get_scenario(env)
    nominal = torch.tensor(config["robot"]["camera_mount_xyz_m"], device=env.device, dtype=position.dtype)
    mount = nominal[None, :] + torch.stack([
        scenario["camera_mount_dx_m"], scenario["camera_mount_dy_m"], scenario["camera_mount_dz_m"],
    ], dim=-1)
    eye = position + torch.einsum("nij,nj->ni", rotation, mount)
    # Tilt and yaw are both bracket geometry: a printed mount delivers the down
    # angle with a couple of degrees of slop and rarely points exactly along the
    # robot's centreline.
    down = torch.deg2rad(scenario["camera_down_angle_deg"])
    yaw = torch.deg2rad(scenario["camera_yaw_deg"])
    optical = torch.stack([
        torch.cos(down) * torch.cos(yaw), torch.cos(down) * torch.sin(yaw), -torch.sin(down),
    ], dim=-1)
    forward = torch.einsum("nij,nj->ni", rotation, optical)
    return eye, forward, position[:, :2]


def wheel_signs(env) -> torch.Tensor:
    """Return the per-DOF velocity signs that mirror the right-hand wheel pair.

    Four wheel DOFs, two logical sides.  Commands expand to
    ``[left, left, right, right]`` and measured speed averages back per side;
    both directions must use these same signs or the encoder feedback silently
    inverts.
    """
    config = env.cfg.line_config
    return torch.tensor(
        config["robot"]["wheel_joint_velocity_signs"], device=env.device, dtype=torch.float32,
    )


def measured_rpm(env) -> torch.Tensor:
    """Return ``[N, 2]`` front-wheel RPM, the only wheel speeds in the ABI.

    Firmware has one encoder on each *front* wheel; the rear pair is driven but
    never measured, so it must not leak into the observation.
    """
    robot = env.scene["robot"]
    indices = env.cfg.wheel_dof_indices
    encoder_slots = env.cfg.encoder_dof_slots
    radps = robot.data.joint_vel[:, indices] * wheel_signs(env)
    return radps[:, encoder_slots] * 60.0 / (2.0 * math.pi)
