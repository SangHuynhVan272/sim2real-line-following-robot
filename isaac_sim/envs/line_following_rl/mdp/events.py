"""Domain randomization on reset.

Isaac Lab cannot re-spawn geometry when an environment resets, and this lane has
almost no geometry to re-spawn anyway.  So randomization here changes
*parameters* of prims that already exist -- which is a real restriction, not a
formality, and it splits ``default.json``'s ranges into two groups.

**Randomized here (they change the plant or the projection):**
camera tilt, yaw and HFOV, the three camera mount offsets, start pose, wheel
radius and track width scale, encoder scale and noise, supply voltage, driver
drop, no-load speed, stall torque, Coulomb friction, motor imbalance, and the
four lens coefficients.  Base mass and centre of mass are randomized too, but
by Isaac Lab's own event terms in ``env_cfg.py`` rather than here, because they
act on the articulation instead of on the projection. Wheel/floor static and
dynamic friction are likewise assigned by an Isaac Lab material event; a unit
multiply-combine floor keeps the effective contact coefficient equal to the
wheel-side draw.

``camera_roll_deg`` rotates the geometric image basis around the optical axis,
matching the rendered camera transform.  It therefore perturbs both e_y and
e_theta without moving the optical ray itself.

**Not randomized here (they only change what a rendered image looks like):**
``tape_brightness``, ``floor_luma``, ``floor_tint``, ``floor_roughness``,
``tape_roughness``, ``floor_dirt_*``, ``dome_intensity``, ``key_intensity``,
``jpeg_quality``, ``motion_blur_mix``, ``rolling_shutter_fraction``,
``exposure_ev_bias``, ``auto_gain_*``.  There is no camera in this lane, so
these have nothing to act on.  ``tape_width_m`` is in the same group: the
projection follows the tape centreline, so its width does not move ``e_y``.

That gap is the honest cost of training without rendering, and it is exactly why
the validate lane still runs the real camera.  A policy that only ever met a
perfect detector has never been asked what to do with a bad one -- which is the
whole reason ``line_confidence`` is in the observation.
"""

from __future__ import annotations

import torch

from ..motor_batch import motor_curve
from .scene_state import get_line_state


def _uniform(env, low: float, high: float, env_ids: torch.Tensor) -> torch.Tensor:
    return torch.rand(len(env_ids), device=env.device) * (high - low) + low


def _max_of(config: dict, key: str, fallback: float) -> float:
    """Nominal value for a field whose randomization is a range of delays.

    The nominal scenario has to be the *deepest* delay, not the shallowest: the
    buffers are sized from the same maximum, and a nominal run that asked for a
    shallower one would read past what it had filled.
    """
    value = config["randomization"].get(key, fallback)
    return float(max(value)) if isinstance(value, (list, tuple)) else float(value)


def _range(config: dict, key: str, fallback: float) -> tuple[float, float]:
    """Read a ``[low, high]`` randomization range, tolerating a scalar entry."""
    value = config["randomization"].get(key, fallback)
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return float(value), float(value)


#: Ranges pulled from ``randomization`` and applied per environment on reset.
#: Keys are the scenario field names used by ``run_line_following.py`` so the two
#: lanes stay describable in the same vocabulary.
SCENARIO_FIELDS = (
    "camera_down_angle_deg",
    "camera_hfov_deg",
    # wheel_radius_scale and track_width_scale are deliberately absent.  They
    # only feed the analytic teacher's formula; the imported wheel collision
    # and joint frames never move, so randomizing them perturbs the label BC
    # regresses against without perturbing the plant PPO acts on.  That is
    # label noise dressed as domain randomization.
    "encoder_scale",
    "supply_voltage_v",
    "driver_drop_v",
    "no_load_wheel_radps",
    "stall_torque_nm",
    "coulomb_friction_nm",
    "loaded_breakaway_duty",
    "scrub_breakaway_gain",
    "motor_imbalance",
    # Build tolerance.  The CAD puts the camera 40.8 mm off the floor and the
    # assembled robot agrees to within a few millimetres, but the bracket is 3D
    # printed and the module is glued by hand, so the residual is real. A policy
    # trained at one exact mount pose depends on a build nobody can reproduce.
    "camera_mount_dx_m",
    "camera_mount_dy_m",
    "camera_mount_dz_m",
    "camera_yaw_deg",
    "camera_roll_deg",
    # Latency is not a constant on the real pipeline: a dropped frame, a slow
    # JPEG or a scheduler hiccup moves it by a whole period.  Drawn as a float
    # and rounded where it is consumed, so it stays one uniform draw like every
    # other field.
    "camera_delay_ticks",
    "actuation_delay_ticks",
    # Encoder noise and viscous friction were single unmeasured scalars.  A
    # policy trained against one value depends on a number nobody has measured.
    "encoder_noise_rpm",
    "viscous_friction_nm_per_radps",
    # A 2S pack droops under load and over a run, so the dead band the policy
    # meets late in an episode is not the one it met at the start.
    "supply_sag_fraction",
)

#: Lens coefficients, named as the perception batch expects them.
LENS_FIELDS = (
    ("lens_k1", "lens_radial_k1"),
    ("lens_k2", "lens_radial_k2"),
    ("lens_p1", "lens_tangential_p1"),
    ("lens_p2", "lens_tangential_p2"),
)


def nominal_scenario_values(config: dict) -> dict[str, float]:
    """Return scalar calibrated defaults from the shared configuration."""
    robot, motor = config["robot"], config["motor"]
    return {
        "camera_down_angle_deg": float(robot["camera_down_angle_deg"]),
        "camera_hfov_deg": float(robot["camera_hfov_deg"]),
        "wheel_radius_scale": 1.0, "track_width_scale": 1.0, "encoder_scale": 1.0,
        "supply_voltage_v": float(motor["supply_voltage_v"]),
        "driver_drop_v": float(motor["driver_drop_v"]),
        "no_load_wheel_radps": float(motor["no_load_wheel_radps"]),
        "stall_torque_nm": float(motor["stall_torque_nm"]),
        "coulomb_friction_nm": float(motor["coulomb_friction_nm"]),
        "loaded_breakaway_duty": float(motor["loaded_breakaway_duty"]),
        "scrub_breakaway_gain": float(motor["scrub_breakaway_gain"]),
        "motor_imbalance": 1.0,
        "camera_mount_dx_m": 0.0, "camera_mount_dy_m": 0.0, "camera_mount_dz_m": 0.0,
        "camera_yaw_deg": 0.0, "camera_roll_deg": 0.0,
        "camera_delay_ticks": float(_max_of(config, "camera_delay_ticks", 2.0)),
        "actuation_delay_ticks": float(_max_of(config, "actuation_delay_ticks", 1.0)),
        "encoder_noise_rpm": float(_max_of(config, "encoder_noise_rpm", 1.0)),
        "viscous_friction_nm_per_radps": float(motor["viscous_friction_nm_per_radps"]),
        "supply_sag_fraction": 0.0,
        "lens_k1": 0.0, "lens_k2": 0.0, "lens_p1": 0.0, "lens_p2": 0.0,
    }


def nominal_scenario(env) -> dict[str, torch.Tensor]:
    """Return every environment at the calibrated nominal, identity lens included."""
    defaults = nominal_scenario_values(env.cfg.line_config)
    return {
        key: torch.full((env.num_envs,), value, device=env.device)
        for key, value in defaults.items()
    }


def _apply_motor_line(env, env_ids: torch.Tensor, scenario: dict[str, torch.Tensor]) -> None:
    """Apply reset-time motor draws to the explicit actuator and joint friction.

    The actuator owns damping and torque clipping, so changing only the target
    speed would leave the randomized plant with its nominal torque/speed slope.
    Keep the solver's drive passive; the explicit actuator below is the sole
    source of motor effort.
    """
    wheel_ids = env.cfg.wheel_dof_indices
    if wheel_ids is None:
        raise RuntimeError("Wheel DOF IDs must be resolved before reset events run.")
    robot = env.scene["robot"]
    actuator = robot.actuators.get("wheels")
    if actuator is None:
        raise RuntimeError("The line-following robot requires the explicit 'wheels' actuator.")

    config = env.cfg.line_config
    _, damping = motor_curve(
        scenario["no_load_wheel_radps"][env_ids],
        scenario["stall_torque_nm"][env_ids],
        float(config["motor"]["measured_at_volts"]),
        # The sampled tensor, not the config scalar.  Drawing a range and then
        # handing the plant a constant is randomization that never happened:
        # every environment ran the nominal damping while the report claimed a
        # 3x spread.
        scenario["viscous_friction_nm_per_radps"][env_ids],
    )
    stall_torque = scenario["stall_torque_nm"][env_ids]
    actuator.damping[env_ids] = damping[:, None]
    actuator.effort_limit[env_ids] = stall_torque[:, None]

    # Isaac Sim 5 models static and dynamic Coulomb friction separately.  Both
    # must replace the SolidWorks 0.1 N m default; the motor slope above already
    # contains the modeled viscous term, so do not add it a second time here.
    friction = scenario["coulomb_friction_nm"][env_ids, None].expand(-1, len(wheel_ids))
    robot.write_joint_friction_coefficient_to_sim(
        friction,
        joint_ids=wheel_ids,
        env_ids=env_ids,
        joint_dynamic_friction_coeff=friction,
        joint_viscous_friction_coeff=torch.zeros_like(friction),
    )


def randomize_scenario(env, env_ids: torch.Tensor) -> None:
    """Draw a fresh plant and lens for the environments that just reset."""
    config = env.cfg.line_config
    scenario = getattr(env, "_line_scenario", None)
    if scenario is None:
        scenario = nominal_scenario(env)
        env._line_scenario = scenario
    defaults = nominal_scenario_values(config)
    for key in SCENARIO_FIELDS:
        low, high = _range(config, key, defaults[key])
        scenario[key][env_ids] = _uniform(env, low, high, env_ids)
    for name, source in LENS_FIELDS:
        low, high = _range(config, source, 0.0)
        scenario[name][env_ids] = _uniform(env, low, high, env_ids)
    _apply_motor_line(env, env_ids, scenario)


def reset_robot_on_track(env, env_ids: torch.Tensor) -> None:
    """Place the robot along the route with a randomized pose error.

    The lateral and heading offsets use the same ranges as evaluation. They
    matter because camera field of view and tape geometry determine whether the
    line crosses enough image rows to produce a valid observation.
    """
    config = env.cfg.line_config
    robot = env.scene["robot"]
    state = get_line_state(env)

    lateral_low, lateral_high = _range(config, "start_lateral_m", 0.0)
    heading_low, heading_high = _range(config, "start_heading_deg", 0.0)
    lateral = _uniform(env, lateral_low, lateral_high, env_ids)
    heading = torch.deg2rad(_uniform(env, heading_low, heading_high, env_ids))

    training = config["rl"]["training"]
    progress_low, progress_high = (float(value) for value in training["start_progress_fraction"])
    full_start_probability = float(training["full_route_start_probability"])
    if not 0.0 <= progress_low <= progress_high < 1.0:
        raise ValueError("rl.training.start_progress_fraction must satisfy 0 <= low <= high < 1.")
    if not 0.0 <= full_start_probability <= 1.0:
        raise ValueError("rl.training.full_route_start_probability must be in [0, 1].")
    start_progress = _uniform(env, progress_low, progress_high, env_ids) * state.track.total_length
    full_route = torch.rand(len(env_ids), device=env.device) < full_start_probability
    start_progress = torch.where(full_route, torch.zeros_like(start_progress), start_progress)
    start = state.track.point_at(start_progress)
    tangent = state.track.tangent_at(start_progress)
    track_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
    yaw = track_yaw + heading
    normal = torch.stack([-torch.sin(track_yaw), torch.cos(track_yaw)], dim=-1)

    root = robot.data.default_root_state[env_ids].clone()
    root[:, :2] = start + normal * lateral[:, None] + env.scene.env_origins[env_ids, :2]
    half = yaw / 2.0
    root[:, 3] = torch.cos(half)
    root[:, 4:6] = 0.0
    root[:, 6] = torch.sin(half)
    robot.write_root_state_to_sim(root, env_ids)
    robot.write_joint_state_to_sim(
        torch.zeros_like(robot.data.joint_pos[env_ids]),
        torch.zeros_like(robot.data.joint_vel[env_ids]),
        env_ids=env_ids,
    )
    state.reset_idx(env_ids)
    progress, distance = state.track.project(root[:, :2] - env.scene.env_origins[env_ids, :2])
    state.progress[env_ids] = progress
    state.previous_progress[env_ids] = progress
    state.stall_reference_progress[env_ids] = progress
    state.stall_duration_s[env_ids] = 0.0
    state.distance_to_track[env_ids] = distance
