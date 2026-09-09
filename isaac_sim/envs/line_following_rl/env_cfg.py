# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Huynh Van Sang.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the line-following RL training lane.

The scene here is almost empty on purpose.  The validated simulator spawns a
floor, 52 tape boxes, up to 60 dirt patches, four materials and two lights --
roughly 115 prims.  At 1024 environments that is ~118,000 prims, and none of
them would be looked at, because this lane has no camera.  Replacing the camera
with a geometric projection collapses the scene to a ground plane and the robot,
which is what makes the environment count affordable.

Everything numeric comes from ``isaac_sim/config/default.json``.  It stays the
single source of truth across both lanes; nothing is hard-coded here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as isaaclab_mdp

from . import mdp
from .timing import integer_rate_stride


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_line_config(path: Path | None = None) -> dict:
    """Read ``default.json`` -- the same file the validate lane reads."""
    target = path or project_root() / "isaac_sim/config/default.json"
    return json.loads(target.read_text(encoding="utf-8"))


LINE_CONFIG = load_line_config()
RL_REWARD_CONFIG = LINE_CONFIG["rl"]["rewards"]


def nominal_motor_damping(config: dict) -> float:
    """Return the DC motor torque/speed slope for the nominal hardware model.

    With stiffness fixed at zero, an explicit velocity target and this damping
    implement ``tau = Kt * (V - Ke * omega) / R``.  This is a motor line, not
    a wheel-speed servo.
    """
    motor = config["motor"]
    return (
        float(motor["stall_torque_nm"]) / float(motor["no_load_wheel_radps"])
        + float(motor["viscous_friction_nm_per_radps"])
    )


@configclass
class LineFollowingSceneCfg(InteractiveSceneCfg):
    """Ground plane, robot, and one light so the GUI is not black.

    No tape, no dirt, no materials.  The tape exists only as a polyline inside
    the observation term.
    """

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(200.0, 200.0),
            # Wheel materials carry the episode draw below. Multiplying by a
            # unit-friction floor makes the effective contact coefficient equal
            # that draw instead of averaging it with PhysX's default 0.5.
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        ),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(project_root() / LINE_CONFIG["robot"]["usd_asset"]),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(*LINE_CONFIG["track"]["start_xy_m"], LINE_CONFIG["robot"]["base_height_m"]),
            rot=(
                math.cos(math.radians(LINE_CONFIG["track"]["start_heading_deg"]) / 2.0),
                0.0, 0.0,
                math.sin(math.radians(LINE_CONFIG["track"]["start_heading_deg"]) / 2.0),
            ),
        ),
        actuators={
            # The target velocity written by DutyAction is the no-load speed at
            # the applied armature voltage.  An explicit zero-stiffness drive
            # with this damping therefore produces the physical DC motor line
            # rather than closing a wheel-speed loop.
            "wheels": IdealPDActuatorCfg(
                joint_names_expr=list(LINE_CONFIG["robot"]["wheel_joints"]),
                effort_limit=float(LINE_CONFIG["motor"]["stall_torque_nm"]),
                velocity_limit_sim=float(LINE_CONFIG["robot"]["wheel_drive"]["max_velocity_radps"]),
                stiffness=0.0,
                damping=nominal_motor_damping(LINE_CONFIG),
                # Override SolidWorks' 0.1 N m joint friction.  The per-episode
                # draw is applied in randomize_scenario() after reset.
                friction=float(LINE_CONFIG["motor"]["coulomb_friction_nm"]),
                dynamic_friction=float(LINE_CONFIG["motor"]["coulomb_friction_nm"]),
                viscous_friction=0.0,
            ),
        },
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """One action term: signed duty per side."""

    duty = mdp.DutyActionCfg(
        asset_name="robot",
        blind_duty_scale=float(LINE_CONFIG["rl"]["blind_duty_scale"]),
    )


@configclass
class ObservationsCfg:
    """Frozen actor ABI, plus simulator-only state for the PPO critic.

    The critic is never exported, so giving it the hidden state that decides
    rewards and terminations costs nothing on hardware and makes the value
    function well-posed: two identical camera/encoder observations can other-
    wise carry different remaining distance and different terminal returns.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        line = ObsTerm(func=mdp.line_observation)

        def __post_init__(self) -> None:
            # The seven values are already a single packed vector in ABI order.
            # Isaac Lab must not reorder or re-normalise them.
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        state = ObsTerm(func=mdp.critic_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Reset-time domain randomization; see ``mdp/events.py`` for what is missing.

    Mass and centre of mass use Isaac Lab's own terms rather than the scenario
    dict, because they act on the articulation instead of on the projection.
    They matter for a direct-duty policy specifically: with no inner loop to
    absorb it, a heavier or nose-heavy chassis changes how a given duty
    accelerates the robot, and the policy has to have met that spread.
    """

    scenario = EventTerm(func=mdp.randomize_scenario, mode="reset")
    start_pose = EventTerm(func=mdp.reset_robot_on_track, mode="reset")
    base_mass = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_mass, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": tuple(LINE_CONFIG["randomization"]["base_mass_scale"]),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_com, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {
                "x": tuple(LINE_CONFIG["randomization"]["base_com_offset_m"]),
                "y": tuple(LINE_CONFIG["randomization"]["base_com_offset_m"]),
                "z": (0.0, 0.0),
            },
        },
    )
    wheel_material = EventTerm(
        func=isaaclab_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel_.*"]),
            "static_friction_range": tuple(LINE_CONFIG["randomization"]["static_friction"]),
            "dynamic_friction_range": tuple(LINE_CONFIG["randomization"]["dynamic_friction"]),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class RewardsCfg:
    """Completion-shaped reward; all scales live in ``default.json``."""

    progress = RewTerm(
        func=mdp.normalized_track_progress,
        weight=float(RL_REWARD_CONFIG["progress_target_step"]),
        params={"max_fraction": float(RL_REWARD_CONFIG["progress_max_fraction"])},
    )
    lateral = RewTerm(
        func=mdp.lateral_error_penalty,
        weight=float(RL_REWARD_CONFIG["lateral_error"]),
        params={"scale_m": float(RL_REWARD_CONFIG["lateral_scale_m"])},
    )
    heading = RewTerm(func=mdp.heading_alignment, weight=float(RL_REWARD_CONFIG["heading_error"]))
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=float(RL_REWARD_CONFIG["action_rate"]))
    duty_excess = RewTerm(
        func=mdp.duty_excess_penalty, weight=float(RL_REWARD_CONFIG["duty_excess"]),
    )
    time = RewTerm(func=mdp.time_penalty, weight=float(RL_REWARD_CONFIG["time"]))
    finish = RewTerm(func=mdp.reached_finish_bonus, weight=float(RL_REWARD_CONFIG["finish"]))
    failure = RewTerm(func=mdp.terminal_failure_penalty, weight=float(RL_REWARD_CONFIG["failure"]))


@configclass
class TerminationsCfg:
    """The gate's own limits, not softer ones."""

    time_out = DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True)
    line_lost = DoneTerm(func=mdp.line_lost)
    off_track = DoneTerm(func=mdp.off_track)
    unstable = DoneTerm(func=mdp.unstable_physics)
    stalled = DoneTerm(func=mdp.stalled)
    finished = DoneTerm(func=mdp.reached_finish)


@configclass
class LineFollowingEnvCfg(ManagerBasedRLEnvCfg):
    """Direct-duty line following with a geometric stand-in for the camera."""

    line_config: dict = LINE_CONFIG
    wheel_dof_indices: list[int] = None
    encoder_dof_slots: list[int] = None
    camera_stride: int = 1

    scene: LineFollowingSceneCfg = LineFollowingSceneCfg(num_envs=1024, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        physics = self.line_config["physics"]
        self.sim.dt = float(physics["dt_s"])
        physics_hz = 1.0 / self.sim.dt
        policy_hz = float(physics["policy_hz"])
        camera_hz = float(physics["camera_hz"])
        # PhysX remains at its accurate fixed rate while each RL action is held
        # for an exact integer number of physics steps.  The camera stride is in
        # policy ticks, so it must also be exact rather than silently rounded.
        self.decimation = integer_rate_stride(physics_hz, policy_hz, label="physics-to-policy")
        self.camera_stride = integer_rate_stride(policy_hz, camera_hz, label="policy-to-camera")
        termination_config = self.line_config["rl"]["terminations"]
        if float(termination_config["stall_timeout_s"]) <= 0.0:
            raise ValueError("rl.terminations.stall_timeout_s must be positive.")
        if float(termination_config["stall_progress_m"]) <= 0.0:
            raise ValueError("rl.terminations.stall_progress_m must be positive.")
        training_config = self.line_config["rl"]["training"]
        progress_range = [float(value) for value in training_config["start_progress_fraction"]]
        if len(progress_range) != 2 or not 0.0 <= progress_range[0] <= progress_range[1] < 1.0:
            raise ValueError("rl.training.start_progress_fraction must satisfy 0 <= low <= high < 1.")
        full_route_probability = float(training_config["full_route_start_probability"])
        if not 0.0 <= full_route_probability <= 1.0:
            raise ValueError("rl.training.full_route_start_probability must be in [0, 1].")
        # The terminal weights have to outweigh the progress stream, or early
        # termination dominates the start-to-finish objective.  Compare against
        # the *discounted* ceiling: PPO optimises a discounted return, and the
        # per-step progress term is clamped at ``progress_max_fraction``, so the
        # most progress any route can return is that clamp summed over the
        # discount horizon.  Sizing the terminal weights against the raw route
        # length instead -- track length over one target-speed step -- makes
        # them roughly an order of magnitude too large, and an oversized
        # terminal bonus under discounting pays for reaching the finish
        # *sooner*, which is a bounty on speed rather than on line following.
        rewards_config = self.line_config["rl"]["rewards"]
        gamma = float(self.line_config["rl"]["ppo"]["gamma"])
        if not 0.0 < gamma < 1.0:
            raise ValueError("rl.ppo.gamma must be in (0, 1).")
        maximum_progress_return = (
            float(rewards_config["progress_target_step"])
            * float(rewards_config["progress_max_fraction"])
            / (1.0 - gamma)
        )
        for terminal_name in ("finish", "failure"):
            terminal_weight = float(rewards_config[terminal_name])
            if terminal_weight <= maximum_progress_return:
                raise ValueError(
                    f"rl.rewards.{terminal_name}={terminal_weight:g} must exceed the discounted "
                    f"maximum progress return {maximum_progress_return:.1f} "
                    f"(progress_target_step * progress_max_fraction / (1 - gamma)); otherwise "
                    "early termination can dominate the start-to-finish objective."
                )
        self.episode_length_s = float(self.line_config["physics"]["maximum_duration_s"])
        self.sim.render_interval = self.decimation
        # Front wheels carry the encoders; all four are driven.  The slots index
        # into wheel_joints, not into the articulation's full DOF list.
        joints = self.line_config["robot"]["wheel_joints"]
        encoders = self.line_config["robot"]["encoder_joints"]
        self.encoder_dof_slots = [joints.index(name) for name in encoders]
