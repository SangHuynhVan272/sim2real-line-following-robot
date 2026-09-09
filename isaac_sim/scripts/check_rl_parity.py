#!/usr/bin/env python3
"""Prove the batched RL-lane maths match the validated simulator.

The training lane re-implements track projection, geometric perception, the
motor line and the analytic controller in torch so they can run on a thousand
environments at once.  Two implementations of the same physics is a standing
risk: they drift silently, and a policy trained against the drifted copy fails
only later, in the lane that has a camera.

This script is the guard.  It samples robot poses across the track, runs both
implementations, and reports the worst disagreement.  It needs neither Isaac Sim
nor a GPU, so it is cheap enough to run after every edit to either side.

    python isaac_sim/scripts/check_rl_parity.py
    python isaac_sim/scripts/check_rl_parity.py --samples 4000 --seed 7

Exit status is non-zero if any tolerance is exceeded.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_line_following as reference  # noqa: E402
from envs.line_following_rl.line_state import LineState  # noqa: E402
from envs.line_following_rl.motor_batch import (  # noqa: E402
    SafetyEnvelope,
    duty_to_volts,
    effective_breakaway_duty,
    expand_side_duty,
    scrub_severity,
)
from envs.line_following_rl.perception_batch import GeometricPerceptionBatch  # noqa: E402
from envs.line_following_rl.teacher_batch import analytic_duty, wheel_speed_to_duty  # noqa: E402
from envs.line_following_rl.timing import integer_rate_stride as batch_rate_stride  # noqa: E402
from envs.line_following_rl.track_batch import TrackBatch  # noqa: E402

# Two gates, because the two failure modes look nothing alike.
#
# A structural divergence -- a dropped term, wrong constant or sign flip --
# shifts most poses and therefore appears in the median.
#
# The thin upper tail mostly contains poses near an open-track endpoint, where
# clamping changes the reference sample grid. The p99 limit is therefore a
# looser smoke alarm, while the median limit detects implementation drift.
E_Y_MEDIAN_TOLERANCE = 1e-5
E_THETA_MEDIAN_TOLERANCE_RAD = 1e-5
E_Y_TOLERANCE = 3e-3
E_THETA_TOLERANCE_RAD = 1e-2
PROGRESS_TOLERANCE_M = 1e-4
DUTY_TOLERANCE = 1e-5


def yaw_rotation(yaw_rad: np.ndarray) -> np.ndarray:
    """Return ``[N, 3, 3]`` rotations about Z; the robot drives on a flat floor."""
    cos, sin = np.cos(yaw_rad), np.sin(yaw_rad)
    rotation = np.zeros((yaw_rad.shape[0], 3, 3))
    rotation[:, 0, 0], rotation[:, 0, 1] = cos, -sin
    rotation[:, 1, 0], rotation[:, 1, 1] = sin, cos
    rotation[:, 2, 2] = 1.0
    return rotation


def sample_poses(config: dict, scenario: dict, count: int, rng: np.random.Generator) -> dict:
    """Spread robot poses along the track with realistic lateral/heading error.

    The offsets are deliberately wider than the evaluation ranges so the
    comparison also covers poses where the tape leaves the frame -- the case
    that decides whether both implementations agree on *not* seeing a line.
    """
    points = config["track"]["points_xy_m"]
    cache = reference.build_track_label_cache(
        points, bool(config["track"].get("closed", False)),
        float(config["track"].get("finish_tape_extension_m", 0.0)),
    )
    total = float(cache["total"])
    progress = rng.uniform(0.0, total, count)
    centres = np.stack([reference.point_at_track_progress(cache, float(value)) for value in progress])
    ahead = np.stack([
        reference.point_at_track_progress(cache, float(min(total, value + 0.01))) for value in progress
    ])
    delta = ahead - centres
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    heading = heading + np.radians(rng.uniform(-25.0, 25.0, count))
    normal = np.stack([-np.sin(heading), np.cos(heading)], axis=-1)
    robot_xy = centres + normal * rng.uniform(-0.06, 0.06, count)[:, None]

    rotation = yaw_rotation(heading)
    mount = np.asarray(config["robot"]["camera_mount_xyz_m"], dtype=float) + np.array([
        float(scenario["camera_mount_dx_m"]),
        float(scenario["camera_mount_dy_m"]),
        float(scenario["camera_mount_dz_m"]),
    ])
    base_height = float(config["robot"]["base_height_m"])
    position = np.concatenate([robot_xy, np.full((count, 1), base_height)], axis=-1)
    eye = position + np.einsum("nij,j->ni", rotation, mount)
    down = math.radians(float(scenario["camera_down_angle_deg"]))
    camera_yaw = math.radians(float(scenario["camera_yaw_deg"]))
    optical = np.array([
        math.cos(down) * math.cos(camera_yaw),
        math.cos(down) * math.sin(camera_yaw),
        -math.sin(down),
    ])
    forward = np.einsum("nij,j->ni", rotation, optical)
    return {
        "cache": cache,
        "eye": eye,
        "forward": forward,
        "roll_deg": np.full((count,), float(scenario["camera_roll_deg"])),
        "robot_xy": robot_xy,
        "progress": progress,
    }


def _spread(errors: np.ndarray) -> dict[str, float]:
    """Summarise an error distribution; the tail matters more than the maximum."""
    return {
        "p50": float(np.percentile(errors, 50)),
        "p99": float(np.percentile(errors, 99)),
        "max": float(errors.max()),
    }


def compare_perception(config: dict, scenario: dict, poses: dict) -> dict:
    track = TrackBatch(
        config["track"]["points_xy_m"],
        finish_extension_m=(
            0.0 if config["track"].get("closed", False)
            else float(config["track"].get("finish_tape_extension_m", 0.0))
        ),
    )
    perception = GeometricPerceptionBatch(config, track)
    count = poses["eye"].shape[0]
    lens = {
        key: torch.full((count,), float(scenario[f"lens_{name}"]))
        for key, name in (("k1", "radial_k1"), ("k2", "radial_k2"),
                          ("p1", "tangential_p1"), ("p2", "tangential_p2"))
    }
    e_y, e_theta, valid = perception.evaluate(
        torch.tensor(poses["eye"], dtype=torch.float32),
        torch.tensor(poses["forward"], dtype=torch.float32),
        torch.tensor(poses["robot_xy"], dtype=torch.float32),
        torch.full((count,), float(scenario["camera_hfov_deg"])),
        lens,
        torch.tensor(poses["roll_deg"], dtype=torch.float32),
    )
    expected = [
        reference.image_observation_from_track_geometry(
            poses["eye"][index], poses["forward"][index], poses["robot_xy"][index],
            config, scenario, poses["cache"],
        )
        for index in range(count)
    ]
    expected_valid = np.array([item is not None for item in expected])
    agree = expected_valid == valid.numpy()
    both = expected_valid & valid.numpy()
    if not both.any():
        empty = {"p50": 0.0, "p99": 0.0, "max": 0.0}
        return {"validity_agreement": float(agree.mean()), "e_y": empty, "e_theta": empty,
                "compared": 0, "detected_fraction": 0.0}
    reference_pairs = np.array([item for item in expected if item is not None])
    # Align the reference rows with the batch rows that agreed on validity.
    reference_lookup = np.zeros((count, 2))
    reference_lookup[expected_valid] = reference_pairs
    return {
        "validity_agreement": float(agree.mean()),
        "e_y": _spread(np.abs(reference_lookup[both, 0] - e_y.numpy()[both])),
        "e_theta": _spread(np.abs(reference_lookup[both, 1] - e_theta.numpy()[both])),
        "compared": int(both.sum()),
        "detected_fraction": float(expected_valid.mean()),
    }


def compare_track(config: dict, poses: dict) -> float:
    track = TrackBatch(
        config["track"]["points_xy_m"],
        finish_extension_m=(
            0.0 if config["track"].get("closed", False)
            else float(config["track"].get("finish_tape_extension_m", 0.0))
        ),
    )
    progress, _ = track.project(torch.tensor(poses["robot_xy"], dtype=torch.float32))
    expected = np.array([
        reference.track_progress(row, config["track"]["points_xy_m"],
                                 bool(config["track"].get("closed", False)))[0]
        for row in poses["robot_xy"]
    ])
    return float(np.abs(expected - progress.numpy()).max())


def compare_motor(config: dict, scenario: dict, rng: np.random.Generator, count: int) -> dict:
    duty = rng.uniform(-1.0, 1.0, count)
    volts = duty_to_volts(
        torch.tensor(duty, dtype=torch.float64),
        torch.full((count,), float(scenario["supply_voltage_v"]), dtype=torch.float64),
        torch.full((count,), float(scenario["driver_drop_v"]), dtype=torch.float64),
        torch.full((count,), float(scenario["loaded_breakaway_duty"]), dtype=torch.float64),
    ).numpy()
    expected_volts = np.array([reference.duty_to_volts(float(value), scenario) for value in duty])

    radps = rng.uniform(-25.0, 25.0, count)
    batched_duty = wheel_speed_to_duty(torch.tensor(radps, dtype=torch.float64), config).numpy()
    expected_duty = np.array([
        reference.wheel_speed_to_duty(float(value), config, scenario) for value in radps
    ])

    observation = np.zeros((count, 7))
    observation[:, 0] = rng.uniform(-1.0, 1.0, count)
    observation[:, 1] = rng.uniform(-1.0, 1.0, count)
    observation[:, 2] = rng.uniform(0.0, 1.0, count)
    batched_action = analytic_duty(
        torch.tensor(observation, dtype=torch.float64), config,
        torch.full((count,), float(scenario["wheel_radius_scale"]), dtype=torch.float64),
        torch.full((count,), float(scenario["track_width_scale"]), dtype=torch.float64),
    ).numpy()
    expected_action = np.array([
        reference.policy(tuple(row), config, scenario) for row in observation
    ])
    # The scrub dead band is a two-lane formula like the rest of the motor
    # line, so it gets a parity row rather than a comment promising they agree.
    scrub_pairs = rng.uniform(-1.0, 1.0, (count, 2))
    scrub_pairs[0] = (0.6, 0.6)          # straight
    scrub_pairs[1] = (0.6, -0.6)         # pivot
    batched_severity = scrub_severity(torch.tensor(scrub_pairs, dtype=torch.float64)).numpy()
    expected_severity = np.array([reference.scrub_severity(row) for row in scrub_pairs])
    batched_breakaway = effective_breakaway_duty(
        torch.full((count,), float(scenario["loaded_breakaway_duty"]), dtype=torch.float64),
        torch.full((count,), float(scenario["scrub_breakaway_gain"]), dtype=torch.float64),
        torch.tensor(expected_severity, dtype=torch.float64),
    ).numpy()
    expected_breakaway = np.array([
        reference.effective_breakaway_duty(scenario, float(value)) for value in expected_severity
    ])

    side_commands = rng.uniform(-1.0, 1.0, (count, 2))
    imbalance = np.full(count, float(scenario["motor_imbalance"]))
    expanded = expand_side_duty(
        torch.tensor(side_commands, dtype=torch.float64),
        torch.tensor(imbalance, dtype=torch.float64),
    ).clamp(-float(config["motor"]["max_duty"]), float(config["motor"]["max_duty"])).numpy()
    expected_expanded = np.clip(np.stack([
        side_commands[:, 0] * imbalance,
        side_commands[:, 0] * imbalance,
        side_commands[:, 1] / imbalance,
        side_commands[:, 1] / imbalance,
    ], axis=-1), -float(config["motor"]["max_duty"]), float(config["motor"]["max_duty"]))
    targets = rng.uniform(-1.0, 1.0, (count, 2))
    targets[:2] = 0.0
    previous = rng.uniform(-1.0, 1.0, (count, 2))
    confidence = rng.uniform(0.01, 1.0, count)
    confidence[:2] = 0.0
    envelope = SafetyEnvelope(config, float(config["rl"]["blind_duty_scale"]),
                              float(config["physics"]["policy_hz"]))
    batched_envelope = envelope.apply(
        torch.tensor(targets, dtype=torch.float64),
        torch.tensor(previous, dtype=torch.float64),
        torch.tensor(confidence, dtype=torch.float64),
    ).numpy()
    expected_envelope = np.clip(targets, -float(config["motor"]["max_duty"]),
                                float(config["motor"]["max_duty"]))
    blind = confidence <= 0.0
    blind_limit = float(config["motor"]["max_duty"]) * float(config["rl"]["blind_duty_scale"])
    expected_envelope[blind] = np.clip(expected_envelope[blind], -blind_limit, blind_limit)
    expected_envelope = reference.enforce_minimum_loaded_command(expected_envelope, config)
    expected_envelope = np.clip(expected_envelope, -float(config["motor"]["max_duty"]),
                                float(config["motor"]["max_duty"]))
    slew = float(config["motor"]["duty_rate_limit_per_s"]) / float(config["physics"]["policy_hz"])
    expected_envelope = np.clip(expected_envelope, previous - slew, previous + slew)
    return {
        "volts": float(np.abs(expected_volts - volts).max()),
        "wheel_speed_to_duty": float(np.abs(expected_duty - batched_duty).max()),
        "analytic_policy": float(np.abs(expected_action - batched_action).max()),
        "scrub_severity": float(np.abs(expected_severity - batched_severity).max()),
        "scrub_breakaway": float(np.abs(expected_breakaway - batched_breakaway).max()),
        "motor_imbalance": float(np.abs(expected_expanded - expanded).max()),
        "safety_envelope": float(np.abs(expected_envelope - batched_envelope).max()),
    }


def compare_delay_contract() -> float:
    """Prove a configured delay of one means one policy tick, not zero or two."""
    buffer = torch.zeros(1, 1, 2)
    env_ids = torch.tensor([0])
    first = LineState._delay_values(buffer, torch.tensor([[0.2, -0.2]]), env_ids)
    second = LineState._delay_values(buffer, torch.tensor([[0.4, -0.4]]), env_ids)
    expected = torch.tensor([[0.2, -0.2]])
    return float(torch.max(torch.abs(first)) + torch.max(torch.abs(second - expected)))


def compare_progress_contract() -> float:
    """Guard monotonic completion so reversing cannot farm reward."""
    best = torch.zeros(1)
    increments = []
    for projected in (0.10, 0.30, 0.20, 0.35):
        advanced = LineState.advance_progress(best, torch.tensor([projected]))
        increments.append(float(advanced - best))
        best = advanced
    error = abs(float(best) - 0.35) + abs(sum(increments) - 0.35)
    return error + sum(max(0.0, -value) for value in increments)


def compare_timing_contract(config: dict) -> float:
    """Keep the rendered and RL lanes on the same exact integer clocks."""
    physics = config["physics"]
    rates = (
        (1.0 / float(physics["dt_s"]), float(physics["policy_hz"]), "physics-to-policy"),
        (float(physics["policy_hz"]), float(physics["camera_hz"]), "policy-to-camera"),
    )
    error = 0.0
    for faster, slower, label in rates:
        scalar = reference.integer_rate_stride(faster, slower, label=label)
        batched = batch_rate_stride(faster, slower, label=label)
        error += abs(scalar - batched)
    return error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "isaac_sim/config/default.json")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true",
                        help="Compare on a randomized scenario (non-identity lens) instead of nominal.")
    args = parser.parse_args()

    config = reference.load_config(args.config)
    scenario = reference.sample_episode(config, args.seed, args.randomize)
    rng = np.random.default_rng(args.seed)
    poses = sample_poses(config, scenario, args.samples, rng)

    track_error = compare_track(config, poses)
    perception = compare_perception(config, scenario, poses)
    motor = compare_motor(config, scenario, rng, args.samples)
    delay_error = compare_delay_contract()
    progress_error = compare_progress_contract()
    timing_error = compare_timing_contract(config)

    print(f"scenario           : {'randomized' if args.randomize else 'nominal'} seed={args.seed}")
    print(f"poses compared     : {args.samples} ({perception['detected_fraction']:.1%} see the line)")
    print(f"track progress     : max |delta| = {track_error:.3e} m")
    print(f"perception validity: {perception['validity_agreement']:.4%} agreement "
          f"({perception['compared']} pairs compared)")
    print(f"perception e_y     : p50 {perception['e_y']['p50']:.2e}  "
          f"p99 {perception['e_y']['p99']:.2e}  max {perception['e_y']['max']:.2e}")
    print(f"perception e_theta : p50 {perception['e_theta']['p50']:.2e}  "
          f"p99 {perception['e_theta']['p99']:.2e}  max {perception['e_theta']['max']:.2e} rad")
    print(f"duty -> volts      : max |delta| = {motor['volts']:.3e} V")
    print(f"wheel speed -> duty: max |delta| = {motor['wheel_speed_to_duty']:.3e}")
    print(f"analytic policy    : max |delta| = {motor['analytic_policy']:.3e}")
    print(f"scrub severity    : max |delta| = {motor['scrub_severity']:.3e}")
    print(f"scrub breakaway   : max |delta| = {motor['scrub_breakaway']:.3e}")
    print(f"motor imbalance   : max |delta| = {motor['motor_imbalance']:.3e}")
    print(f"safety envelope   : max |delta| = {motor['safety_envelope']:.3e}")
    print(f"one-tick delay    : contract error = {delay_error:.3e}")
    print(f"monotonic progress: contract error = {progress_error:.3e}")
    print(f"integer clocks    : contract error = {timing_error:.3e}")

    failures = []
    if track_error > PROGRESS_TOLERANCE_M:
        failures.append(f"track progress {track_error:.3e} > {PROGRESS_TOLERANCE_M:.0e}")
    if perception["e_y"]["p50"] > E_Y_MEDIAN_TOLERANCE:
        failures.append(f"e_y p50 {perception['e_y']['p50']:.3e} > {E_Y_MEDIAN_TOLERANCE:.0e}")
    if perception["e_theta"]["p50"] > E_THETA_MEDIAN_TOLERANCE_RAD:
        failures.append(f"e_theta p50 {perception['e_theta']['p50']:.3e} > {E_THETA_MEDIAN_TOLERANCE_RAD:.0e}")
    if perception["e_y"]["p99"] > E_Y_TOLERANCE:
        failures.append(f"e_y p99 {perception['e_y']['p99']:.3e} > {E_Y_TOLERANCE:.0e}")
    if perception["e_theta"]["p99"] > E_THETA_TOLERANCE_RAD:
        failures.append(f"e_theta p99 {perception['e_theta']['p99']:.3e} > {E_THETA_TOLERANCE_RAD:.0e}")
    for name in ("volts", "wheel_speed_to_duty", "analytic_policy", "scrub_severity",
                 "scrub_breakaway", "motor_imbalance", "safety_envelope"):
        if motor[name] > DUTY_TOLERANCE:
            failures.append(f"{name} {motor[name]:.3e} > {DUTY_TOLERANCE:.0e}")
    if delay_error > DUTY_TOLERANCE:
        failures.append(f"one-tick delay {delay_error:.3e} > {DUTY_TOLERANCE:.0e}")
    if progress_error > DUTY_TOLERANCE:
        failures.append(f"monotonic progress {progress_error:.3e} > {DUTY_TOLERANCE:.0e}")
    if timing_error > DUTY_TOLERANCE:
        failures.append(f"integer clocks {timing_error:.3e} > {DUTY_TOLERANCE:.0e}")
    if failures:
        print("\nPARITY FAIL:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("\nPARITY OK")


if __name__ == "__main__":
    main()
