#!/usr/bin/env python3
"""PhysX camera-in-the-loop line-following episode for the ESP32-S3 contract.

The controller is intentionally small: adaptive dark-tape thresholding in two
image ROIs, then a P steering law.  Physics, camera rendering, actual wheel
RPM, motor lag, and encoder corruption are simulated; the policy itself
remains portable to plain C/C++ on the ESP32-S3.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from policy_header import load_header_policy
from checkpoint_contract import validate_checkpoint_config


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    asset = Path(config["robot"]["usd_asset"])
    config["robot"]["usd_asset"] = asset if asset.is_absolute() else project_root() / asset
    return config


def integer_rate_stride(faster_hz: float, slower_hz: float, *, label: str) -> int:
    """Return an exact clock stride; never silently change a configured rate."""
    if faster_hz <= 0.0 or slower_hz <= 0.0:
        raise ValueError(f"{label} rates must be positive, got {faster_hz} and {slower_hz} Hz.")
    ratio = faster_hz / slower_hz
    stride = round(ratio)
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(
            f"{label} requires an integer rate ratio, got "
            f"{faster_hz:g} / {slower_hz:g} = {ratio:g}."
        )
    return int(stride)


def sample_range(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def sample_episode(config: dict, seed: int | None, randomize: bool) -> dict:
    """Return deterministic nominal or domain-randomized parameters."""
    robot, track, motor = config["robot"], config["track"], config["motor"]
    if not randomize:
        return {
            "seed": seed, "camera_down_angle_deg": robot["camera_down_angle_deg"],
            "camera_hfov_deg": robot["camera_hfov_deg"], "tape_width_m": track["tape_width_m"],
            "start_lateral_m": 0.0, "start_heading_deg": 0.0, "static_friction": 0.9,
            "dynamic_friction": 0.7, "wheel_radius_scale": 1.0, "track_width_scale": 1.0,
            "supply_voltage_v": motor["supply_voltage_v"], "driver_drop_v": motor["driver_drop_v"],
            "no_load_wheel_radps": motor["no_load_wheel_radps"], "stall_torque_nm": motor["stall_torque_nm"],
            "coulomb_friction_nm": motor["coulomb_friction_nm"],
            "loaded_breakaway_duty": motor["loaded_breakaway_duty"],
            "scrub_breakaway_gain": motor["scrub_breakaway_gain"], "motor_imbalance": 1.0,
            "encoder_scale": 1.0, "tape_brightness": 0.01, "tape_roughness": 0.55,
            "floor_luma": 0.82, "floor_tint": [1.0, 1.0, 1.0], "floor_roughness": 0.82,
            "floor_dirt_patches": 0, "floor_dirt_size_m": 0.02, "floor_dirt_luma_scale": 1.0,
            "dome_intensity": 500.0, "key_intensity": 1200.0,
            # Nominal is the calibrated ideal sensor.  The following fields
            # remain in every scenario record so a real OV2640 measurement can
            # later replace the assumed randomized ranges without changing the
            # episode schema.
            "lens_radial_k1": 0.0, "lens_radial_k2": 0.0,
            "lens_tangential_p1": 0.0, "lens_tangential_p2": 0.0,
            "motion_blur_mix": 0.0, "jpeg_quality": 100,
            "exposure_ev_bias": 0.0, "auto_gain_drift_ev": 0.0,
            "auto_gain_drift_hz": 0.0, "auto_gain_phase_rad": 0.0,
            "rolling_shutter_fraction": 0.0,
            # Build tolerance.  Nominal is the drawing; a randomized episode is
            # a robot somebody actually assembled.
            "camera_mount_dx_m": 0.0, "camera_mount_dy_m": 0.0, "camera_mount_dz_m": 0.0,
            "camera_yaw_deg": 0.0, "camera_roll_deg": 0.0,
            "base_mass_scale": 1.0, "base_com_offset_m": 0.0, "base_com_offset_y_m": 0.0,
            # Nominal takes the deepest delay, matching the training lane: the
            # buffers are sized from the same maximum.
            "camera_delay_ticks": int(max(config["randomization"]["camera_delay_ticks"]))
            if isinstance(config["randomization"]["camera_delay_ticks"], (list, tuple))
            else int(config["randomization"]["camera_delay_ticks"]),
            "actuation_delay_ticks": int(max(config["randomization"]["actuation_delay_ticks"]))
            if isinstance(config["randomization"]["actuation_delay_ticks"], (list, tuple))
            else int(config["randomization"]["actuation_delay_ticks"]),
            "encoder_noise_rpm": float(max(config["randomization"]["encoder_noise_rpm"]))
            if isinstance(config["randomization"]["encoder_noise_rpm"], (list, tuple))
            else float(config["randomization"]["encoder_noise_rpm"]),
            "viscous_friction_nm_per_radps": float(
                config["motor"]["viscous_friction_nm_per_radps"]),
            "supply_sag_fraction": 0.0,
        }
    rng = np.random.default_rng(seed)
    dr = config["randomization"]
    scenario = {
        "seed": seed,
        "camera_down_angle_deg": sample_range(rng, dr["camera_down_angle_deg"]),
        "camera_hfov_deg": sample_range(rng, dr["camera_hfov_deg"]),
        "tape_width_m": sample_range(rng, dr["tape_width_m"]),
        "start_lateral_m": sample_range(rng, dr["start_lateral_m"]),
        "start_heading_deg": sample_range(rng, dr["start_heading_deg"]),
        "static_friction": sample_range(rng, dr["static_friction"]),
        "dynamic_friction": sample_range(rng, dr["dynamic_friction"]),
        # Drawn, then discarded.  These scale only the analytic teacher's own
        # formula -- the imported wheel collision and joint frames never move --
        # so randomizing them adds label noise without randomizing the plant.
        # Consume the draw to preserve the deterministic mapping between seed
        # and every later camera/motor randomization field.
        "wheel_radius_scale": (sample_range(rng, dr["wheel_radius_scale"]), 1.0)[1],
        "track_width_scale": (sample_range(rng, dr["track_width_scale"]), 1.0)[1],
        # Electrical actuation: duty maps to a torque/speed line, so the spread
        # that matters is supply voltage, driver drop, and motor constants --
        # not an abstract RPM gain.
        "supply_voltage_v": sample_range(rng, dr["supply_voltage_v"]),
        "driver_drop_v": sample_range(rng, dr["driver_drop_v"]),
        "no_load_wheel_radps": sample_range(rng, dr["no_load_wheel_radps"]),
        "stall_torque_nm": sample_range(rng, dr["stall_torque_nm"]),
        "coulomb_friction_nm": sample_range(rng, dr["coulomb_friction_nm"]),
        "loaded_breakaway_duty": sample_range(rng, dr["loaded_breakaway_duty"]),
        "motor_imbalance": sample_range(rng, dr["motor_imbalance"]),
        "encoder_scale": sample_range(rng, dr["encoder_scale"]),
        "tape_brightness": sample_range(rng, dr["tape_brightness"]),
        "tape_roughness": sample_range(rng, dr["tape_roughness"]),
        # A real floor is rarely neutral white: it can be blue-grey lino, beige
        # tile, or scuffed concrete, and that shifts luma far more than a grey
        # brightness sweep does.
        "floor_luma": sample_range(rng, dr["floor_luma"]),
        "floor_tint": [sample_range(rng, dr["floor_tint"]) for _ in range(3)],
        "floor_roughness": sample_range(rng, dr["floor_roughness"]),
        "floor_dirt_patches": int(round(sample_range(rng, dr["floor_dirt_patches"]))),
        "floor_dirt_size_m": sample_range(rng, dr["floor_dirt_size_m"]),
        "floor_dirt_luma_scale": sample_range(rng, dr["floor_dirt_luma_scale"]),
        "dome_intensity": sample_range(rng, dr["dome_intensity"]),
        "key_intensity": sample_range(rng, dr["key_intensity"]),
        # Keep visual additions at the end so extending domain randomization
        # does not silently change established physics/material seed mappings.
        "lens_radial_k1": sample_range(rng, dr["lens_radial_k1"]),
        "lens_radial_k2": sample_range(rng, dr["lens_radial_k2"]),
        "lens_tangential_p1": sample_range(rng, dr["lens_tangential_p1"]),
        "lens_tangential_p2": sample_range(rng, dr["lens_tangential_p2"]),
        "motion_blur_mix": sample_range(rng, dr["motion_blur_mix"]),
        "jpeg_quality": int(round(sample_range(rng, dr["jpeg_quality"]))),
        "exposure_ev_bias": sample_range(rng, dr["exposure_ev_bias"]),
        "auto_gain_drift_ev": sample_range(rng, dr["auto_gain_drift_ev"]),
        "auto_gain_drift_hz": sample_range(rng, dr["auto_gain_drift_hz"]),
        "auto_gain_phase_rad": sample_range(rng, dr["auto_gain_phase_rad"]),
        "rolling_shutter_fraction": sample_range(rng, dr["rolling_shutter_fraction"]),
        # Build tolerance, appended last for the same reason as the visual
        # draws above. Printed brackets and camera assembly introduce residual
        # pose error, so randomize it instead of training against a perfect
        # mount that physical builds cannot reproduce exactly.
        "camera_mount_dx_m": sample_range(rng, dr["camera_mount_dx_m"]),
        "camera_mount_dy_m": sample_range(rng, dr["camera_mount_dy_m"]),
        "camera_mount_dz_m": sample_range(rng, dr["camera_mount_dz_m"]),
        "camera_yaw_deg": sample_range(rng, dr["camera_yaw_deg"]),
        "camera_roll_deg": sample_range(rng, dr["camera_roll_deg"]),
        "base_mass_scale": sample_range(rng, dr["base_mass_scale"]),
        "base_com_offset_m": sample_range(rng, dr["base_com_offset_m"]),
    }
    # Keep the same two RNG draws, but project impossible material
    # pairs onto PhysX's physical constraint.  The Isaac Lab material event
    # already does this with ``make_consistent=True``; the rendered lane must
    # describe the same contact contract.
    scenario["dynamic_friction"] = min(
        scenario["dynamic_friction"], scenario["static_friction"],
    )
    # Training profiles may choose deliberately realistic floor families (for
    # example warm brown, neutral grey, and blue-grey lino) rather than relying
    # only on three independent tint draws. Keep this optional draw after the
    # fixed fields above so adding a palette does not perturb their seed map.
    palettes = dr.get("floor_tint_palettes")
    if palettes is not None:
        if not isinstance(palettes, list) or not palettes:
            raise ValueError("randomization.floor_tint_palettes must be a non-empty list")
        palette_selection = str(dr.get("floor_tint_palette_selection", "random"))
        if palette_selection == "random":
            palette_index = int(rng.integers(len(palettes)))
        elif palette_selection == "seed_cycle":
            if seed is None:
                raise ValueError("floor_tint_palette_selection=seed_cycle requires an episode seed")
            palette_index = int(seed) % len(palettes)
        else:
            raise ValueError(
                "randomization.floor_tint_palette_selection must be random or seed_cycle",
            )
        selection = palettes[palette_index]
        if isinstance(selection, dict):
            tint_values = selection.get("tint")
            palette_name = str(selection.get("name", "unnamed"))
        else:
            tint_values = selection
            palette_name = "unnamed"
        if not isinstance(tint_values, list) or len(tint_values) != 3:
            raise ValueError("Each floor_tint_palettes item must contain exactly three tint values")
        jitter = float(dr.get("floor_tint_palette_jitter", 0.0))
        if jitter < 0.0:
            raise ValueError("randomization.floor_tint_palette_jitter must be non-negative")
        scenario["floor_tint"] = [
            float(max(0.01, float(value) + rng.uniform(-jitter, jitter)))
            for value in tint_values
        ]
        scenario["floor_palette_name"] = palette_name
    else:
        scenario["floor_palette_name"] = "continuous_random_tint"
    # Append new fields so earlier randomization fields keep their deterministic
    # seed mapping across revisions.
    scenario["scrub_breakaway_gain"] = sample_range(rng, dr["scrub_breakaway_gain"])
    # Appended at the end so every seed drawn before these existed still
    # reproduces: the generator is consumed in declaration order.
    scenario["camera_delay_ticks"] = round(sample_range(rng, dr["camera_delay_ticks"]))
    scenario["actuation_delay_ticks"] = round(sample_range(rng, dr["actuation_delay_ticks"]))
    scenario["encoder_noise_rpm"] = sample_range(rng, dr["encoder_noise_rpm"])
    scenario["viscous_friction_nm_per_radps"] = sample_range(
        rng, dr["viscous_friction_nm_per_radps"])
    scenario["supply_sag_fraction"] = sample_range(rng, dr["supply_sag_fraction"])
    # The training lane draws X and Y COM offsets independently. Append Y so
    # fields above keep their deterministic seed ordering.
    scenario["base_com_offset_y_m"] = sample_range(rng, dr["base_com_offset_m"])
    return scenario


def floor_color(scenario: dict) -> tuple[float, float, float]:
    """Tinted floor colour renormalized back to the sampled luma."""
    weights = (0.299, 0.587, 0.114)
    tint = scenario["floor_tint"]
    luma = sum(w * t for w, t in zip(weights, tint))
    scale = scenario["floor_luma"] / max(1e-6, luma)
    return tuple(float(min(1.0, max(0.0, t * scale))) for t in tint)


def adaptive_otsu_threshold_details(luminance: np.ndarray, vision: dict) -> tuple[float | None, str]:
    """Return an Otsu threshold and the guard that accepted or rejected it.

    A fixed cut fails as soon as the floor stops being white: blue lino, dusty
    concrete and a bright lamp all move the tape/floor pair bodily up or down
    the scale.  Otsu tracks that, and the guards below stop it from inventing a
    split in a frame that is uniformly floor (or uniformly dark).  Both the
    histogram and the guards are a single integer pass, so this ports to the
    ESP32-S3 unchanged.
    """
    counts = np.bincount(np.clip(luminance, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return None, "empty_histogram"
    levels = np.arange(256, dtype=np.float64)
    weight_low = np.cumsum(counts)
    weight_high = total - weight_low
    sum_low = np.cumsum(counts * levels)
    sum_total = sum_low[-1]
    valid = (weight_low > 0.0) & (weight_high > 0.0)
    if not valid.any():
        return None, "single_population"
    mean_low = np.where(valid, sum_low / np.where(weight_low > 0.0, weight_low, 1.0), 0.0)
    mean_high = np.where(valid, (sum_total - sum_low) / np.where(weight_high > 0.0, weight_high, 1.0), 0.0)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~valid] = -1.0
    threshold = float(np.argmax(between))
    split = int(threshold)
    # The two clusters must actually be separated, or Otsu is just cutting
    # noise in a single-population frame.
    if mean_high[split] - mean_low[split] < float(vision["adaptive_minimum_contrast_px"]):
        return None, "contrast_guard"
    dark_fraction = float(weight_low[split] / total)
    if dark_fraction < float(vision["adaptive_minimum_dark_fraction"]):
        return None, "dark_fraction_guard"
    if 1.0 - dark_fraction < float(vision["adaptive_minimum_bright_fraction"]):
        return None, "bright_fraction_guard"
    return float(np.clip(threshold, float(vision["adaptive_threshold_min_px"]),
                         float(vision["adaptive_threshold_max_px"]))), "accepted"


def adaptive_otsu_threshold(luminance: np.ndarray, vision: dict) -> float | None:
    """Otsu split of the frame histogram, or None when no tape can be present."""
    threshold, _guard = adaptive_otsu_threshold_details(luminance, vision)
    return threshold


def dark_mask_and_threshold_details(
    rgb: np.ndarray, vision: dict,
) -> tuple[np.ndarray, float | None, str] | None:
    """Return the dark-tape mask, threshold, and adaptive-threshold outcome."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    image = rgb[..., :3].astype(np.float32)
    if image.max(initial=0.0) <= 1.5:
        image *= 255.0
    # The OV2640 hands the ESP32 a Y channel, not an RGB mean.  On a blue or
    # beige floor the two differ enough to swallow the tape contrast, so match
    # the sensor's YUV luma here.
    weights = np.asarray(vision["luma_weights"], dtype=np.float32)
    luminance = image @ (weights / float(weights.sum()))
    # Otsu quantizes this same luma value to uint8 for its histogram.  Keep the
    # quantized image so the selected histogram bin can be classified without
    # a floating-point edge mismatch (for example 30 becoming 30.000000000004).
    luminance_bins = np.clip(luminance, 0.0, 255.0).astype(np.uint8)
    mode = str(vision.get("threshold_mode", "absolute"))
    if mode == "absolute":
        threshold: float | None = float(vision["black_threshold"])
        threshold_guard = "absolute"
    elif mode == "adaptive_otsu":
        threshold, threshold_guard = adaptive_otsu_threshold_details(luminance, vision)
    else:
        raise ValueError(f"Unknown vision.threshold_mode: {mode}")
    if threshold is None:
        return np.zeros(luminance.shape, dtype=bool), None, threshold_guard
    # Otsu chooses the last histogram bin belonging to the dark class.  Its
    # corresponding inverse threshold is therefore inclusive (``Y_bin <= T``).
    # The absolute threshold preserves its existing strict ``Y < T`` firmware
    # convention.  Treating both as strict silently removed a uniformly dark
    # tape when its pixels occupied Otsu's selected bin.
    if mode == "adaptive_otsu":
        return luminance_bins <= int(threshold), threshold, threshold_guard
    return luminance < threshold, threshold, threshold_guard


def dark_mask_and_threshold(rgb: np.ndarray, vision: dict) -> tuple[np.ndarray, float | None] | None:
    """Return the dark-tape mask and its threshold for controller diagnostics."""
    result = dark_mask_and_threshold_details(rgb, vision)
    if result is None:
        return None
    mask, threshold, _guard = result
    return mask, threshold


def black_mask(rgb: np.ndarray, vision: dict) -> np.ndarray | None:
    """Return the lightweight dark-tape mask shared by both controllers."""
    result = dark_mask_and_threshold(rgb, vision)
    return None if result is None else result[0]


def tape_run(pixels: np.ndarray, minimum_width: int, anchor: float | None) -> tuple[int, int] | None:
    """Contiguous dark run for one image strip.

    Picking a *run* instead of the mean of every dark column rejects renderer
    speckle, and anchoring to the previous strip keeps the trace on one branch
    of the tape when the far side of a closed track drifts into frame.
    """
    edges = np.flatnonzero(np.diff(np.r_[False, pixels, False]))
    runs = [(start, end) for start, end in zip(edges[::2], edges[1::2]) if end - start >= minimum_width]
    if not runs:
        return None
    if anchor is None:
        return max(runs, key=lambda run: run[1] - run[0])
    return min(runs, key=lambda run: abs((run[0] + run[1] - 1) / 2.0 - anchor))


def trace_centerline(mask: np.ndarray, vision: dict) -> tuple[list[float], list[float]]:
    """Walk the tape from the nearest strip upwards, returning (rows_px, centers_px).

    The walk stops at the first strip that loses the tape.  Anything above such
    a gap belongs to a different part of the track, so consuming it would make
    the fit describe a line the robot is not on.
    """
    height = mask.shape[0]
    band = int(vision["centerline_row_halfheight_px"])
    minimum_width = int(vision["minimum_run_width_px"])
    rows: list[float] = []
    centers: list[float] = []
    anchor: float | None = None
    for fraction in sorted(vision["centerline_rows_fraction"], reverse=True):
        row = int(round(fraction * (height - 1)))
        run = tape_run(mask[max(0, row - band):min(height, row + band + 1)].any(axis=0), minimum_width, anchor)
        if run is None:
            if rows:
                break
            continue
        anchor = (run[0] + run[1] - 1) / 2.0
        rows.append(float(row))
        centers.append(anchor)
    return rows, centers


def steering_errors(near_x: float, far_x: float, near_row: float, far_row: float, width: int) -> tuple[float, float]:
    """Pack two image measurements into the frozen [e_y, e_theta] ABI pair.

    Both errors are positive when the tape lies to the *right*, so the single
    ``steering_sign`` turns them into a yaw command without cancelling.
    """
    half_width = (width - 1) / 2.0
    lateral = (far_x - half_width) / half_width
    heading = math.atan2(far_x - near_x, max(1.0, near_row - far_row))
    return float(np.clip(lateral, -1.0, 1.0)), float(heading)


def image_observation_two_roi(mask: np.ndarray, vision: dict) -> tuple[float, float] | None:
    """Baseline: center the dark tape at two horizontal image strips."""
    height, width = mask.shape
    rows = [round(fraction * (height - 1)) for fraction in vision["roi_rows_fraction"]]
    minimum_width = int(vision["minimum_run_width_px"])
    centers: list[float] = []
    anchor: float | None = None
    for row in rows:
        run = tape_run(mask[max(0, row - 4):min(height, row + 5)].any(axis=0), minimum_width, anchor)
        if run is None:
            return None
        anchor = (run[0] + run[1] - 1) / 2.0
        centers.append(anchor)
    # The baseline steers on the near strip; the far strip only sets the heading.
    _lookahead, heading = steering_errors(centers[0], centers[1], rows[0], rows[1], width)
    half_width = (width - 1) / 2.0
    return float(np.clip((centers[0] - half_width) / half_width, -1.0, 1.0)), heading


def image_observation_centerline(mask: np.ndarray, vision: dict) -> tuple[float, float] | None:
    """Fit a local tape centerline and steer on a farther look-ahead point.

    Per-row contiguous runs keep this suitable for a C/C++ ESP32 implementation
    without OpenCV or an ML runtime.
    """
    height, width = mask.shape
    rows, centers = trace_centerline(mask, vision)
    if len(rows) < int(vision["minimum_centerline_points"]):
        return None
    # A quadratic through the strips that still see the tape diverges violently
    # outside their span, which pins e_y to +-1 for a whole corner.  Fit in
    # pixels and only ever evaluate *inside* the traced span.
    coefficients = np.polyfit(rows, centers, 2 if len(rows) >= 4 else 1)
    nearest, farthest = max(rows), min(rows)
    near_row = float(np.clip(vision["roi_rows_fraction"][0] * (height - 1), farthest, nearest))
    far_row = float(np.clip(vision["lookahead_row_fraction"] * (height - 1), farthest, nearest))
    return steering_errors(float(np.polyval(coefficients, near_row)),
                           float(np.polyval(coefficients, far_row)), near_row, far_row, width)


def image_observation(rgb: np.ndarray, config: dict) -> tuple[float, float] | None:
    """Dispatch the baseline or centerline-lookahead perception algorithm."""
    observation, _diagnostic, _mask = image_observation_with_diagnostics(rgb, config)
    return observation


def image_observation_with_diagnostics(
    rgb: np.ndarray, config: dict,
) -> tuple[tuple[float, float] | None, dict[str, float | int | str | None], np.ndarray | None]:
    """Run perception and expose why a camera sample was accepted or rejected.

    The extra metadata is deliberately observational: the policy still receives
    exactly the frozen error pair (or its held predecessor) and line-loss time
    remains driven solely by new camera samples.  It makes an intermittent loss
    auditable without turning a confidence value into the loss detector.
    """
    result = dark_mask_and_threshold_details(rgb, config["vision"])
    if result is None:
        return None, {
            "state": "invalid_frame", "threshold_px": None,
            "threshold_guard": "invalid_frame", "dark_fraction": 0.0, "trace_points": 0,
        }, None
    mask, threshold, threshold_guard = result
    diagnostics: dict[str, float | int | str | None] = {
        "state": "detected" if threshold is not None else "otsu_rejected",
        "threshold_px": threshold, "threshold_guard": threshold_guard,
        "dark_fraction": float(mask.mean()),
        "trace_points": 0,
    }
    if threshold is None:
        return None, diagnostics, mask
    algorithm = config["controller"].get("algorithm", "two_roi")
    if algorithm == "two_roi":
        observation = image_observation_two_roi(mask, config["vision"])
        if observation is None:
            diagnostics["state"] = "roi_run_missing"
        return observation, diagnostics, mask
    if algorithm == "centerline_lookahead":
        rows, _centers = trace_centerline(mask, config["vision"])
        diagnostics["trace_points"] = len(rows)
        observation = image_observation_centerline(mask, config["vision"])
        if observation is None:
            diagnostics["state"] = "centerline_gap"
        return observation, diagnostics, mask
    raise ValueError(f"Unknown controller.algorithm: {algorithm}")


def motor_curve(scenario: dict, motor: dict) -> tuple[float, float, float]:
    """Return (radps_per_volt, torque_nm_per_volt, damping) for the DC motor line.

    A brushed motor obeys ``tau = Kt*(V - Ke*w)/R``.  Rearranged that is a
    velocity drive whose damping is the constant ``stall/no_load`` slope and
    whose target is the no-load speed for the applied volts -- so PhysX
    reproduces the real torque/speed line exactly, with no PID anywhere.
    """
    reference = float(motor["measured_at_volts"])
    radps_per_volt = float(scenario["no_load_wheel_radps"]) / reference
    torque_per_volt = float(scenario["stall_torque_nm"]) / reference
    damping = torque_per_volt / radps_per_volt + float(
        scenario.get("viscous_friction_nm_per_radps",
                     motor["viscous_friction_nm_per_radps"]))
    return radps_per_volt, torque_per_volt, damping


def scrub_severity(duty):
    """How tight a turn a left/right duty pair asks for, on [0, 1].

    Skid steer has no steering geometry, so heading only changes by sliding all
    four tyres sideways.  What loads the motors is therefore the curvature the
    command asks for, not its magnitude: 0 for a straight run, 0.5 for turning
    about one stopped wheel, 1.0 for a pivot in place.
    """
    duty = np.asarray(duty, dtype=float)
    if duty.shape[-1] != 2:
        raise ValueError("scrub severity expects [..., 2] left/right duty")
    difference = np.abs(duty[..., 0] - duty[..., 1])
    common = np.abs(duty[..., 0] + duty[..., 1])
    return difference / np.maximum(difference + common, 1.0e-9)


def effective_breakaway_duty(scenario: dict, severity: float) -> float:
    """Duty a turning command must clear before the wheels move at all.

    ``loaded_breakaway_duty`` is the straight-line bench value. Charging the
    commanded turn's scrub load on top of it prevents the simulator from
    crediting a weak pivot with motion that the loaded physical drivetrain
    cannot produce, which would otherwise freeze the observation and command.
    """
    return float(scenario["loaded_breakaway_duty"]) + float(
        scenario["scrub_breakaway_gain"]) * float(severity)


def duty_to_volts(duty: float, scenario: dict, breakaway: float | None = None) -> float:
    """Effective armature volts for a PWM duty.

    The H-bridge drop is what creates the dead zone: below it the wheel does
    not turn at all, and because the drop is fixed while the pack sags, the
    dead zone widens as the battery drains.  ``breakaway`` overrides the
    straight-line dead band with the scrub-loaded one for the commanded turn.
    """
    # Bench testing on the fully loaded robot found that commands below this
    # duty do not move it.  This is a plant dead band, not an output clamp: the
    # policy still emits raw duty and the hardware naturally has the same load.
    if breakaway is None:
        breakaway = float(scenario["loaded_breakaway_duty"])
    if abs(duty) < breakaway:
        return 0.0
    supply, drop = float(scenario["supply_voltage_v"]), float(scenario["driver_drop_v"])
    magnitude = max(0.0, abs(duty) * supply - drop)
    return math.copysign(magnitude, duty)


def wheel_speed_to_duty(radps: float, config: dict, scenario: dict) -> float:
    """Open-loop inverse of the motor model -- the feedforward that replaces a PID.

    It uses the *nominal* motor constants, so a randomized episode is driven
    with the wrong map on purpose; the camera loop absorbs the error exactly
    as it must on the real robot.
    """
    motor = config["motor"]
    reference = float(motor["measured_at_volts"])
    radps_per_volt = float(motor["no_load_wheel_radps"]) / reference
    damping = float(motor["stall_torque_nm"]) / reference / radps_per_volt + float(motor["viscous_friction_nm_per_radps"])
    load = float(motor["coulomb_friction_nm"]) + float(motor["viscous_friction_nm_per_radps"]) * abs(radps)
    volts = (abs(radps) + load / damping) / radps_per_volt
    duty = (volts + float(motor["driver_drop_v"])) / float(motor["supply_voltage_v"])
    if abs(radps) <= 1.0e-9:
        duty = 0.0
    maximum = float(motor["max_duty"])
    return float(np.clip(math.copysign(duty, radps), -maximum, maximum))


def enforce_minimum_loaded_command(duty: np.ndarray, config: dict) -> np.ndarray:
    """Clear the dead zone while retaining left/right differential steering.

    Applied twice: the plant charges scrub against the duty it is given, while
    one pass can only read severity from the request, and lifting an asymmetric
    request makes it more of a pivot than it was.  The second pass sees the
    severity the plant will see.

    A per-wheel command below ``minimum_command_release_fraction`` of the loaded
    minimum is released to zero before either pass.  The lift had no lower bound
    on what it would honour, so a request of -0.02 came back as -0.90 and turned
    a near-zero steering error into a full-rate pivot.  The plant produces no
    motion anywhere below breakaway, so releasing is the honest reading; the
    only thing lost is the envelope inventing a large command on the policy's
    behalf.
    """
    motor = config["motor"]
    release_duty = float(motor["minimum_command_release_fraction"]) * float(
        motor["minimum_loaded_command_duty"])
    released = np.where(np.abs(np.asarray(duty, dtype=float)) >= release_duty, duty, 0.0)
    return _minimum_loaded_pass(_minimum_loaded_pass(released, config), config)


def _minimum_loaded_pass(duty: np.ndarray, config: dict) -> np.ndarray:
    """One lift using the severity of the duty as handed in."""
    duty = np.asarray(duty, dtype=float)
    if duty.shape[-1] != 2:
        raise ValueError("loaded-command compensation expects [..., 2] left/right duty")
    # A turning command must clear the scrub-loaded dead band, not the
    # straight-line one, or the envelope emits pivots the plant ignores.
    minimum = float(config["motor"]["minimum_loaded_command_duty"]) + float(
        config["motor"]["scrub_command_margin"]) * scrub_severity(duty)
    magnitude = np.abs(duty)
    individually_lifted = np.where(
        magnitude > 1.0e-9,
        np.copysign(np.maximum(magnitude, np.asarray(minimum)[..., None]), duty),
        0.0,
    )
    minimum_magnitude = magnitude.min(axis=-1)
    same_sign = (duty[..., 0] * duty[..., 1] > 0.0) & (minimum_magnitude > 1.0e-9)
    common_offset = np.maximum(0.0, minimum - minimum_magnitude)
    pair_lifted = duty + np.sign(duty) * common_offset[..., None]
    return np.where(same_sign[..., None], pair_lifted, individually_lifted)


def policy(observation: tuple[float, ...], config: dict, scenario: dict) -> tuple[float, float]:
    """Return left/right PWM duty in [-1, 1] -- the frozen firmware action.

    Steering is pure pursuit on the look-ahead point; the duty conversion is a
    feedforward motor inverse.  No PID, no inner loop: everything the ESP32
    needs is this function plus the motor constants.
    """
    e_y, e_theta, confidence = observation[0], observation[1], observation[2]
    controller, robot = config["controller"], config["robot"]
    yaw_rate = controller["steering_sign"] * float(np.clip(
        controller["kp_lateral"] * e_y + controller["kp_heading"] * e_theta,
        -controller["max_yaw_rate_radps"], controller["max_yaw_rate_radps"],
    ))
    radius = robot["wheel_radius_m"] * scenario["wheel_radius_scale"]
    track = robot["track_width_m"] * scenario["track_width_scale"]
    speed = controller["target_speed_mps"]
    if controller.get("algorithm") == "centerline_lookahead":
        speed = max(float(config["vision"]["minimum_speed_mps"]),
                    speed / (1.0 + float(config["vision"]["speed_slowdown_gain"]) * abs(e_theta)))
    # A stale look-ahead is worth less than a fresh one; easing off is the
    # firmware safety envelope, and it must be identical here and on the robot.
    speed *= max(float(controller["minimum_confidence_speed_scale"]), confidence)
    left = (speed - yaw_rate * track / 2.0) / radius
    right = (speed + yaw_rate * track / 2.0) / radius
    raw_duty = np.asarray([
        wheel_speed_to_duty(left, config, scenario),
        wheel_speed_to_duty(right, config, scenario),
    ])
    compensated = enforce_minimum_loaded_command(raw_duty, config)
    maximum = float(config["motor"]["max_duty"])
    return tuple(float(value) for value in np.clip(compensated, -maximum, maximum))


def make_visual_material(stage, path: str, color, roughness: float = 0.82):
    """Diffuse material.  ``color`` is a scalar grey or an (r, g, b) triple.

    Roughness is a scenario parameter because glossy black tape can throw a
    specular highlight that reads brighter than the floor around it -- a real
    failure mode that a matte-only sweep never produces.
    """
    from pxr import Gf, Sdf, UsdShade

    rgb = (color, color, color) if isinstance(color, (int, float)) else tuple(color)
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def add_box(stage, path: str, center, scale, material, yaw_deg: float = 0.0, collision: bool = False) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, yaw_deg))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def bind_physics_material(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")


def scatter_floor_dirt(stage, config: dict, scenario: dict, ground) -> None:
    """Scatter scuffs, dust and tile seams over the floor.

    These are what make a real floor fail a fixed threshold: they add dark runs
    away from the tape, so the per-row run tracer has to stay anchored instead
    of grabbing the widest blob in the strip.
    """
    count = int(scenario["floor_dirt_patches"])
    if count <= 0:
        return
    rng = np.random.default_rng((scenario["seed"] or 0) * 7919 + 13)
    half_x, half_y = (value / 2.0 for value in config["track"]["floor_size_m"])
    scale = float(scenario["floor_dirt_luma_scale"])
    shade = tuple(float(min(1.0, max(0.0, channel * scale))) for channel in ground)
    material = make_visual_material(stage, "/World/Looks/Dirt", shade, scenario["floor_roughness"])
    for index in range(count):
        size = float(rng.uniform(0.4, 1.0)) * float(scenario["floor_dirt_size_m"])
        add_box(stage, f"/World/Dirt/Patch_{index:03d}",
                (float(rng.uniform(-half_x, half_x)), float(rng.uniform(-half_y, half_y)), 0.0005),
                (size, size * float(rng.uniform(0.2, 1.0)), 0.001), material,
                float(rng.uniform(0.0, 180.0)))


def build_track(stage, config: dict, scenario: dict, contact_material) -> None:
    from pxr import Gf, UsdPhysics

    ground = floor_color(scenario)
    floor = make_visual_material(stage, "/World/Looks/Floor", ground, scenario["floor_roughness"])
    tape = make_visual_material(stage, "/World/Looks/Tape", scenario["tape_brightness"], scenario["tape_roughness"])
    floor_size = config["track"]["floor_size_m"]
    add_box(stage, "/World/Floor", (0.0, 0.0, -0.025), (floor_size[0], floor_size[1], 0.05), floor, collision=True)
    bind_physics_material(stage.GetPrimAtPath("/World/Floor"), contact_material)
    scatter_floor_dirt(stage, config, scenario, ground)
    points = [np.asarray(point, dtype=float) for point in config["track"]["points_xy_m"]]
    segments = list(zip(points, points[1:]))
    if config["track"].get("closed", False):
        segments.append((points[-1], points[0]))
    for index, (start, end) in enumerate(segments):
        delta = end - start
        center = (start + end) / 2.0
        add_box(stage, f"/World/Tape/Segment_{index:02d}", (center[0], center[1], 0.001),
                (float(np.linalg.norm(delta)), scenario["tape_width_m"], 0.002), tape,
                math.degrees(math.atan2(delta[1], delta[0])))
    # Open tracks keep tape visible for the final camera frame after the
    # measured finish point. Closed tracks have no artificial finish segment.
    final_delta = points[-1] - points[-2]
    final_direction = final_delta / np.linalg.norm(final_delta)
    extension = float(config["track"].get("finish_tape_extension_m", 0.0))
    if extension > 0.0 and not config["track"].get("closed", False):
        start, end = points[-1], points[-1] + final_direction * extension
        add_box(stage, "/World/Tape/FinishExtension", ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, 0.001),
                (extension, scenario["tape_width_m"], 0.002), tape,
                math.degrees(math.atan2(final_direction[1], final_direction[0])))


def create_contact_material(stage, scenario: dict):
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, "/World/Looks/WheelContact")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(scenario["static_friction"]))
    api.CreateDynamicFrictionAttr().Set(float(scenario["dynamic_friction"]))
    api.CreateRestitutionAttr().Set(0.0)
    return material


def look_at_matrix(eye, target, roll_rad: float = 0.0):
    """Return a camera transform looking at ``target`` with optical-axis roll.

    USD cameras look down local ``-Z``.  ``right`` and ``image_up`` are their
    local X/Y axes; rotating those axes around the fixed optical axis models a
    bracket that is installed with roll error without changing where it looks.
    """
    from pxr import Gf

    forward = (target - eye).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    if abs(forward * up) > 0.99:
        up = Gf.Vec3d(0, 1, 0)
    right = Gf.Vec3d.GetCross(forward, up).GetNormalized()
    image_up = Gf.Vec3d.GetCross(right, forward)
    if abs(roll_rad) > 1e-12:
        cosine, sine = math.cos(roll_rad), math.sin(roll_rad)
        right, image_up = right * cosine + image_up * sine, image_up * cosine - right * sine
    return Gf.Matrix4d(right[0], right[1], right[2], 0, *image_up, 0,
                       -forward[0], -forward[1], -forward[2], 0, eye[0], eye[1], eye[2], 1)


def rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def roll_pitch_deg(quaternion: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    return math.degrees(roll), math.degrees(pitch)


def track_progress(position_xy: np.ndarray, points: list[list[float]], closed: bool = False) -> tuple[float, float]:
    progress = 0.0
    best_distance, best_progress = float("inf"), 0.0
    segments = list(zip(points, points[1:]))
    if closed:
        segments.append((points[-1], points[0]))
    for start, end in segments:
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        delta = b - a
        length = float(np.linalg.norm(delta))
        fraction = float(np.clip(np.dot(position_xy - a, delta) / (length * length), 0.0, 1.0))
        distance = float(np.linalg.norm(position_xy - (a + fraction * delta)))
        if distance < best_distance:
            best_distance, best_progress = distance, progress + fraction * length
        progress += length
    return best_progress, progress


def nearest_track_segment(
    position_xy: np.ndarray, points: list[list[float]], closed: bool = False,
) -> tuple[int, float, float]:
    """Return nearest segment index, within-segment fraction, and distance."""
    segments = list(zip(points, points[1:]))
    if closed:
        segments.append((points[-1], points[0]))
    best_index, best_fraction, best_distance = -1, 0.0, float("inf")
    for index, (start, end) in enumerate(segments):
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        delta = b - a
        length_sq = float(np.dot(delta, delta))
        if length_sq <= 0.0:
            continue
        fraction = float(np.clip(np.dot(position_xy - a, delta) / length_sq, 0.0, 1.0))
        distance = float(np.linalg.norm(position_xy - (a + fraction * delta)))
        if distance < best_distance:
            best_index, best_fraction, best_distance = index, fraction, distance
    return best_index, best_fraction, best_distance


def build_track_label_cache(
    points: list[list[float]], closed: bool, finish_extension_m: float = 0.0,
) -> dict[str, np.ndarray | float | bool]:
    """Precompute the centerline polyline used for camera-supervision labels."""
    polyline = np.asarray(points, dtype=float)
    if closed:
        polyline = np.vstack([polyline, polyline[0]])
    deltas = np.diff(polyline, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    if not np.all(lengths > 1e-8):
        raise ValueError("track.points_xy_m must not contain duplicate adjacent points")
    return {
        "points": polyline,
        "lengths": lengths,
        "cumulative": np.r_[0.0, np.cumsum(lengths)],
        "total": float(lengths.sum()),
        "closed": closed,
        # Matches the run-out tape the scene lays past the last vertex.
        "finish_extension": 0.0 if closed else max(0.0, float(finish_extension_m)),
    }


def point_at_track_progress(cache: dict[str, np.ndarray | float | bool], progress_m: float) -> np.ndarray:
    """Interpolate a world XY point on the label polyline at path progress."""
    total = float(cache["total"])
    if bool(cache["closed"]):
        progress_m %= total
    else:
        progress_m = float(np.clip(
            progress_m, 0.0, total + float(cache.get("finish_extension", 0.0)),
        ))
    cumulative = np.asarray(cache["cumulative"], dtype=float)
    lengths = np.asarray(cache["lengths"], dtype=float)
    points = np.asarray(cache["points"], dtype=float)
    index = int(np.clip(np.searchsorted(cumulative, progress_m, side="right") - 1, 0, len(lengths) - 1))
    fraction = (progress_m - cumulative[index]) / lengths[index]
    return points[index] + fraction * (points[index + 1] - points[index])


def distort_projected_pixel(
    pixel_x: float, pixel_y: float, width: int, height: int, scenario: dict,
) -> tuple[float, float]:
    """Apply the same forward Brown-Conrady model used by the rendered sensor."""
    k1, k2 = float(scenario["lens_radial_k1"]), float(scenario["lens_radial_k2"])
    p1, p2 = float(scenario["lens_tangential_p1"]), float(scenario["lens_tangential_p2"])
    if max(abs(k1), abs(k2), abs(p1), abs(p2)) < 1e-9:
        return pixel_x, pixel_y
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    scale_x, scale_y = max(center_x, 1.0), max(center_y, 1.0)
    x, y = (pixel_x - center_x) / scale_x, (pixel_y - center_y) / scale_y
    radius_sq = x * x + y * y
    radial = 1.0 + k1 * radius_sq + k2 * radius_sq * radius_sq
    distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (radius_sq + 2.0 * x * x)
    distorted_y = y * radial + p1 * (radius_sq + 2.0 * y * y) + 2.0 * p2 * x * y
    return distorted_x * scale_x + center_x, distorted_y * scale_y + center_y


def image_observation_from_track_geometry(
    eye: np.ndarray,
    forward: np.ndarray,
    robot_xy: np.ndarray,
    config: dict,
    scenario: dict,
    label_cache: dict[str, np.ndarray | float | bool],
) -> tuple[float, float] | None:
    """Project the known tape centreline to label a rendered camera frame.

    This is independent of Otsu, the trace algorithm, and image brightness.  It
    gives supervised training the exact image-space ABI that the policy expects,
    including the configured lens distortion. The lens transform is evaluated
    only near the physical sensor frustum: Brown-Conrady polynomials are not
    meaningful for points thousands of pixels off-screen and can otherwise
    fold an invisible continuation of the tape back through an ROI.
    """
    width, height = (int(value) for value in config["robot"]["camera_resolution_px"])
    half_width = (width - 1) / 2.0
    near_row = float(config["vision"]["roi_rows_fraction"][0]) * (height - 1)
    far_row = float(config["vision"]["lookahead_row_fraction"]) * (height - 1)
    sensor_margin = float(config["vision"].get("geometry_label_sensor_margin_px", 16.0))
    forward = np.asarray(forward, dtype=float)
    forward /= max(1e-9, float(np.linalg.norm(forward)))
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-6:
        return None
    right /= float(np.linalg.norm(right))
    image_up = np.cross(right, forward)
    image_up /= max(1e-9, float(np.linalg.norm(image_up)))
    roll_rad = math.radians(float(scenario.get("camera_roll_deg", 0.0)))
    if abs(roll_rad) > 1e-12:
        cosine, sine = math.cos(roll_rad), math.sin(roll_rad)
        right, image_up = right * cosine + image_up * sine, image_up * cosine - right * sine
    current_progress, _ = track_progress(robot_xy, config["track"]["points_xy_m"], bool(label_cache["closed"]))
    total = float(label_cache["total"])
    start = current_progress - 0.06
    stop = current_progress + 0.65
    if not bool(label_cache["closed"]):
        start = max(0.0, start)
        stop = min(total + float(label_cache.get("finish_extension", 0.0)), stop)
    samples = np.linspace(start, stop, max(2, int(math.ceil((stop - start) / 0.004)) + 1))
    focal_px = half_width / math.tan(math.radians(float(scenario["camera_hfov_deg"])) / 2.0)
    projected: list[tuple[float, float, float] | None] = []
    for sample_progress in samples:
        point_xy = point_at_track_progress(label_cache, float(sample_progress))
        offset = np.array([point_xy[0], point_xy[1], 0.002], dtype=float) - eye
        depth = float(np.dot(offset, forward))
        if depth <= 1e-5:
            projected.append(None)
            continue
        pixel_x = half_width + focal_px * float(np.dot(offset, right)) / depth
        pixel_y = (height - 1) / 2.0 - focal_px * float(np.dot(offset, image_up)) / depth
        if (
            pixel_x < -sensor_margin or pixel_x > (width - 1) + sensor_margin
            or pixel_y < -sensor_margin or pixel_y > (height - 1) + sensor_margin
        ):
            projected.append(None)
            continue
        pixel_x, pixel_y = distort_projected_pixel(pixel_x, pixel_y, width, height, scenario)
        projected.append((float(sample_progress), pixel_x, pixel_y))

    def crossing_x(row: float) -> float | None:
        candidates: list[tuple[float, float]] = []
        for point_a, point_b in zip(projected, projected[1:]):
            if point_a is None or point_b is None:
                continue
            progress_a, x_a, y_a = point_a
            progress_b, x_b, y_b = point_b
            if abs(y_b - y_a) < 1e-6 or row < min(y_a, y_b) or row > max(y_a, y_b):
                continue
            fraction = (row - y_a) / (y_b - y_a)
            candidates.append((progress_a + fraction * (progress_b - progress_a), x_a + fraction * (x_b - x_a)))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    trace_rows: list[float] = []
    trace_centers: list[float] = []
    for row in sorted(
        (float(value) * (height - 1) for value in config["vision"]["centerline_rows_fraction"]),
        reverse=True,
    ):
        center = crossing_x(row)
        if center is None:
            if trace_rows:
                break
            continue
        trace_rows.append(row)
        trace_centers.append(center)
    if len(trace_rows) < int(config["vision"]["minimum_centerline_points"]):
        return None
    coefficients = np.polyfit(trace_rows, trace_centers, 2 if len(trace_rows) >= 4 else 1)
    nearest, farthest = max(trace_rows), min(trace_rows)
    near_row = float(np.clip(near_row, farthest, nearest))
    far_row = float(np.clip(far_row, farthest, nearest))
    near_x = float(np.polyval(coefficients, near_row))
    far_x = float(np.polyval(coefficients, far_row))
    lateral = float(np.clip((far_x - half_width) / half_width, -1.0, 1.0))
    heading = math.atan2(far_x - near_x, max(1.0, near_row - far_row))
    return lateral, float(heading)


def save_rgb_image(path: Path, frame: np.ndarray) -> None:
    """Save an RGB sensor frame without altering its perception input."""
    from PIL import Image

    image = frame[..., :3].astype(np.float32)
    if image.max(initial=0.0) <= 1.5:
        image *= 255.0
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def save_mask_image(path: Path, mask: np.ndarray) -> None:
    """Save a binary perception mask for an opt-in debugging trace."""
    from PIL import Image

    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def rgb_u8(frame: np.ndarray) -> np.ndarray:
    """Copy an Isaac RGB/RGBA frame as three uint8 camera channels."""
    image = np.asarray(frame[..., :3], dtype=np.float32)
    if image.max(initial=0.0) <= 1.5:
        image *= 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def load_rl_policy(
    checkpoint_path: Path,
    config_path: Path,
) -> Callable[[tuple[float, ...]], tuple[float, float]]:
    """Load the checkpoint actor only and reproduce training's tanh action map.

    The validator must execute the exact actor that will be exported, but it
    must not create a second Isaac Lab environment merely to deserialize it.
    RSL-RL checkpoints store the actor under ``actor.*``; reject any shape or
    key mismatch instead of silently evaluating a different network.
    """
    if not checkpoint_path.is_file():
        raise ValueError(f"RL checkpoint does not exist: {checkpoint_path}")
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("RL checkpoint does not contain an RSL-RL model_state_dict.")
    validate_checkpoint_config(checkpoint, config_path)
    actor = torch.nn.Sequential(
        torch.nn.Linear(7, 64), torch.nn.ELU(),
        torch.nn.Linear(64, 64), torch.nn.ELU(),
        torch.nn.Linear(64, 2),
    )
    actor_state = {
        name.removeprefix("actor."): value
        for name, value in checkpoint["model_state_dict"].items()
        if name.startswith("actor.")
    }
    expected = set(actor.state_dict())
    if set(actor_state) != expected:
        raise ValueError(
            "RL checkpoint actor is not the frozen 7->64(ELU)->64(ELU)->2 layout: "
            f"expected {sorted(expected)}, got {sorted(actor_state)}."
        )
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()

    def infer(observation: tuple[float, ...]) -> tuple[float, float]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"RL inference requires seven finite ABI values, got {values!r}.")
        with torch.inference_mode():
            raw = actor(torch.from_numpy(values).unsqueeze(0)).numpy()[0]
        duty = np.tanh(np.asarray(raw, dtype=np.float32))
        if duty.shape != (2,) or not np.isfinite(duty).all():
            raise RuntimeError("RL actor produced a non-finite or wrong-shaped duty action.")
        return float(duty[0]), float(duty[1])

    return infer


def camera_gain(capture_time_s: float, scenario: dict) -> float:
    """Deterministic exposure/auto-gain multiplier for one sensor frame."""
    exposure_ev = float(scenario.get("exposure_ev_bias", 0.0))
    exposure_ev += float(scenario.get("auto_gain_drift_ev", 0.0)) * math.sin(
        2.0 * math.pi * float(scenario.get("auto_gain_drift_hz", 0.0)) * capture_time_s
        + float(scenario.get("auto_gain_phase_rad", 0.0)),
    )
    return float(2.0 ** exposure_ev)


def build_lens_remap(
    height: int, width: int, scenario: dict,
) -> dict[str, np.ndarray] | None:
    """Build one inverse Brown-Conrady lens map for a fixed camera resolution.

    Coefficients use the conventional normalized-image Brown-Conrady model.
    Iterating its inverse a few times avoids a dependency on OpenCV and keeps
    the exact same image transform available to an ESP-side test tool.
    """
    k1 = float(scenario.get("lens_radial_k1", 0.0))
    k2 = float(scenario.get("lens_radial_k2", 0.0))
    p1 = float(scenario.get("lens_tangential_p1", 0.0))
    p2 = float(scenario.get("lens_tangential_p2", 0.0))
    if max(abs(k1), abs(k2), abs(p1), abs(p2)) < 1e-9:
        return None
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    scale_x = max(center_x, 1.0)
    scale_y = max(center_y, 1.0)
    target_x, target_y = np.meshgrid(
        (np.arange(width, dtype=np.float32) - center_x) / scale_x,
        (np.arange(height, dtype=np.float32) - center_y) / scale_y,
    )
    source_x, source_y = target_x.copy(), target_y.copy()
    # Solve distorted(source) = target.  The configured range is deliberately
    # modest, so fixed-point iteration is stable and only runs once/episode.
    for _ in range(4):
        radius_sq = source_x * source_x + source_y * source_y
        radial = 1.0 + k1 * radius_sq + k2 * radius_sq * radius_sq
        distorted_x = source_x * radial + 2.0 * p1 * source_x * source_y + p2 * (
            radius_sq + 2.0 * source_x * source_x
        )
        distorted_y = source_y * radial + p1 * (radius_sq + 2.0 * source_y * source_y) + 2.0 * p2 * source_x * source_y
        source_x += target_x - distorted_x
        source_y += target_y - distorted_y
    source_x = np.clip(source_x * scale_x + center_x, 0.0, width - 1.0)
    source_y = np.clip(source_y * scale_y + center_y, 0.0, height - 1.0)
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    return {
        "x0": x0,
        "x1": np.minimum(x0 + 1, width - 1),
        "y0": y0,
        "y1": np.minimum(y0 + 1, height - 1),
        "x_weight": (source_x - x0).astype(np.float32),
        "y_weight": (source_y - y0).astype(np.float32),
    }


def remap_bilinear(image: np.ndarray, remap: dict[str, np.ndarray]) -> np.ndarray:
    """Apply a precomputed lens map with bilinear sampling."""
    x0, x1 = remap["x0"], remap["x1"]
    y0, y1 = remap["y0"], remap["y1"]
    x_weight = remap["x_weight"][..., None]
    y_weight = remap["y_weight"][..., None]
    top = image[y0, x0] * (1.0 - x_weight) + image[y0, x1] * x_weight
    bottom = image[y1, x0] * (1.0 - x_weight) + image[y1, x1] * x_weight
    return top * (1.0 - y_weight) + bottom * y_weight


def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip through the same 4:2:0 JPEG class produced by an OV2640."""
    from PIL import Image

    quality = int(np.clip(quality, 1, 100))
    if quality >= 100:
        return image
    stream = io.BytesIO()
    Image.fromarray(np.clip(image, 0.0, 255.0).astype(np.uint8)).save(
        stream, format="JPEG", quality=quality, subsampling=2,
    )
    stream.seek(0)
    with Image.open(stream) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float32)


def apply_camera_sensor_artifacts(
    current_frame: np.ndarray,
    previous_frame: np.ndarray | None,
    capture_time_s: float,
    camera_period_s: float,
    scenario: dict,
    lens_remap: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """Apply deterministic camera-domain randomization to a rendered image.

    Lens distortion is spatial.  Motion blur is temporal integration of the
    preceding sensor frame; rolling shutter then gives upper rows an older
    exposure than lower rows.  With only camera-rate renders this is a
    deliberately conservative approximation of an OV2640 readout, but it
    creates the relevant line displacement without stealing the PhysX clock.
    """
    current = rgb_u8(current_frame).astype(np.float32) * camera_gain(capture_time_s, scenario)
    image = current
    if previous_frame is not None:
        previous = rgb_u8(previous_frame).astype(np.float32) * camera_gain(
            capture_time_s - camera_period_s, scenario,
        )
        motion_mix = float(np.clip(scenario.get("motion_blur_mix", 0.0), 0.0, 1.0))
        image = current * (1.0 - motion_mix) + previous * motion_mix
        readout_fraction = float(np.clip(scenario.get("rolling_shutter_fraction", 0.0), 0.0, 1.0))
        if readout_fraction > 0.0:
            # The top line was read first, so it is closer to the old sample.
            row_old_weight = readout_fraction * np.linspace(1.0, 0.0, image.shape[0], dtype=np.float32)
            image = image * (1.0 - row_old_weight[:, None, None]) + previous * row_old_weight[:, None, None]
    if lens_remap is not None:
        image = remap_bilinear(image, lens_remap)
    return jpeg_compress(np.clip(image, 0.0, 255.0), int(scenario.get("jpeg_quality", 100))).astype(np.uint8)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def run_episode(config: dict, args, scenario: dict) -> dict:
    """Run one PhysX episode and return a serializable summary."""
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 720, "renderer": "RayTracedLighting"})
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from isaacsim.core.api.simulation_context import SimulationContext
    from isaacsim.core.experimental.prims import Articulation
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

    # ``evaluate_line_following.py`` can outlive an edit to this script while it
    # forks one episode at a time.  Keep old evaluator namespaces analytical so
    # an in-flight baseline never turns into a spurious runner failure.
    policy_backend = getattr(args, "policy_backend", "analytic")
    checkpoint_path = getattr(args, "checkpoint", None)
    learned_policy: Callable[[tuple[float, ...]], tuple[float, float]] | None = None
    policy_id: str | None = None
    if policy_backend == "rl":
        if checkpoint_path is None:
            raise ValueError("RL policy backend requires a checkpoint path.")
        learned_policy = load_rl_policy(checkpoint_path, Path(args.config))
    elif policy_backend in {"reference", "deployed"}:
        artifact_name = "reference" if policy_backend == "reference" else "generated"
        artifact_dir = project_root() / "firmware" / artifact_name
        learned_policy, policy_id = load_header_policy(
            artifact_dir / "line_following_policy.h",
            artifact_dir / "line_following_policy_manifest.json",
            Path(args.config),
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = float(config["physics"]["dt_s"])
    # Two independent rates.  The camera is the slow one; the actuation loop
    # runs as fast as the firmware will, so encoder and duty feedback are fresh
    # even while the look-ahead is held between frames.
    physics_hz = 1.0 / dt
    policy_hz = float(config["physics"]["policy_hz"])
    camera_hz = float(config["physics"]["camera_hz"])
    policy_stride = integer_rate_stride(physics_hz, policy_hz, label="physics-to-policy")
    camera_stride = integer_rate_stride(physics_hz, camera_hz, label="physics-to-camera")
    integer_rate_stride(policy_hz, camera_hz, label="policy-to-camera")
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr().Set(round(1.0 / dt))
    physx_scene.CreateSolverTypeAttr().Set(config["physics"]["solver"])
    physx_scene.CreateEnableCCDAttr().Set(bool(config["physics"]["enable_ccd"]))
    physx_scene.CreateEnableStabilizationAttr().Set(bool(config["physics"]["enable_stabilization"]))
    physx_scene.CreateEnableGPUDynamicsAttr().Set(bool(config["physics"]["enable_gpu_dynamics"]))
    physx_scene.CreateEnableEnhancedDeterminismAttr().Set(bool(config["physics"]["enhanced_determinism"]))
    contact_material = create_contact_material(stage, scenario)
    build_track(stage, config, scenario, contact_material)

    robot_prim = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot_prim.GetPrim().GetReferences().AddReference(str(config["robot"]["usd_asset"]))
    # Apply the same base-mass and centre-of-mass draw used by the training lane;
    # sampling a value without changing the rendered plant would make the two
    # evaluation paths describe different distributions.
    base_prim = stage.GetPrimAtPath("/World/Robot/base_link")
    if not base_prim.IsValid() or not base_prim.HasAPI(UsdPhysics.MassAPI):
        raise RuntimeError("Robot base_link must carry UsdPhysics.MassAPI for mass randomization.")
    mass_api = UsdPhysics.MassAPI.Get(stage, base_prim.GetPath())
    nominal_mass = mass_api.GetMassAttr().Get()
    nominal_com = mass_api.GetCenterOfMassAttr().Get()
    if nominal_mass is None or nominal_com is None:
        raise RuntimeError("Robot base_link must author both mass and centerOfMass.")
    mass_api.GetMassAttr().Set(float(nominal_mass) * float(scenario["base_mass_scale"]))
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(
        float(nominal_com[0]) + float(scenario["base_com_offset_m"]),
        float(nominal_com[1]) + float(scenario["base_com_offset_y_m"]),
        float(nominal_com[2]),
    ))
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/Robot") and prim.HasAPI(UsdPhysics.CollisionAPI):
            bind_physics_material(prim, contact_material)
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(scenario["dome_intensity"])
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr().Set(scenario["key_intensity"])
    key.AddRotateXYZOp().Set(Gf.Vec3f(35.0, 0.0, 25.0))
    overview = UsdGeom.Camera.Define(stage, "/World/TeachingOverviewCamera")
    overview.CreateFocalLengthAttr().Set(18.0)
    overview_xform = UsdGeom.Xformable(overview.GetPrim()).AddTransformOp()
    overview_xform.Set(look_at_matrix(Gf.Vec3d(1.2, -1.65, 1.15), Gf.Vec3d(0.0, 0.0, 0.0)))

    robot = Articulation("/World/Robot")
    sim = SimulationContext(physics_dt=dt, rendering_dt=dt)
    sim.play()
    for _ in range(8):
        sim.step(render=True)
    wheel_indices = robot.get_dof_indices(config["robot"]["wheel_joints"]).numpy().tolist()
    if len(wheel_indices) != 4:
        raise RuntimeError(f"Expected four wheel DOFs, got {robot.dof_names}")
    wheel_signs = np.asarray(config["robot"]["wheel_joint_velocity_signs"], dtype=float)
    if wheel_signs.shape != (4,):
        raise RuntimeError("robot.wheel_joint_velocity_signs must contain four values")
    wheel_slots = {name: index for index, name in enumerate(config["robot"]["wheel_joints"])}
    try:
        encoder_slots = [wheel_slots[name] for name in config["robot"]["encoder_joints"]]
        rear_slots = [wheel_slots[name] for name in config["robot"]["rear_diagnostic_joints"]]
    except KeyError as error:
        raise RuntimeError(f"Encoder/diagnostic joint is not a wheel joint: {error}") from error
    if len(encoder_slots) != 2 or len(rear_slots) != 2:
        raise RuntimeError("Configure exactly left/right front encoders and rear diagnostic joints")
    start_x, start_y = config["track"]["start_xy_m"]
    start_yaw = math.radians(float(config["track"].get("start_heading_deg", 0.0)) + scenario["start_heading_deg"])
    robot.set_world_poses(
        positions=[[start_x, start_y + scenario["start_lateral_m"], config["robot"]["base_height_m"]]],
        orientations=[[math.cos(start_yaw / 2.0), 0.0, 0.0, math.sin(start_yaw / 2.0)]],
    )
    robot.set_dof_velocities([[0.0, 0.0, 0.0, 0.0]], dof_indices=wheel_indices)

    camera = UsdGeom.Camera.Define(stage, "/World/RobotCamera")
    camera.CreateHorizontalApertureAttr().Set(20.955)
    camera.CreateVerticalApertureAttr().Set(15.716)
    hfov = math.radians(scenario["camera_hfov_deg"])
    camera.CreateFocalLengthAttr().Set(20.955 / (2.0 * math.tan(hfov / 2.0)))
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 5.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim()).AddTransformOp()
    if args.gui:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = str(camera.GetPath())
    width, height = config["robot"]["camera_resolution_px"]
    render_product = rep.create.render_product(camera.GetPath(), (width, height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])
    camera_period_s = camera_stride * dt
    lens_remap = build_lens_remap(height, width, scenario)

    # Overwrite the drive baked at import time with this episode's motor line.
    # Doing it here (not in the USD) is what lets the electrical parameters be
    # domain-randomized without a re-import.
    radps_per_volt, _torque_per_volt, motor_damping = motor_curve(scenario, config["motor"])
    for name in config["robot"]["wheel_joints"]:
        joint = stage.GetPrimAtPath(f"/World/Robot/joints/{name}")
        if not joint.IsValid():
            raise RuntimeError(f"Wheel joint prim missing: {name}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        drive.GetDampingAttr().Set(float(motor_damping))
        drive.GetStiffnessAttr().Set(0.0)
        drive.GetMaxForceAttr().Set(float(scenario["stall_torque_nm"]))
        # The URDF ships SolidWorks' default 0.1 N.m of joint friction, which is
        # several times a small gearmotor's whole stall torque.  Left alone the
        # wheels cannot turn once the drive is a real motor instead of a servo.
        PhysxSchema.PhysxJointAPI.Apply(joint).CreateJointFrictionAttr().Set(
            float(scenario["coulomb_friction_nm"]))

    rng = np.random.default_rng(scenario["seed"])
    command_duty = np.zeros(2, dtype=float)
    duty_prev = np.zeros(2, dtype=float)
    applied_duty = np.zeros(2, dtype=float)
    blind_duty_scale = float(config.get("rl", {}).get("blind_duty_scale", 0.0))
    if not 0.0 <= blind_duty_scale <= 1.0:
        raise ValueError("rl.blind_duty_scale must be in [0, 1].")
    # Left/right motors are never identical; the mismatch is a steering bias the
    # camera loop has to absorb, which is exactly the real-robot behaviour.
    imbalance = float(scenario["motor_imbalance"])
    motor_side_gain = np.array([imbalance, imbalance, 1.0 / imbalance, 1.0 / imbalance])
    actuation_delay_ticks = int(scenario["actuation_delay_ticks"])
    encoder_delay_ticks = int(config["randomization"]["encoder_delay_ticks"])
    if actuation_delay_ticks < 0 or encoder_delay_ticks < 0:
        raise ValueError("actuation_delay_ticks and encoder_delay_ticks must be non-negative.")
    delayed_duty = [np.zeros(2, dtype=float) for _ in range(actuation_delay_ticks)]
    delayed_rpm = [np.zeros(2, dtype=float) for _ in range(encoder_delay_ticks)]
    logs: list[list[float]] = []
    perception_logs: list[list[float | int | str | None]] = []
    # Saving PNGs synchronously can stall the render thread and turn an opt-in
    # diagnostic flag into a different simulation.  Keep the small set of
    # transition frames in memory and write them only after physics stops.
    deferred_debug_images: list[tuple[Path, np.ndarray, bool]] = []
    deferred_dataset_images: list[tuple[Path, np.ndarray]] = []
    dataset_records: list[dict[str, float | int | str]] = []

    def defer_debug_image(path: Path, image: np.ndarray, is_mask: bool = False) -> None:
        deferred_debug_images.append((path, image.copy(), is_mask))

    previous_camera_detection: bool | None = None
    loss_event_index = 0
    last_valid_frame: np.ndarray | None = None
    last_valid_mask: np.ndarray | None = None
    pending_camera_frame: np.ndarray | None = None
    pending_camera_context: dict[str, np.ndarray | float] | None = None
    previous_camera_context: dict[str, np.ndarray | float] | None = None
    previous_raw_camera_frame: np.ndarray | None = None
    perception_debug_dir = output_dir / "perception_debug"
    if args.save_perception_debug:
        perception_debug_dir.mkdir(parents=True, exist_ok=True)
    line_loss_s = 0.0
    max_line_loss_s = 0.0
    max_tilt = 0.0
    debug_written = False
    last_valid_observation: tuple[float, float] | None = None
    completion_time: float | None = None
    points = config["track"]["points_xy_m"]
    closed_track = bool(config["track"].get("closed", False))
    label_cache = build_track_label_cache(
        points, closed_track, float(config["track"].get("finish_tape_extension_m", 0.0)),
    )
    dataset_root = args.dataset_dir
    dataset_prefix = Path(args.dataset_prefix or f"seed_{scenario['seed']:04d}").name
    dataset_camera_frames = 0
    _progress, track_length = track_progress(
        np.asarray(config["track"]["start_xy_m"], dtype=float),
        points,
        closed_track,
    )
    configured_duration = args.duration_s if args.duration_s is not None else config["physics"].get("duration_s")
    if configured_duration is None:
        nominal_time = track_length / float(config["controller"]["target_speed_mps"])
        duration = min(
            float(config["physics"]["maximum_duration_s"]),
            max(float(config["physics"]["minimum_duration_s"]), nominal_time * float(config["physics"]["timeout_multiplier"])),
        )
    else:
        duration = float(configured_duration)
    steps = round(duration / dt)
    failure_reason = "safety_timeout"
    travelled_distance = 0.0
    previous_xy = np.asarray([start_x, start_y + scenario["start_lateral_m"]], dtype=float)
    for step in range(steps):
        position, orientation = robot.get_world_poses()
        pos = position.numpy()[0]
        quat = orientation.numpy()[0]
        rotation = rotation_matrix_wxyz(quat)
        mount = np.asarray(config["robot"]["camera_mount_xyz_m"], dtype=float) + np.array([
            scenario["camera_mount_dx_m"], scenario["camera_mount_dy_m"], scenario["camera_mount_dz_m"],
        ])
        eye_np = pos + rotation @ mount
        # Tilt, yaw and roll are all bracket geometry.  Yaw points the optical
        # axis off the robot's centreline and roll rotates the image; a printed
        # bracket delivers all three with a couple of degrees of slop.
        down = math.radians(scenario["camera_down_angle_deg"])
        yaw = math.radians(scenario["camera_yaw_deg"])
        optical = np.array([
            math.cos(down) * math.cos(yaw), math.cos(down) * math.sin(yaw), -math.sin(down),
        ])
        direction = rotation @ optical
        eye = Gf.Vec3d(*eye_np)
        camera_xform.Set(look_at_matrix(
            eye,
            eye + Gf.Vec3d(*(0.25 * direction)),
            math.radians(scenario["camera_roll_deg"]),
        ))
        if step % camera_stride == 0:
            # Render without a physics step, then consume the preceding frame.
            # Keeping the single camera-period queue models sensor latency, but
            # it also makes the image/physics pairing independent of how long a
            # CSV or debug-image write takes.  rep.orchestrator.step() must NOT
            # be used here: it takes the timeline over, so PhysX then advances
            # only on rendered ticks and the robot travels at 1/camera_stride
            # of the commanded speed.
            sim.render()
            rendered_frame = rgb.get_data()
            raw_frame = pending_camera_frame
            raw_camera_context = pending_camera_context
            pending_camera_frame = (
                rendered_frame.copy()
                if isinstance(rendered_frame, np.ndarray) and rendered_frame.ndim == 3
                else None
            )
            # The annotator pipeline is one frame behind the render call, so the
            # array read back here holds the scene as it stood at the *previous*
            # camera tick. Pairing it with the pose sampled now would label the
            # frame using a pose it never saw. Carry the prior context forward
            # so rendered perception and its geometric reference share a time.
            pending_camera_context = previous_camera_context
            previous_camera_context = {
                "eye": eye_np.copy(), "forward": direction.copy(), "robot_xy": pos[:2].copy(),
                "capture_time_s": step * dt,
            }
            capture_time_s = max(0.0, (step - camera_stride) * dt)
            frame: np.ndarray | None = None
            if raw_frame is None:
                observation = None
                perception: dict[str, float | int | str | None] = {
                    "state": "invalid_frame", "threshold_px": None,
                    "threshold_guard": "queue_not_ready", "dark_fraction": 0.0, "trace_points": 0,
                    "backend": "classical", "raw_e_y": None, "raw_e_theta_rad": None,
                    "raw_confidence_logit": None,
                    "confidence_probability": None, "inference_ms": None,
                }
                mask = None
            else:
                frame = apply_camera_sensor_artifacts(
                    raw_frame, previous_raw_camera_frame, capture_time_s, camera_period_s, scenario, lens_remap,
                )
                previous_raw_camera_frame = raw_frame.copy()
                observation, perception, mask = image_observation_with_diagnostics(frame, config)
                perception["backend"] = "classical"
                # Store the camera's own output beside the geometric label for
                # this frame. The policy observation is delayed, so joining
                # through observations.csv would mix sensor error with pipeline
                # latency.
                perception["raw_e_y"] = None if observation is None else float(observation[0])
                perception["raw_e_theta_rad"] = None if observation is None else float(observation[1])
                perception["raw_confidence_logit"] = None
                perception["confidence_probability"] = None
                perception["inference_ms"] = None
            if args.save_debug_frame and not debug_written and frame is not None and frame.size:
                defer_debug_image(output_dir / "camera_debug.png", frame)
                debug_written = True
            camera_progress, _camera_total = track_progress(pos[:2], points, closed_track)
            segment_index, segment_fraction, segment_distance = nearest_track_segment(pos[:2], points, closed_track)
            # Log the geometric training observation and rendered-camera result
            # together. Parity checks implementation consistency; this paired
            # record exposes modeling error between geometry and pixels.
            geometry_e_y = ""
            geometry_e_theta = ""
            geometry_valid = 0
            if raw_camera_context is not None:
                geometry = image_observation_from_track_geometry(
                    np.asarray(raw_camera_context["eye"], dtype=float),
                    np.asarray(raw_camera_context["forward"], dtype=float),
                    np.asarray(raw_camera_context["robot_xy"], dtype=float),
                    config, scenario, label_cache,
                )
                if geometry is not None:
                    geometry_e_y, geometry_e_theta = geometry
                    geometry_valid = 1
            perception_logs.append([
                step * dt, max(0.0, (step - camera_stride) * dt), perception["state"],
                perception["threshold_px"], perception["threshold_guard"], perception["dark_fraction"],
                perception["trace_points"],
                camera_progress, segment_index, segment_fraction, segment_distance,
                float(pos[0]), float(pos[1]), float(pos[2]), perception["backend"],
                perception["raw_e_y"], perception["raw_e_theta_rad"], perception["raw_confidence_logit"],
                perception["confidence_probability"], perception["inference_ms"],
                geometry_valid, geometry_e_y, geometry_e_theta,
                # How old the frame this row describes is, in seconds.  The
                # rendered lane holds one camera period of sensor latency; the
                # training lane has none.  Logging the age makes that difference
                # a measured number rather than something inferred from a join.
                "" if raw_camera_context is None
                else round(step * dt - float(raw_camera_context["capture_time_s"]), 6),
            ])
            camera_detected = observation is not None
            if dataset_root is not None and frame is not None and raw_camera_context is not None:
                if dataset_camera_frames % args.dataset_stride == 0 and (
                    args.dataset_max_frames == 0 or len(dataset_records) < args.dataset_max_frames
                ):
                    geometry_observation = image_observation_from_track_geometry(
                        np.asarray(raw_camera_context["eye"], dtype=float),
                        np.asarray(raw_camera_context["forward"], dtype=float),
                        np.asarray(raw_camera_context["robot_xy"], dtype=float),
                        config, scenario, label_cache,
                    )
                    sample_index = len(dataset_records)
                    relative_filename = f"frames/{dataset_prefix}_{sample_index:06d}.png"
                    deferred_dataset_images.append((dataset_root / relative_filename, frame.copy()))
                    record: dict[str, float | int | str] = {
                        "image": relative_filename,
                        "capture_time_s": float(raw_camera_context["capture_time_s"]),
                        "track_progress_m": track_progress(
                            np.asarray(raw_camera_context["robot_xy"], dtype=float), points, closed_track,
                        )[0],
                        "label_valid": int(geometry_observation is not None),
                        "e_y": float(geometry_observation[0]) if geometry_observation is not None else 0.0,
                        "e_theta_rad": float(geometry_observation[1]) if geometry_observation is not None else 0.0,
                        "classical_valid": int(observation is not None),
                        "classical_e_y": float(observation[0]) if observation is not None else 0.0,
                        "classical_e_theta_rad": float(observation[1]) if observation is not None else 0.0,
                        "perception_state": str(perception["state"]),
                        "scenario_seed": int(scenario["seed"] or 0),
                        "dataset_split": args.dataset_split or "unspecified",
                        "episode_id": args.dataset_episode_id or dataset_prefix,
                    }
                    dataset_records.append(record)
                dataset_camera_frames += 1
            if args.save_perception_debug:
                if not camera_detected and previous_camera_detection is not False:
                    loss_event_index += 1
                    prefix = perception_debug_dir / f"loss_{loss_event_index:03d}"
                    if last_valid_frame is not None:
                        defer_debug_image(prefix.with_name(f"{prefix.name}_last_valid_rgb.png"), last_valid_frame)
                    if last_valid_mask is not None:
                        defer_debug_image(prefix.with_name(f"{prefix.name}_last_valid_mask.png"), last_valid_mask, is_mask=True)
                    if isinstance(frame, np.ndarray) and frame.ndim == 3:
                        defer_debug_image(prefix.with_name(f"{prefix.name}_start_rgb.png"), frame)
                    if mask is not None:
                        defer_debug_image(prefix.with_name(f"{prefix.name}_start_mask.png"), mask, is_mask=True)
                elif camera_detected and previous_camera_detection is False:
                    prefix = perception_debug_dir / f"loss_{loss_event_index:03d}"
                    if isinstance(frame, np.ndarray) and frame.ndim == 3:
                        defer_debug_image(prefix.with_name(f"{prefix.name}_recovered_rgb.png"), frame)
                    if mask is not None:
                        defer_debug_image(prefix.with_name(f"{prefix.name}_recovered_mask.png"), mask, is_mask=True)
                if camera_detected and isinstance(frame, np.ndarray) and frame.ndim == 3:
                    last_valid_frame = frame.copy()
                    last_valid_mask = None if mask is None else mask.copy()
            previous_camera_detection = camera_detected
            if observation is None:
                # Staleness is measured on the camera clock, which is the only
                # clock that can refresh it.  Deriving it from the confidence it
                # feeds would make the two lock each other at zero.
                #
                # The clock only starts once the tape has been seen at least
                # once.  Firmware separates "never locked on" (wait, forever if
                # need be) from "locked on and lost it" (safe stop after
                # max_line_loss_s), and only the second is a failure.  Counting
                # pre-lock wait as line loss would disagree with firmware,
                # which waits for a valid line before arming. The episode timer
                # still makes waiting costly.
                if last_valid_observation is not None:
                    line_loss_s += camera_stride * dt
            else:
                last_valid_observation = observation
                line_loss_s = 0.0
            max_line_loss_s = max(max_line_loss_s, line_loss_s)
        if step % policy_stride == 0:
            roll_deg, pitch_deg = roll_pitch_deg(quat)
            wheel_radps = robot.get_dof_velocities(dof_indices=wheel_indices).numpy()[0] * wheel_signs
            # Firmware has one encoder on each *front* wheel.  All four motors
            # are commanded by side, but only these two measurements cross the
            # simulation/ESP32 policy boundary.
            actual_rpm = wheel_radps[encoder_slots] * 60.0 / (2.0 * math.pi)
            rear_rpm = wheel_radps[rear_slots] * 60.0 / (2.0 * math.pi)
            measured = actual_rpm * scenario["encoder_scale"] + rng.normal(0.0, scenario["encoder_noise_rpm"], 2)
            delayed_rpm.append(measured)
            measured = delayed_rpm.pop(0)
            # Confidence is the observation that tells the policy how old the
            # look-ahead is.  Without it a held sample is indistinguishable
            # from a fresh one and the controller drives on stale data.
            confidence = 0.0 if last_valid_observation is None else float(
                np.clip(1.0 - line_loss_s / max(1e-6, float(config["evaluation"]["max_line_loss_s"])), 0.0, 1.0))
            if last_valid_observation is None:
                if policy_backend == "analytic":
                    # The analytical baseline keeps its original safe stop
                    # before first lock.  The RL lane is trained differently:
                    # it sees zero held error at confidence=0 and is capped.
                    e_y, e_theta = float("nan"), float("nan")
                    command_duty[:] = 0.0
                else:
                    e_y, e_theta = 0.0, 0.0
                    assert learned_policy is not None
                    command_duty[:] = learned_policy((e_y, e_theta, confidence, measured[0], measured[1],
                                                       duty_prev[0], duty_prev[1]))
            else:
                e_y, e_theta = last_valid_observation
                observation = (e_y, e_theta, confidence, measured[0], measured[1], duty_prev[0], duty_prev[1])
                if policy_backend == "analytic":
                    command_duty[:] = policy(observation, config, scenario)
                else:
                    assert learned_policy is not None
                    command_duty[:] = learned_policy(observation)
            if policy_backend != "analytic" and confidence <= 0.0:
                blind_limit = float(config["motor"]["max_duty"]) * blind_duty_scale
                command_duty[:] = np.clip(command_duty, -blind_limit, blind_limit)
            command_duty[:] = enforce_minimum_loaded_command(command_duty, config)
            command_duty[:] = np.clip(
                command_duty, -float(config["motor"]["max_duty"]), float(config["motor"]["max_duty"]),
            )
            # Firmware slew limit: the same envelope must exist in sim, or a
            # learned policy will rely on steps the hardware cannot deliver.
            slew = float(config["motor"]["duty_rate_limit_per_s"]) * policy_stride * dt
            command_duty[:] = np.clip(command_duty, duty_prev - slew, duty_prev + slew)
            duty_prev[:] = command_duty
            delayed_duty.append(command_duty.copy())
            applied_duty[:] = delayed_duty.pop(0)
            logs.append([step * dt, e_y, e_theta, confidence, measured[0], measured[1],
                         command_duty[0], command_duty[1], rear_rpm[0], rear_rpm[1],
                         roll_deg, pitch_deg, float(pos[0]), float(pos[1]), float(pos[2])])
        # Duty drives the DC-motor torque/speed line directly: the drive target
        # is the no-load speed for the applied volts and the constant damping is
        # the stall/no-load slope.  There is no velocity servo and no PID.
        side_duty = np.clip(
            np.array([applied_duty[0], applied_duty[0], applied_duty[1], applied_duty[1]]) * motor_side_gain,
            -float(config["motor"]["max_duty"]), float(config["motor"]["max_duty"]),
        )
        # Severity describes the chassis, so it comes from the left/right pair
        # before the four-wheel expansion above.
        breakaway = effective_breakaway_duty(scenario, scrub_severity(applied_duty))
        # The pack droops under load and over the run.  Same formula as the
        # training lane: linear in elapsed fraction of the episode budget.
        sag_scenario = dict(scenario)
        sag_scenario["supply_voltage_v"] = float(scenario["supply_voltage_v"]) * (
            1.0 - float(scenario.get("supply_sag_fraction", 0.0))
            * min(1.0, step * dt / max(1e-6, duration))
        )
        no_load = np.array(
            [duty_to_volts(float(value), sag_scenario, breakaway) for value in side_duty]
        ) * radps_per_volt
        robot.set_dof_velocity_targets(np.asarray([wheel_signs * no_load]), dof_indices=wheel_indices)
        # Physics stays at dt; ``sim.render()`` above refreshes the camera at
        # camera_hz without advancing the timeline a second time.
        sim.step(render=False, update_fabric=True)
        position, orientation = robot.get_world_poses()
        pos = position.numpy()[0]
        quat = orientation.numpy()[0]
        roll, pitch = roll_pitch_deg(quat)
        max_tilt = max(max_tilt, abs(roll), abs(pitch))
        progress, total = track_progress(pos[:2], points, closed_track)
        travelled_distance += float(np.linalg.norm(pos[:2] - previous_xy))
        previous_xy = pos[:2].copy()
        completion_tolerance = float(config["evaluation"]["completion_tolerance_m"])
        if closed_track:
            near_start = float(np.linalg.norm(pos[:2] - np.asarray([start_x, start_y]))) <= completion_tolerance
            completed = travelled_distance >= total - completion_tolerance and near_start
        else:
            completed = progress >= total - completion_tolerance
        if completed:
            completion_time, failure_reason = step * dt, "completed"
            break
        if max_line_loss_s > float(config["evaluation"]["max_line_loss_s"]):
            failure_reason = "line_lost"
            break
        if pos[2] < -0.01 or max_tilt > float(config["evaluation"]["max_roll_pitch_deg"]):
            failure_reason = "unstable_physics"
            break

    progress, total = track_progress(pos[:2], points, closed_track)
    if closed_track:
        progress = min(total, travelled_distance)
    success = failure_reason == "completed"
    with (output_dir / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "e_y", "e_theta_rad", "line_confidence", "rpm_left", "rpm_right", "duty_left", "duty_right", "rear_rpm_left_diagnostic", "rear_rpm_right_diagnostic", "roll_deg", "pitch_deg", "x_m", "y_m", "z_m"])
        writer.writerows(logs)
    with (output_dir / "perception_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "control_time_s", "rendered_frame_time_s", "state", "threshold_px", "threshold_guard",
            "dark_fraction", "trace_points", "track_progress_m", "track_segment_index", "segment_fraction",
            "distance_to_track_m", "x_m", "y_m", "z_m", "backend", "raw_e_y",
            "raw_e_theta_rad", "raw_confidence_logit", "confidence_probability", "inference_ms",
            "geometry_valid", "geometry_e_y", "geometry_e_theta_rad", "frame_age_s",
        ])
        writer.writerows(perception_logs)
    stage.GetRootLayer().Export(str(output_dir / "line_following_scene.usda"))
    summary = {
        "success": success, "reason": failure_reason, "completion_time_s": completion_time,
        "allowed_duration_s": duration,
        "progress_m": progress, "track_length_m": total, "max_line_loss_s": max_line_loss_s,
        "max_tilt_deg": max_tilt, "final_position_m": [float(value) for value in pos],
        "mean_abs_duty": float(np.mean(np.abs(np.asarray(logs)[:, 6:8]))) if logs else 0.0,
        "dataset_samples": len(dataset_records),
        "scenario": scenario, "wheel_dofs": robot.dof_names,
        "policy_backend": policy_backend,
        "policy_checkpoint": str(checkpoint_path) if policy_backend == "rl" else None,
        "policy_id": policy_id,
        "perception_backend": "classical",
    }
    write_json(output_dir / "episode_summary.json", summary)
    rgb.detach()
    if args.gui:
        print("GUI is open: RobotCamera is active. Select TeachingOverviewCamera from the Camera menu to inspect the track.")
        while app.is_running():
            app.update()
    sim.stop()
    for path, image, is_mask in deferred_debug_images:
        if is_mask:
            save_mask_image(path, image)
        else:
            save_rgb_image(path, image)
    if dataset_root is not None:
        labels_dir = dataset_root / "labels"
        scenarios_dir = dataset_root / "scenarios"
        labels_dir.mkdir(parents=True, exist_ok=True)
        scenarios_dir.mkdir(parents=True, exist_ok=True)
        for path, image in deferred_dataset_images:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_rgb_image(path, image)
        label_path = labels_dir / f"{dataset_prefix}.csv"
        fieldnames = [
            "image", "capture_time_s", "track_progress_m", "label_valid", "e_y", "e_theta_rad",
            "classical_valid", "classical_e_y", "classical_e_theta_rad", "perception_state", "scenario_seed",
            "dataset_split", "episode_id",
        ]
        with label_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset_records)
        write_json(scenarios_dir / f"{dataset_prefix}.json", scenario)
    app.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root() / "isaac_sim/config/default.json")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "isaac_sim/output")
    parser.add_argument("--gui", action="store_true", help="Open the teaching GUI and keep Isaac Sim running after the episode.")
    parser.add_argument("--headless", action="store_true", help="Compatibility flag; headless is the default without --gui.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm", choices=("two_roi", "centerline_lookahead"), help="Override controller.algorithm for a comparison run.")
    parser.add_argument(
        "--policy-backend", choices=("analytic", "reference", "deployed", "rl"), default="analytic",
        help=("Analytical baseline, optional public reference header, locally deployed learner "
              "header, or RSL-RL checkpoint."),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="RSL-RL .pt checkpoint required with --policy-backend rl.",
    )
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--save-debug-frame", action="store_true")
    parser.add_argument(
        "--save-perception-debug", action="store_true",
        help="Save frames at camera-loss transitions alongside perception_diagnostics.csv.",
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        help="Optional dataset root; saves post-artifact RGB frames and geometry-derived labels after the episode.",
    )
    parser.add_argument(
        "--dataset-prefix", help="Filename prefix within --dataset-dir (defaults to the episode seed).",
    )
    parser.add_argument(
        "--dataset-stride", type=int, default=6,
        help="Save every Nth valid camera frame for a dataset (default: 6 = 5 Hz at a 30 Hz camera).",
    )
    parser.add_argument(
        "--dataset-max-frames", type=int, default=0,
        help="Maximum dataset frames per episode; 0 means no limit.",
    )
    parser.add_argument(
        "--dataset-split", choices=("train", "validation", "test"),
        help="Explicit episode-level split written into dataset labels.",
    )
    parser.add_argument(
        "--dataset-episode-id",
        help="Stable episode identifier written into dataset labels; defaults to --dataset-prefix.",
    )
    args = parser.parse_args()
    if args.dataset_stride < 1:
        raise SystemExit("--dataset-stride must be positive")
    if args.dataset_max_frames < 0:
        raise SystemExit("--dataset-max-frames must be non-negative")
    if args.policy_backend == "rl" and args.checkpoint is None:
        raise SystemExit("--policy-backend rl requires --checkpoint <model.pt>.")
    if args.policy_backend != "rl" and args.checkpoint is not None:
        raise SystemExit("--checkpoint is valid only with --policy-backend rl.")
    if args.policy_backend in {"reference", "deployed"}:
        artifact_name = "reference" if args.policy_backend == "reference" else "generated"
        artifact_dir = project_root() / "firmware" / artifact_name
        missing = [
            path.name
            for path in (
                artifact_dir / "line_following_policy.h",
                artifact_dir / "line_following_policy_manifest.json",
            )
            if not path.is_file()
        ]
        if missing:
            if args.policy_backend == "deployed":
                raise SystemExit(
                    "No learner policy is deployed. Complete docs/TRAINING.md and run "
                    "tools/project.py export-header before using --policy-backend deployed."
                )
            raise SystemExit(f"Reference policy is incomplete; missing: {', '.join(missing)}")
    config = load_config(args.config)
    if args.algorithm:
        config["controller"]["algorithm"] = args.algorithm
    if not Path(config["robot"]["usd_asset"]).is_file():
        raise SystemExit(f"Missing {config['robot']['usd_asset']}. Run import_robot.py first.")
    scenario = sample_episode(config, args.seed, args.randomize)
    summary = run_episode(copy.deepcopy(config), args, scenario)
    print(json.dumps({key: summary[key] for key in ("success", "reason", "completion_time_s", "progress_m")}, indent=2))


if __name__ == "__main__":
    main()
