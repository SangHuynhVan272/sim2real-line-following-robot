#!/usr/bin/env python3
"""Export a direct-duty policy as portable C and golden test vectors.

Without ``--onnx-model`` this exports the analytical baseline.  With it, the
tool accepts only the straight 7 -> 64 -> 64 -> 2 ELU actor exported by
``play_policy_rl.py`` and writes the same firmware ABI as a C-only MLP.  In both
cases the generated header has no Isaac, Python, ONNX, ESP-DL, heap, or ML
runtime dependency.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from run_line_following import enforce_minimum_loaded_command, load_config, policy, sample_episode


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_relative_path(path: Path, label: str) -> str:
    """Return a portable repository-relative path or reject the export.

    The manifest is committed while checkpoints and ONNX files are not. It must
    never capture the exporting user's home directory.
    """
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must be inside the repository: {path}") from error


def c_float(value: float) -> str:
    """Format a finite Python float as a C single-precision literal."""
    text = format(float(value), ".9g")
    if "e" not in text and "E" not in text and "." not in text:
        text += ".0"
    return f"{text}f"


def render_header_prefix(
    config: dict,
    config_sha256: str,
    *,
    policy_kind: str,
    onnx_sha256: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Return the ABI and perception portion shared by both policy headers."""
    controller = config["controller"]
    motor = config["motor"]
    robot = config["robot"]
    vision = config["vision"]
    physics = config["physics"]
    evaluation = config["evaluation"]
    resolution = robot["camera_resolution_px"]
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("robot.camera_resolution_px must contain [width, height]")
    threshold_mode = str(vision.get("threshold_mode", "absolute"))
    if threshold_mode not in {"absolute", "adaptive_otsu"}:
        raise ValueError(f"Unsupported vision.threshold_mode for firmware export: {threshold_mode}")
    centerline_rows = [float(value) for value in vision["centerline_rows_fraction"]]
    if not centerline_rows:
        raise ValueError("vision.centerline_rows_fraction must not be empty")
    encoder_counts = int(robot.get("encoder_counts_per_wheel_rev", 0))
    constants = {
        "LF_POLICY_HZ": 1.0 * float(physics["policy_hz"]),
        "LF_CAMERA_HZ": 1.0 * float(physics["camera_hz"]),
        "LF_WHEEL_RADIUS_M": float(robot["wheel_radius_m"]),
        "LF_TRACK_WIDTH_M": float(robot["track_width_m"]),
        "LF_TARGET_SPEED_MPS": float(controller["target_speed_mps"]),
        "LF_KP_LATERAL": float(controller["kp_lateral"]),
        "LF_KP_HEADING": float(controller["kp_heading"]),
        "LF_MAX_YAW_RATE_RADPS": float(controller["max_yaw_rate_radps"]),
        "LF_STEERING_SIGN": float(controller["steering_sign"]),
        "LF_MIN_CONFIDENCE_SPEED_SCALE": float(controller["minimum_confidence_speed_scale"]),
        "LF_MIN_SPEED_MPS": float(vision["minimum_speed_mps"]),
        "LF_SPEED_SLOWDOWN_GAIN": float(vision["speed_slowdown_gain"]),
        "LF_SUPPLY_VOLTAGE_V": float(motor["supply_voltage_v"]),
        "LF_DRIVER_DROP_V": float(motor["driver_drop_v"]),
        "LF_MEASURED_AT_VOLTS": float(motor["measured_at_volts"]),
        "LF_NO_LOAD_WHEEL_RADPS": float(motor["no_load_wheel_radps"]),
        "LF_STALL_TORQUE_NM": float(motor["stall_torque_nm"]),
        "LF_COULOMB_FRICTION_NM": float(motor["coulomb_friction_nm"]),
        "LF_VISCOUS_FRICTION_NM_PER_RADPS": float(motor["viscous_friction_nm_per_radps"]),
        "LF_MAX_DUTY": float(motor["max_duty"]),
        "LF_DUTY_RATE_LIMIT_PER_S": float(motor["duty_rate_limit_per_s"]),
        # Measured loaded-vehicle plant dead band.  The RL path does not clamp
        # to this value; the actor was trained through the same dead band.
        "LF_LOADED_BREAKAWAY_DUTY": float(motor["loaded_breakaway_duty"]),
        "LF_MINIMUM_LOADED_COMMAND_DUTY": float(motor["minimum_loaded_command_duty"]),
        # Skid steer only changes heading by sliding all four tyres, so a turn
        # has a higher dead band than a straight run.  Without this the emitted
        # pivot sits under the plant threshold and the wheels never break away.
        "LF_SCRUB_COMMAND_MARGIN": float(motor["scrub_command_margin"]),
        # Commands far below breakaway are released to zero rather than lifted
        # into a large unintended turn. The plant makes no useful motion below
        # breakaway either way.
        "LF_MINIMUM_COMMAND_RELEASE_DUTY": float(
            motor["minimum_command_release_fraction"]
        ) * float(motor["minimum_loaded_command_duty"]),
    }
    lines = [
        "/* Generated by isaac_sim/scripts/export_policy.py.  Do not edit manually. */",
        f"/* default.json SHA-256: {config_sha256} */",
        f"/* policy kind: {policy_kind} */",
        "#ifndef LINE_FOLLOWING_POLICY_H",
        "#define LINE_FOLLOWING_POLICY_H",
        "",
        "#include <math.h>",
        "",
        "/* Frozen observation ABI: [e_y, e_theta, confidence, rpm_l, rpm_r, duty_prev_l, duty_prev_r]. */",
        "typedef struct {",
        "    float e_y;",
        "    float e_theta_rad;",
        "    float line_confidence;",
        "    float rpm_left;",
        "    float rpm_right;",
        "    float duty_prev_left;",
        "    float duty_prev_right;",
        "} line_following_observation_t;",
        "",
        "typedef struct {",
        "    float duty_left;",
        "    float duty_right;",
        "} line_following_action_t;",
        "",
    ]
    if onnx_sha256 is not None:
        lines.insert(3, f"/* actor ONNX SHA-256: {onnx_sha256} */")
    # A build identity the firmware can print at boot.  A comment cannot be
    # read back off a running board, and "is the board flashed with this
    # header?" is otherwise unanswerable from a serial log alone.
    identity = (onnx_sha256 or config_sha256)[:12]
    lines.append(f'#define LF_POLICY_ID "{policy_kind}:{identity}"')
    lines.append(f'#define LF_POLICY_CONFIG_ID "{config_sha256[:12]}"')
    lines.extend(f"#define {name} {c_float(value)}" for name, value in constants.items())
    lines.append(f"#define LF_POLICY_IS_RL {1 if policy_kind == 'rl_mlp' else 0}")
    lines.append(f"#define LF_ENCODER_COUNTS_PER_WHEEL_REV {encoder_counts}")
    return lines, {
        "evaluation": evaluation,
        "resolution": resolution,
        "robot": robot,
        "threshold_mode": threshold_mode,
        "vision": vision,
        "centerline_rows": centerline_rows,
    }


def render_header(config: dict, config_sha256: str) -> str:
    """Render the analytical C implementation that exactly mirrors ``policy()``."""
    lines, context = render_header_prefix(config, config_sha256, policy_kind="analytic")
    evaluation = context["evaluation"]
    resolution = context["resolution"]
    robot = context["robot"]
    threshold_mode = context["threshold_mode"]
    vision = context["vision"]
    centerline_rows = context["centerline_rows"]
    lines.extend([
        f"#define LF_CAMERA_WIDTH_PX {int(resolution[0])}",
        f"#define LF_CAMERA_HEIGHT_PX {int(resolution[1])}",
        f"#define LF_CAMERA_HFOV_DEG {c_float(float(robot['camera_hfov_deg']))}",
        f"#define LF_MAX_LINE_LOSS_S {c_float(float(evaluation['max_line_loss_s']))}",
        f"#define LF_VISION_USE_ADAPTIVE_OTSU {1 if threshold_mode == 'adaptive_otsu' else 0}",
        f"#define LF_VISION_BLACK_THRESHOLD_PX {c_float(float(vision['black_threshold']))}",
        f"#define LF_VISION_ADAPTIVE_MIN_DARK_FRACTION {c_float(float(vision['adaptive_minimum_dark_fraction']))}",
        f"#define LF_VISION_ADAPTIVE_MIN_BRIGHT_FRACTION {c_float(float(vision['adaptive_minimum_bright_fraction']))}",
        f"#define LF_VISION_ADAPTIVE_MIN_CONTRAST_PX {c_float(float(vision['adaptive_minimum_contrast_px']))}",
        f"#define LF_VISION_ADAPTIVE_THRESHOLD_MIN_PX {c_float(float(vision['adaptive_threshold_min_px']))}",
        f"#define LF_VISION_ADAPTIVE_THRESHOLD_MAX_PX {c_float(float(vision['adaptive_threshold_max_px']))}",
        f"#define LF_VISION_ROI_NEAR_FRACTION {c_float(float(vision['roi_rows_fraction'][0]))}",
        f"#define LF_VISION_LOOKAHEAD_ROW_FRACTION {c_float(float(vision['lookahead_row_fraction']))}",
        f"#define LF_VISION_MIN_CENTERLINE_POINTS {int(vision['minimum_centerline_points'])}",
        f"#define LF_VISION_MIN_RUN_WIDTH_PX {int(vision['minimum_run_width_px'])}",
        f"#define LF_VISION_CENTERLINE_ROW_HALFHEIGHT_PX {int(vision['centerline_row_halfheight_px'])}",
        f"#define LF_VISION_CENTERLINE_ROW_COUNT {len(centerline_rows)}",
        "",
        "/* Camera-to-observation constants generated from default.json. */",
        "static const float LF_VISION_CENTERLINE_ROWS_FRACTION[LF_VISION_CENTERLINE_ROW_COUNT] = {",
        "    " + ", ".join(c_float(value) for value in centerline_rows),
        "};",
        "",
    ])
    lines.extend([
        "",
        "static inline float lf_clip(float value, float low, float high) {",
        "    return fminf(high, fmaxf(low, value));",
        "}",
        "",
        "static inline line_following_action_t lf_make_action(float duty_left, float duty_right) {",
        "    line_following_action_t action;",
        "    action.duty_left = duty_left;",
        "    action.duty_right = duty_right;",
        "    return action;",
        "}",
        "",
        "/* How tight a turn a duty pair asks for: 0 straight, 0.5 about one",
        " * stopped wheel, 1 pivot in place.  Scale free, because stiction is a",
        " * threshold on duty and must not move when the robot is merely asked",
        " * to go faster along the same arc. */",
        "static inline float lf_scrub_severity(float duty_left, float duty_right) {",
        "    const float difference = fabsf(duty_left - duty_right);",
        "    const float common = fabsf(duty_left + duty_right);",
        "    return difference / fmaxf(difference + common, 1.0e-9f);",
        "}",
        "",
        "/* One lift using the severity of the duty as handed in. */",
        "static inline line_following_action_t lf_minimum_loaded_pass(line_following_action_t duty) {",
        "    /* A turn is charged its scrub load, so a pivot has to clear a higher",
        "     * dead band than a straight run.  Severity is read from the request,",
        "     * before any lift, so the rule cannot chase its own output. */",
        "    const float minimum = LF_MINIMUM_LOADED_COMMAND_DUTY",
        "        + LF_SCRUB_COMMAND_MARGIN * lf_scrub_severity(duty.duty_left, duty.duty_right);",
        "    const float left_mag = fabsf(duty.duty_left);",
        "    const float right_mag = fabsf(duty.duty_right);",
        "    if (duty.duty_left * duty.duty_right > 0.0f && fminf(left_mag, right_mag) > 1.0e-9f) {",
        "        const float offset = fmaxf(0.0f, minimum - fminf(left_mag, right_mag));",
        "        duty.duty_left += copysignf(offset, duty.duty_left);",
        "        duty.duty_right += copysignf(offset, duty.duty_right);",
        "    } else {",
        "        if (left_mag > 1.0e-9f) duty.duty_left = copysignf(fmaxf(left_mag, minimum), duty.duty_left);",
        "        if (right_mag > 1.0e-9f) duty.duty_right = copysignf(fmaxf(right_mag, minimum), duty.duty_right);",
        "    }",
        "    duty.duty_left = lf_clip(duty.duty_left, -LF_MAX_DUTY, LF_MAX_DUTY);",
        "    duty.duty_right = lf_clip(duty.duty_right, -LF_MAX_DUTY, LF_MAX_DUTY);",
        "    return duty;",
        "}",
        "",
        "/* Clear the loaded dead zone while preserving left/right differential duty.",
        " *",
        " * Two passes.  The plant charges scrub against the duty it is *given*, but",
        " * one pass can only read severity from the request, and lifting an",
        " * asymmetric request makes it more of a pivot than it was -- so a single",
        " * pass under-lifts exactly the hard turns that must not stall.  The second",
        " * pass sees the severity the plant will see. */",
        "static inline float lf_release_below_minimum(float duty) {",
        "    return (fabsf(duty) >= LF_MINIMUM_COMMAND_RELEASE_DUTY) ? duty : 0.0f;",
        "}",
        "",
        "static inline line_following_action_t lf_minimum_loaded_pair(line_following_action_t duty) {",
        "    duty.duty_left = lf_release_below_minimum(duty.duty_left);",
        "    duty.duty_right = lf_release_below_minimum(duty.duty_right);",
        "    return lf_minimum_loaded_pass(lf_minimum_loaded_pass(duty));",
        "}",
        "",
        "/* Open-loop inverse of the calibrated DC-motor line.  This is not a speed PID. */",
        "static inline float lf_wheel_speed_to_duty(float wheel_radps) {",
        "    const float radps_per_volt = LF_NO_LOAD_WHEEL_RADPS / LF_MEASURED_AT_VOLTS;",
        "    const float damping = (LF_STALL_TORQUE_NM / LF_MEASURED_AT_VOLTS) / radps_per_volt",
        "        + LF_VISCOUS_FRICTION_NM_PER_RADPS;",
        "    const float load = LF_COULOMB_FRICTION_NM",
        "        + LF_VISCOUS_FRICTION_NM_PER_RADPS * fabsf(wheel_radps);",
        "    const float volts = (fabsf(wheel_radps) + load / damping) / radps_per_volt;",
        "    float duty = (volts + LF_DRIVER_DROP_V) / LF_SUPPLY_VOLTAGE_V;",
        "    if (fabsf(wheel_radps) <= 1.0e-9f) return 0.0f;",
        "    return lf_clip(copysignf(duty, wheel_radps), -LF_MAX_DUTY, LF_MAX_DUTY);",
        "}",
        "",
        "/* Compute the desired duty before the firmware slew limiter. */",
        "static inline line_following_action_t lf_policy_target(const line_following_observation_t *obs) {",
        "    const float yaw_unclamped = LF_KP_LATERAL * obs->e_y + LF_KP_HEADING * obs->e_theta_rad;",
        "    const float yaw_rate = LF_STEERING_SIGN * lf_clip(",
        "        yaw_unclamped, -LF_MAX_YAW_RATE_RADPS, LF_MAX_YAW_RATE_RADPS);",
        "    float speed = LF_TARGET_SPEED_MPS / (1.0f + LF_SPEED_SLOWDOWN_GAIN * fabsf(obs->e_theta_rad));",
        "    speed = fmaxf(LF_MIN_SPEED_MPS, speed);",
        "    speed *= fmaxf(LF_MIN_CONFIDENCE_SPEED_SCALE, obs->line_confidence);",
        "    const float left_radps = (speed - yaw_rate * LF_TRACK_WIDTH_M * 0.5f) / LF_WHEEL_RADIUS_M;",
        "    const float right_radps = (speed + yaw_rate * LF_TRACK_WIDTH_M * 0.5f) / LF_WHEEL_RADIUS_M;",
        "    return lf_minimum_loaded_pair(lf_make_action(",
        "        lf_wheel_speed_to_duty(left_radps),",
        "        lf_wheel_speed_to_duty(right_radps)",
        "    ));",
        "}",
        "",
        "/* Apply the same configured safety envelope as Isaac Sim, then write duty to the H-bridge. */",
        "static inline line_following_action_t lf_policy_step(const line_following_observation_t *obs) {",
        "    line_following_action_t target = lf_policy_target(obs);",
        "    target = lf_minimum_loaded_pair(target);",
        "    const float slew = LF_DUTY_RATE_LIMIT_PER_S / LF_POLICY_HZ;",
        "    return lf_make_action(",
        "        lf_clip(target.duty_left, obs->duty_prev_left - slew, obs->duty_prev_left + slew),",
        "        lf_clip(target.duty_right, obs->duty_prev_right - slew, obs->duty_prev_right + slew)",
        "    );",
        "}",
        "",
        "#endif  /* LINE_FOLLOWING_POLICY_H */",
        "",
    ])
    return "\n".join(lines)


def _onnx_attribute(node: Any, name: str, default: Any) -> Any:
    """Return one decoded ONNX node attribute without importing ONNX at module load."""
    for attribute in node.attribute:
        if attribute.name == name:
            import onnx

            return onnx.helper.get_attribute_value(attribute)
    return default


def extract_sequential_actor(onnx_path: Path) -> tuple[list[dict[str, Any]], Any, str]:
    """Extract the exact actor graph that the firmware MLP can reproduce.

    This deliberately accepts only the fixed actor exported by this project:
    three Gemm layers with ELU after the first two.  Refusing an unfamiliar
    graph is safer than emitting a plausible but wrong duty policy.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as error:
        raise RuntimeError(
            "ONNX export requires the 'onnx' Python package in env_isaaclab."
        ) from error

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    graph = model.graph
    initializers = {
        tensor.name: np.asarray(numpy_helper.to_array(tensor), dtype=np.float32)
        for tensor in graph.initializer
    }
    graph_inputs = [value.name for value in graph.input if value.name not in initializers]
    if len(graph_inputs) != 1 or len(graph.output) != 1:
        raise ValueError("RL C export requires exactly one non-constant input and one output.")

    current = graph_inputs[0]
    layers: list[dict[str, Any]] = []
    for node in graph.node:
        if node.op_type == "Constant":
            raise ValueError("RL C export refuses Constant nodes; export a folded sequential actor.")
        dynamic_inputs = [name for name in node.input if name and name not in initializers]
        if len(dynamic_inputs) != 1 or dynamic_inputs[0] != current or len(node.output) != 1:
            raise ValueError(
                "RL C export requires one unbranched data path; "
                f"cannot reproduce node '{node.name or node.op_type}'."
            )
        if node.op_type == "Identity":
            current = node.output[0]
            continue
        if node.op_type == "Gemm":
            if len(node.input) not in {2, 3} or node.input[0] != current or node.input[1] not in initializers:
                raise ValueError("RL C export supports Gemm(data, constant_weight[, constant_bias]) only.")
            if int(_onnx_attribute(node, "transA", 0)) != 0:
                raise ValueError("RL C export refuses Gemm transA != 0.")
            weight = initializers[node.input[1]]
            if weight.ndim != 2:
                raise ValueError("RL C export requires each Gemm weight to be rank 2.")
            trans_b = int(_onnx_attribute(node, "transB", 0))
            if trans_b not in {0, 1}:
                raise ValueError("RL C export refuses an invalid Gemm transB value.")
            canonical_weight = weight if trans_b else weight.T
            alpha = float(_onnx_attribute(node, "alpha", 1.0))
            beta = float(_onnx_attribute(node, "beta", 1.0))
            canonical_weight = np.asarray(alpha * canonical_weight, dtype=np.float32)
            if len(node.input) == 3:
                if node.input[2] not in initializers:
                    raise ValueError("RL C export requires a constant Gemm bias.")
                bias = np.asarray(beta * initializers[node.input[2]], dtype=np.float32).reshape(-1)
            else:
                bias = np.zeros(canonical_weight.shape[0], dtype=np.float32)
            if bias.size != canonical_weight.shape[0]:
                raise ValueError("RL C export found a Gemm bias with the wrong output width.")
            layers.append({"weights": canonical_weight, "bias": bias, "activation": "none"})
        elif node.op_type == "Elu":
            if not layers:
                raise ValueError("RL C export found ELU before a dense layer.")
            alpha = float(_onnx_attribute(node, "alpha", 1.0))
            if not np.isclose(alpha, 1.0, rtol=0.0, atol=1e-7):
                raise ValueError(f"RL C export supports ELU alpha=1 only, got {alpha}.")
            if layers[-1]["activation"] != "none":
                raise ValueError("RL C export found more than one activation after a dense layer.")
            layers[-1]["activation"] = "elu"
        else:
            raise ValueError(
                f"RL C export refuses ONNX op '{node.op_type}'; expected only Gemm, Elu, or Identity."
            )
        current = node.output[0]

    if current != graph.output[0].name:
        raise ValueError("RL C export path does not end at the actor output.")
    layout = [(int(layer["weights"].shape[1]), int(layer["weights"].shape[0])) for layer in layers]
    activations = [str(layer["activation"]) for layer in layers]
    if layout != [(7, 64), (64, 64), (64, 2)] or activations != ["elu", "elu", "none"]:
        raise ValueError(
            "RL C export requires the frozen actor 7->64(ELU)->64(ELU)->2, "
            f"got {layout} with activations {activations}."
        )
    return layers, model, graph_inputs[0]


def actor_forward(layers: list[dict[str, Any]], observation: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Run the extracted float32 actor exactly as the emitted C loop does.

    The accumulation is written out rather than handed to ``@``.  Both are
    float32, but numpy dispatches the matmul to BLAS, which blocks and
    vectorises the sum while the emitted C walks the row start to finish.
    Re-ordering float32 additions can move the last few bits enough to fail a
    tight golden-vector tolerance. Matching the emitted loop order makes the
    vectors test the generated firmware code rather than a BLAS-specific
    reduction order.
    """
    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    for layer in layers:
        weights = np.asarray(layer["weights"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        accumulated = np.empty(weights.shape[0], dtype=np.float32)
        for out in range(weights.shape[0]):
            total = bias[out]
            row = weights[out]
            for index in range(weights.shape[1]):
                total = np.float32(total + np.float32(row[index] * values[index]))
            accumulated[out] = total
        values = accumulated
        if layer["activation"] == "elu":
            values = values.copy()
            negative = values <= 0.0
            values[negative] = np.expm1(values[negative])
        elif layer["activation"] != "none":
            raise AssertionError(f"Unexpected extracted activation: {layer['activation']}")
    return values


def verify_actor_extraction(layers: list[dict[str, Any]], model: Any, input_name: str) -> None:
    """Refuse export unless the extracted float32 forward pass matches ONNX."""
    try:
        from onnx.reference import ReferenceEvaluator
    except ImportError as error:
        raise RuntimeError("ONNX ReferenceEvaluator is required to verify RL C export.") from error

    evaluator = ReferenceEvaluator(model)
    rng = np.random.default_rng(1730)
    worst = 0.0
    for _ in range(256):
        observation = np.asarray([
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.25, 1.25),
            rng.uniform(0.0, 1.0),
            rng.uniform(-200.0, 200.0),
            rng.uniform(-200.0, 200.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        ], dtype=np.float32)
        with np.errstate(over="ignore"):
            onnx_output = np.asarray(
                evaluator.run(None, {input_name: observation.reshape(1, 7)})[0], dtype=np.float32
            ).reshape(-1)
        extracted_output = actor_forward(layers, observation)
        worst = max(worst, float(np.max(np.abs(onnx_output - extracted_output))))
    if worst > 2e-5:
        raise RuntimeError(f"RL actor extraction mismatch: max |ONNX - C reference| = {worst:.3e}.")
    print(f"Verified actor extraction: max |ONNX - C reference| = {worst:.3e} over 256 ABI samples")


def _append_vision_constants(lines: list[str], context: dict[str, Any]) -> None:
    """Append the classical-perception constants needed by the firmware sketch."""
    evaluation = context["evaluation"]
    resolution = context["resolution"]
    robot = context["robot"]
    threshold_mode = context["threshold_mode"]
    vision = context["vision"]
    centerline_rows = context["centerline_rows"]
    lines.extend([
        f"#define LF_CAMERA_WIDTH_PX {int(resolution[0])}",
        f"#define LF_CAMERA_HEIGHT_PX {int(resolution[1])}",
        f"#define LF_CAMERA_HFOV_DEG {c_float(float(robot['camera_hfov_deg']))}",
        f"#define LF_MAX_LINE_LOSS_S {c_float(float(evaluation['max_line_loss_s']))}",
        f"#define LF_VISION_USE_ADAPTIVE_OTSU {1 if threshold_mode == 'adaptive_otsu' else 0}",
        f"#define LF_VISION_BLACK_THRESHOLD_PX {c_float(float(vision['black_threshold']))}",
        f"#define LF_VISION_ADAPTIVE_MIN_DARK_FRACTION {c_float(float(vision['adaptive_minimum_dark_fraction']))}",
        f"#define LF_VISION_ADAPTIVE_MIN_BRIGHT_FRACTION {c_float(float(vision['adaptive_minimum_bright_fraction']))}",
        f"#define LF_VISION_ADAPTIVE_MIN_CONTRAST_PX {c_float(float(vision['adaptive_minimum_contrast_px']))}",
        f"#define LF_VISION_ADAPTIVE_THRESHOLD_MIN_PX {c_float(float(vision['adaptive_threshold_min_px']))}",
        f"#define LF_VISION_ADAPTIVE_THRESHOLD_MAX_PX {c_float(float(vision['adaptive_threshold_max_px']))}",
        f"#define LF_VISION_ROI_NEAR_FRACTION {c_float(float(vision['roi_rows_fraction'][0]))}",
        f"#define LF_VISION_LOOKAHEAD_ROW_FRACTION {c_float(float(vision['lookahead_row_fraction']))}",
        f"#define LF_VISION_MIN_CENTERLINE_POINTS {int(vision['minimum_centerline_points'])}",
        f"#define LF_VISION_MIN_RUN_WIDTH_PX {int(vision['minimum_run_width_px'])}",
        f"#define LF_VISION_CENTERLINE_ROW_HALFHEIGHT_PX {int(vision['centerline_row_halfheight_px'])}",
        f"#define LF_VISION_CENTERLINE_ROW_COUNT {len(centerline_rows)}",
        "",
        "/* Camera-to-observation constants generated from default.json. */",
        "static const float LF_VISION_CENTERLINE_ROWS_FRACTION[LF_VISION_CENTERLINE_ROW_COUNT] = {",
        "    " + ", ".join(c_float(value) for value in centerline_rows),
        "};",
        "",
    ])


def render_rl_header(
    config: dict,
    config_sha256: str,
    onnx_sha256: str,
    layers: list[dict[str, Any]],
    blind_duty_scale: float,
) -> str:
    """Render an ML-runtime-free C header for the frozen direct-duty actor."""
    lines, context = render_header_prefix(
        config,
        config_sha256,
        policy_kind="rl_mlp",
        onnx_sha256=onnx_sha256,
    )
    lines.append(f"#define LF_BLIND_DUTY_SCALE {c_float(blind_duty_scale)}")
    _append_vision_constants(lines, context)
    layer_count = len(layers)
    max_width = max(max(int(layer["weights"].shape[0]), int(layer["weights"].shape[1])) for layer in layers)
    parameter_count = sum(int(layer["weights"].size + layer["bias"].size) for layer in layers)
    lines.extend([
        "/* Frozen actor: 7 -> 64(ELU) -> 64(ELU) -> 2. No ML runtime or heap. */",
        f"#define LF_RL_INPUT_DIM {int(layers[0]['weights'].shape[1])}",
        f"#define LF_RL_OUTPUT_DIM {int(layers[-1]['weights'].shape[0])}",
        f"#define LF_RL_LAYER_COUNT {layer_count}",
        f"#define LF_RL_MAX_WIDTH {max_width}",
        f"#define LF_RL_PARAMETER_COUNT {parameter_count}",
        "#define LF_RL_ACT_NONE 0",
        "#define LF_RL_ACT_ELU 1",
        "",
    ])
    for index, layer in enumerate(layers):
        weights = np.asarray(layer["weights"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        out_width, in_width = weights.shape
        lines.append(f"/* RL layer {index}: {in_width} -> {out_width}, {layer['activation']}. */")
        lines.append(f"static const float LF_RL_W{index}[{out_width} * {in_width}] = {{")
        for row in weights:
            lines.append("    " + ", ".join(c_float(float(value)) for value in row) + ",")
        lines.append("};")
        lines.append(
            f"static const float LF_RL_B{index}[{out_width}] = {{ "
            + ", ".join(c_float(float(value)) for value in bias) + " };"
        )
        lines.append("")
    activations = {"none": "LF_RL_ACT_NONE", "elu": "LF_RL_ACT_ELU"}
    lines.extend([
        "static const int LF_RL_LAYER_IN[LF_RL_LAYER_COUNT] = { "
        + ", ".join(str(int(layer["weights"].shape[1])) for layer in layers) + " };",
        "static const int LF_RL_LAYER_OUT[LF_RL_LAYER_COUNT] = { "
        + ", ".join(str(int(layer["weights"].shape[0])) for layer in layers) + " };",
        "static const int LF_RL_LAYER_ACT[LF_RL_LAYER_COUNT] = { "
        + ", ".join(activations[str(layer["activation"])] for layer in layers) + " };",
        "static const float *const LF_RL_WEIGHTS[LF_RL_LAYER_COUNT] = { "
        + ", ".join(f"LF_RL_W{index}" for index in range(layer_count)) + " };",
        "static const float *const LF_RL_BIASES[LF_RL_LAYER_COUNT] = { "
        + ", ".join(f"LF_RL_B{index}" for index in range(layer_count)) + " };",
        "",
        "static inline float lf_clip(float value, float low, float high) {",
        "    return fminf(high, fmaxf(low, value));",
        "}",
        "",
        "static inline line_following_action_t lf_make_action(float duty_left, float duty_right) {",
        "    line_following_action_t action;",
        "    action.duty_left = duty_left;",
        "    action.duty_right = duty_right;",
        "    return action;",
        "}",
        "",
        "/* How tight a turn a duty pair asks for: 0 straight, 0.5 about one",
        " * stopped wheel, 1 pivot in place.  Scale free, because stiction is a",
        " * threshold on duty and must not move when the robot is merely asked",
        " * to go faster along the same arc. */",
        "static inline float lf_scrub_severity(float duty_left, float duty_right) {",
        "    const float difference = fabsf(duty_left - duty_right);",
        "    const float common = fabsf(duty_left + duty_right);",
        "    return difference / fmaxf(difference + common, 1.0e-9f);",
        "}",
        "",
        "/* One lift using the severity of the duty as handed in. */",
        "static inline line_following_action_t lf_minimum_loaded_pass(line_following_action_t duty) {",
        "    /* A turn is charged its scrub load, so a pivot has to clear a higher",
        "     * dead band than a straight run.  Severity is read from the request,",
        "     * before any lift, so the rule cannot chase its own output. */",
        "    const float minimum = LF_MINIMUM_LOADED_COMMAND_DUTY",
        "        + LF_SCRUB_COMMAND_MARGIN * lf_scrub_severity(duty.duty_left, duty.duty_right);",
        "    const float left_mag = fabsf(duty.duty_left);",
        "    const float right_mag = fabsf(duty.duty_right);",
        "    if (duty.duty_left * duty.duty_right > 0.0f && fminf(left_mag, right_mag) > 1.0e-9f) {",
        "        const float offset = fmaxf(0.0f, minimum - fminf(left_mag, right_mag));",
        "        duty.duty_left += copysignf(offset, duty.duty_left);",
        "        duty.duty_right += copysignf(offset, duty.duty_right);",
        "    } else {",
        "        if (left_mag > 1.0e-9f) duty.duty_left = copysignf(fmaxf(left_mag, minimum), duty.duty_left);",
        "        if (right_mag > 1.0e-9f) duty.duty_right = copysignf(fmaxf(right_mag, minimum), duty.duty_right);",
        "    }",
        "    duty.duty_left = lf_clip(duty.duty_left, -LF_MAX_DUTY, LF_MAX_DUTY);",
        "    duty.duty_right = lf_clip(duty.duty_right, -LF_MAX_DUTY, LF_MAX_DUTY);",
        "    return duty;",
        "}",
        "",
        "/* Clear the loaded dead zone while preserving left/right differential duty.",
        " *",
        " * Two passes.  The plant charges scrub against the duty it is *given*, but",
        " * one pass can only read severity from the request, and lifting an",
        " * asymmetric request makes it more of a pivot than it was -- so a single",
        " * pass under-lifts exactly the hard turns that must not stall.  The second",
        " * pass sees the severity the plant will see. */",
        "static inline float lf_release_below_minimum(float duty) {",
        "    return (fabsf(duty) >= LF_MINIMUM_COMMAND_RELEASE_DUTY) ? duty : 0.0f;",
        "}",
        "",
        "static inline line_following_action_t lf_minimum_loaded_pair(line_following_action_t duty) {",
        "    duty.duty_left = lf_release_below_minimum(duty.duty_left);",
        "    duty.duty_right = lf_release_below_minimum(duty.duty_right);",
        "    return lf_minimum_loaded_pass(lf_minimum_loaded_pass(duty));",
        "}",
        "",
        "static inline float lf_rl_activate(float value, int activation) {",
        "    if (activation == LF_RL_ACT_ELU) {",
        "        return value > 0.0f ? value : expf(value) - 1.0f;",
        "    }",
        "    return value;",
        "}",
        "",
        "/* Actor target is tanh(raw network output), exactly like DutyAction.process_actions(). */",
        "static inline line_following_action_t lf_policy_target(const line_following_observation_t *obs) {",
        "    const float input[LF_RL_INPUT_DIM] = {",
        "        obs->e_y, obs->e_theta_rad, obs->line_confidence, obs->rpm_left,",
        "        obs->rpm_right, obs->duty_prev_left, obs->duty_prev_right",
        "    };",
        "    float current[LF_RL_MAX_WIDTH] = {0.0f};",
        "    float next[LF_RL_MAX_WIDTH] = {0.0f};",
        "    for (int i = 0; i < LF_RL_INPUT_DIM; ++i) {",
        "        current[i] = input[i];",
        "    }",
        "    for (int layer = 0; layer < LF_RL_LAYER_COUNT; ++layer) {",
        "        const int in_width = LF_RL_LAYER_IN[layer];",
        "        const int out_width = LF_RL_LAYER_OUT[layer];",
        "        const float *weights = LF_RL_WEIGHTS[layer];",
        "        const float *bias = LF_RL_BIASES[layer];",
        "        for (int out = 0; out < out_width; ++out) {",
        "            float value = bias[out];",
        "            const float *row = weights + out * in_width;",
        "            for (int in = 0; in < in_width; ++in) {",
        "                value += row[in] * current[in];",
        "            }",
        "            next[out] = lf_rl_activate(value, LF_RL_LAYER_ACT[layer]);",
        "        }",
        "        for (int out = 0; out < out_width; ++out) {",
        "            current[out] = next[out];",
        "        }",
        "    }",
        "    return lf_make_action(",
        "        lf_clip(tanhf(current[0]), -LF_MAX_DUTY, LF_MAX_DUTY),",
        "        lf_clip(tanhf(current[1]), -LF_MAX_DUTY, LF_MAX_DUTY)",
        "    );",
        "}",
        "",
        "/* Same max-duty, blind-duty and slew envelope used during RL training. */",
        "static inline line_following_action_t lf_policy_step(const line_following_observation_t *obs) {",
        "    line_following_action_t target = lf_policy_target(obs);",
        "    if (obs->line_confidence <= 0.0f) {",
        "        const float blind_limit = LF_MAX_DUTY * LF_BLIND_DUTY_SCALE;",
        "        target.duty_left = lf_clip(target.duty_left, -blind_limit, blind_limit);",
        "        target.duty_right = lf_clip(target.duty_right, -blind_limit, blind_limit);",
        "    }",
        "    target = lf_minimum_loaded_pair(target);",
        "    const float slew = LF_DUTY_RATE_LIMIT_PER_S / LF_POLICY_HZ;",
        "    return lf_make_action(",
        "        lf_clip(target.duty_left, obs->duty_prev_left - slew, obs->duty_prev_left + slew),",
        "        lf_clip(target.duty_right, obs->duty_prev_right - slew, obs->duty_prev_right + slew)",
        "    );",
        "}",
        "",
        "#endif  /* LINE_FOLLOWING_POLICY_H */",
        "",
    ])
    return "\n".join(lines)


def test_observations(count: int) -> list[tuple[float, ...]]:
    """Return deterministic boundary and interior ABI examples for firmware QA."""
    boundary = [
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -1.2, 1.0, -120.0, 120.0, -1.0, 1.0),
        (1.0, 1.2, 1.0, 120.0, -120.0, 1.0, -1.0),
        (0.8, -0.7, 0.0, 25.0, 25.0, 0.2, -0.2),
        (-0.8, 0.7, 0.35, -25.0, -25.0, -0.2, 0.2),
    ]
    rng = np.random.default_rng(1729)
    random_rows = [
        tuple(float(value) for value in (
            rng.uniform(-1.0, 1.0), rng.uniform(-1.25, 1.25), rng.uniform(0.0, 1.0),
            rng.uniform(-200.0, 200.0), rng.uniform(-200.0, 200.0),
            rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0),
        ))
        for _ in range(max(0, count - len(boundary)))
    ]
    return boundary + random_rows


def write_vectors(path: Path, config: dict, scenario: dict, count: int) -> None:
    """Write target and slew-limited outputs for exact firmware comparison."""
    slew = float(config["motor"]["duty_rate_limit_per_s"]) / float(config["physics"]["policy_hz"])
    fields = [
        "e_y", "e_theta_rad", "line_confidence", "rpm_left", "rpm_right", "duty_prev_left", "duty_prev_right",
        "target_duty_left", "target_duty_right", "duty_left", "duty_right",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for observation in test_observations(count):
            target_left, target_right = policy(observation, config, scenario)
            stepped = enforce_minimum_loaded_command(
                np.asarray([target_left, target_right], dtype=np.float32), config,
            )
            maximum = float(config["motor"]["max_duty"])
            stepped = np.clip(stepped, -maximum, maximum)
            output_left = float(np.clip(stepped[0], observation[5] - slew, observation[5] + slew))
            output_right = float(np.clip(stepped[1], observation[6] - slew, observation[6] + slew))
            writer.writerow(dict(zip(fields, (*observation, target_left, target_right, output_left, output_right))))


def write_rl_vectors(
    path: Path,
    config: dict,
    layers: list[dict[str, Any]],
    blind_duty_scale: float,
    count: int,
) -> None:
    """Write C-header vectors for tanh actor output plus the RL safety envelope."""
    slew = float(config["motor"]["duty_rate_limit_per_s"]) / float(config["physics"]["policy_hz"])
    max_duty = float(config["motor"]["max_duty"])
    fields = [
        "e_y", "e_theta_rad", "line_confidence", "rpm_left", "rpm_right", "duty_prev_left", "duty_prev_right",
        "target_duty_left", "target_duty_right", "duty_left", "duty_right",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for observation in test_observations(count):
            raw = actor_forward(layers, observation)
            target = np.clip(np.tanh(raw), -max_duty, max_duty).astype(np.float32)
            stepped_target = target.copy()
            if observation[2] <= 0.0:
                blind_limit = max_duty * blind_duty_scale
                stepped_target = np.clip(stepped_target, -blind_limit, blind_limit)
            stepped_target = enforce_minimum_loaded_command(stepped_target, config)
            stepped_target = np.clip(stepped_target, -max_duty, max_duty)
            output = np.clip(
                stepped_target,
                np.asarray(observation[5:7], dtype=np.float32) - slew,
                np.asarray(observation[5:7], dtype=np.float32) + slew,
            )
            writer.writerow(dict(zip(fields, (*observation, *target, *output))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root() / "isaac_sim/config/default.json")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Write the export here instead of a versioned folder under firmware/policies/.",
    )
    parser.add_argument(
        "--version-name", type=str, default=None,
        help="Folder name under firmware/policies/ (default: <YYYYMMDD>_<kind>_<onnx sha12>). "
             "Refuses to overwrite an existing version.",
    )
    parser.add_argument(
        "--deploy", action="store_true",
        help="After writing the version folder, mirror it into firmware/generated/ -- the set the "
             "sketch compiles against -- and record the name in firmware/generated/SELECTED.txt.",
    )
    parser.add_argument("--vectors", type=int, default=128, help="Number of deterministic golden vectors (minimum 5).")
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Export the frozen sequential RL actor instead of the analytical baseline.",
    )
    parser.add_argument(
        "--blind-duty-scale",
        type=float,
        default=None,
        help="Override the confidence=0 duty cap embedded in an RL header (default: config rl value).",
    )
    args = parser.parse_args()
    if args.vectors < 5:
        raise SystemExit("--vectors must be at least 5")
    config_bytes = args.config.read_bytes()
    config = load_config(args.config)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    # Every export gets its own folder.  firmware/generated/ used to be written
    # in place, so each new header destroyed the evidence for the one the board
    # was actually running, and a serial banner printing an unfamiliar policy id
    # had nothing to be compared against.  The version folder is the archive and
    # is never overwritten; firmware/generated/ is a mirror of whichever version
    # is currently deployed.
    if args.output_dir is None:
        kind = "rl" if args.onnx_model is not None else "analytic"
        if args.version_name:
            version_name = args.version_name
        else:
            stamp = datetime.date.today().strftime("%Y%m%d")
            if args.onnx_model is not None:
                digest = hashlib.sha256(args.onnx_model.read_bytes()).hexdigest()[:12]
            else:
                digest = config_sha256[:12]
            version_name = f"{stamp}_{kind}_{digest}"
        args.output_dir = project_root() / "firmware/policies" / version_name
        if args.output_dir.exists():
            raise SystemExit(
                f"{args.output_dir} already exists.  Pass --version-name to pick another name; "
                "an existing export is never overwritten."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    header_path = args.output_dir / "line_following_policy.h"
    vectors_path = args.output_dir / "line_following_policy_vectors.csv"
    manifest_path = args.output_dir / "line_following_policy_manifest.json"
    manifest: dict[str, Any] = {
        "config": project_relative_path(args.config, "--config"),
        "config_sha256": config_sha256,
        "header": header_path.name,
        "vectors": vectors_path.name,
        "vector_count": args.vectors,
        "abi": ["e_y", "e_theta_rad", "line_confidence", "rpm_left", "rpm_right", "duty_prev_left", "duty_prev_right"],
        "action": ["duty_left", "duty_right"],
    }
    if args.onnx_model is None:
        if args.blind_duty_scale is not None:
            raise SystemExit("--blind-duty-scale is only valid together with --onnx-model.")
        scenario = sample_episode(config, seed=0, randomize=False)
        header_path.write_text(render_header(config, config_sha256), encoding="utf-8")
        write_vectors(vectors_path, config, scenario, args.vectors)
        manifest["kind"] = "analytic"
    else:
        if not args.onnx_model.is_file():
            raise SystemExit(f"Actor ONNX does not exist: {args.onnx_model}")
        rl_config = config.get("rl", {})
        if not isinstance(rl_config, dict):
            raise SystemExit("default.json rl section must be an object when exporting an RL actor.")
        blind_duty_scale = float(
            args.blind_duty_scale
            if args.blind_duty_scale is not None
            else rl_config.get("blind_duty_scale", 0.0)
        )
        if not np.isfinite(blind_duty_scale) or not 0.0 <= blind_duty_scale <= 1.0:
            raise SystemExit("--blind-duty-scale must be finite and in [0, 1].")
        onnx_sha256 = hashlib.sha256(args.onnx_model.read_bytes()).hexdigest()
        layers, model, input_name = extract_sequential_actor(args.onnx_model)
        verify_actor_extraction(layers, model, input_name)
        header_path.write_text(
            render_rl_header(config, config_sha256, onnx_sha256, layers, blind_duty_scale),
            encoding="utf-8",
        )
        write_rl_vectors(vectors_path, config, layers, blind_duty_scale, args.vectors)
        manifest.update({
            "kind": "rl_mlp",
            # The ONNX file is intentionally not committed. Keep only its
            # portable filename; the digest below is its stable identity.
            "onnx": args.onnx_model.name,
            "onnx_sha256": onnx_sha256,
            "blind_duty_scale": blind_duty_scale,
            "architecture": [7, 64, 64, 2],
            "activation": "elu",
            "parameter_count": sum(int(layer["weights"].size + layer["bias"].size) for layer in layers),
        })
    manifest["header_sha256"] = hashlib.sha256(header_path.read_bytes()).hexdigest()
    manifest["vectors_sha256"] = hashlib.sha256(vectors_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {header_path}")
    print(f"Wrote {vectors_path}")
    print(f"Wrote {manifest_path}")
    if args.deploy:
        deployed = project_root() / "firmware/generated"
        deployed.mkdir(parents=True, exist_ok=True)
        for source in (header_path, vectors_path, manifest_path):
            shutil.copyfile(source, deployed / source.name)
        (deployed / "SELECTED.txt").write_text(
            f"{args.output_dir.name}\n"
            "\nThis directory is a copy of firmware/policies/<name> above -- the set the sketch\n"
            "compiles against.  Do not edit it by hand; re-run export_policy.py --deploy.\n",
            encoding="utf-8",
        )
        print(f"Deployed {args.output_dir.name} -> {deployed}")


if __name__ == "__main__":
    main()
