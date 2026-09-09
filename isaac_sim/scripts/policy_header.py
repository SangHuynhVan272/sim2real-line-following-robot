"""Load an exported policy C header for checkpoint-free simulation.

The generated C header is the deployment artifact compiled by the ESP32-S3.
The same loader handles the optional public reference fixture and the local
policy produced by a learner's accepted checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np


EXPECTED_ARCHITECTURE = (7, 64, 64, 2)
ARRAY_PATTERN = re.compile(
    r"static const float\s+(?P<name>LF_RL_[WB][0-2])\s*\[[^]]+\]\s*=\s*\{(?P<body>.*?)\};",
    re.DOTALL,
)
FLOAT_SUFFIX_PATTERN = re.compile(r"(?<=\d)f\b")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_arrays(header_text: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for match in ARRAY_PATTERN.finditer(header_text):
        body = FLOAT_SUFFIX_PATTERN.sub("", match.group("body")).strip().rstrip(",")
        values = np.fromstring(body, sep=",", dtype=np.float32)
        arrays[match.group("name")] = values

    expected_sizes = {
        "LF_RL_W0": 64 * 7,
        "LF_RL_B0": 64,
        "LF_RL_W1": 64 * 64,
        "LF_RL_B1": 64,
        "LF_RL_W2": 2 * 64,
        "LF_RL_B2": 2,
    }
    actual_sizes = {name: int(values.size) for name, values in arrays.items()}
    if actual_sizes != expected_sizes:
        raise ValueError(
            "Policy header does not contain the frozen 7->64->64->2 actor: "
            f"expected {expected_sizes}, got {actual_sizes}."
        )
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Policy header contains a non-finite actor parameter.")
    return arrays


def load_header_policy(
    header_path: Path,
    manifest_path: Path,
    config_path: Path,
) -> tuple[Callable[[tuple[float, ...]], tuple[float, float]], str]:
    """Return the actor encoded in ``line_following_policy.h`` and its ID.

    Hash and architecture checks fail early if a user mixes files from two
    exports. The returned function emits the raw actor target; the simulator
    then applies the same blind-driving, loaded-command and slew envelope as
    the firmware's ``lf_policy_step``.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("architecture", ())) != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"Policy manifest architecture must be {EXPECTED_ARCHITECTURE}, "
            f"got {manifest.get('architecture')}."
        )
    if manifest.get("kind") != "rl_mlp" or manifest.get("activation") != "elu":
        raise ValueError("Policy manifest must describe the frozen ELU RL actor.")
    if manifest.get("header_sha256") != _sha256(header_path):
        raise ValueError("Policy header SHA-256 does not match its manifest.")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise ValueError("Policy was exported from a different default.json.")

    header_text = header_path.read_text(encoding="utf-8")
    onnx_digest = manifest.get("onnx_sha256")
    if not isinstance(onnx_digest, str) or onnx_digest not in header_text:
        raise ValueError("Policy header does not contain the manifest policy fingerprint.")
    arrays = _parse_arrays(header_text)
    layers = (
        (arrays["LF_RL_W0"].reshape(64, 7), arrays["LF_RL_B0"]),
        (arrays["LF_RL_W1"].reshape(64, 64), arrays["LF_RL_B1"]),
        (arrays["LF_RL_W2"].reshape(2, 64), arrays["LF_RL_B2"]),
    )

    def infer(observation: tuple[float, ...]) -> tuple[float, float]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"Header inference requires seven finite ABI values, got {values!r}.")
        hidden = values
        for index, (weight, bias) in enumerate(layers):
            hidden = np.asarray(weight @ hidden + bias, dtype=np.float32)
            if index < len(layers) - 1:
                hidden = np.where(hidden > 0.0, hidden, np.expm1(hidden)).astype(np.float32)
        duty = np.tanh(hidden).astype(np.float32)
        if not np.isfinite(duty).all():
            raise RuntimeError("Header actor produced a non-finite duty action.")
        return float(duty[0]), float(duty[1])

    policy_id = f"{manifest['kind']}:{onnx_digest[:12]}"
    return infer, policy_id
