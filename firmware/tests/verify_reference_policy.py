#!/usr/bin/env python3
"""Verify the optional reference-policy loader against its golden vectors."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaac_sim/scripts"))

from policy_header import load_header_policy  # noqa: E402


def main() -> None:
    reference = ROOT / "firmware/reference"
    infer, policy_id = load_header_policy(
        reference / "line_following_policy.h",
        reference / "line_following_policy_manifest.json",
        ROOT / "isaac_sim/config/default.json",
    )
    maximum_error = 0.0
    count = 0
    with (reference / "line_following_policy_vectors.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            observation = tuple(float(row[name]) for name in (
                "e_y", "e_theta_rad", "line_confidence", "rpm_left", "rpm_right",
                "duty_prev_left", "duty_prev_right",
            ))
            expected = np.asarray(
                [float(row["target_duty_left"]), float(row["target_duty_right"])], dtype=np.float32,
            )
            actual = np.asarray(infer(observation), dtype=np.float32)
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            count += 1
    if count != 512 or maximum_error > 2.0e-5:
        raise SystemExit(
            f"Reference Python policy FAIL: vectors={count}, max_target_error={maximum_error:.3e}"
        )
    print(f"Reference Python policy PASS: {policy_id}, vectors={count}, max_target_error={maximum_error:.3e}")


if __name__ == "__main__":
    main()
