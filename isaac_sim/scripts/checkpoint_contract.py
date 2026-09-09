"""Configuration and implementation identity for deployable checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of one artifact without interpreting its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    """Hash every Python module that defines training or camera-lane semantics.

    Hashing only ``default.json`` allowed a checkpoint trained before a reward,
    delay, or motor-model bug fix to look current.  Paths are included in the
    digest so moving/replacing a module cannot preserve the contract by chance.
    """
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "isaac_sim/envs/line_following_rl").rglob("*.py"))
    files.extend(
        root / relative
        for relative in (
            "isaac_sim/scripts/checkpoint_contract.py",
            "isaac_sim/scripts/run_line_following.py",
            "isaac_sim/scripts/train_policy_bc.py",
            "isaac_sim/scripts/train_policy_rl.py",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def config_contract(config_path: Path) -> dict[str, str]:
    """Metadata embedded in every BC/PPO checkpoint created by this project."""
    return {
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "implementation_sha256": implementation_sha256(),
    }


def validate_checkpoint_config(
    checkpoint: dict[str, Any],
    config_path: Path,
    *,
    label: str = "RL checkpoint",
) -> dict[str, Any]:
    """Reject legacy or stale actors before simulation/export can trust them."""
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict) or not isinstance(infos.get("config_sha256"), str):
        raise ValueError(
            f"{label} has no config_sha256 contract; retrain it with the current scripts."
        )
    expected = file_sha256(config_path)
    actual = infos["config_sha256"]
    if actual != expected:
        raise ValueError(
            f"{label} was trained for default.json SHA-256 {actual}, but the selected "
            f"configuration is {expected}; retrain instead of evaluating a stale actor."
        )
    recorded_implementation = infos.get("implementation_sha256")
    if not isinstance(recorded_implementation, str):
        raise ValueError(
            f"{label} has no implementation_sha256 contract; retrain it with the current scripts."
        )
    expected_implementation = implementation_sha256()
    if recorded_implementation != expected_implementation:
        raise ValueError(
            f"{label} was trained for implementation SHA-256 {recorded_implementation}, but "
            f"the current RL/camera implementation is {expected_implementation}; retrain "
            "instead of evaluating a checkpoint across code changes."
        )
    return infos
