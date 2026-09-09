#!/usr/bin/env python3
"""Cross-platform project commands for Windows and Linux."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(command: list[str]) -> None:
    """Run one command from the repository root and preserve its exit status."""
    print("+", subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault(
        "PYTHONPYCACHEPREFIX",
        str(Path(tempfile.gettempdir()) / "line_following_pycache"),
    )
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def numeric_version(value: str) -> tuple[int, ...]:
    """Return the numeric components of a driver or runtime version."""
    return tuple(int(component) for component in re.findall(r"\d+", value))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 fingerprint of one repository artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_check(*, skip_cpp: bool = False) -> None:
    """Validate repository contents without launching Isaac Sim."""
    errors: list[str] = []
    required = (
        "README.md",
        "docs/TRAINING.md",
        "docs/media/simulation_rollout.gif",
        "isaac_sim/config/default.json",
        "isaac_sim/scripts/run_line_following.py",
        "isaac_sim/scripts/policy_header.py",
        "isaac_sim/scripts/train_policy_bc.py",
        "isaac_sim/scripts/train_policy_rl.py",
        "isaac_sim/scripts/evaluate_line_following.py",
        "isaac_sim/scripts/export_policy.py",
        "isaac_sim/envs/line_following_rl/env_cfg.py",
        "firmware/esp32s3_line_following/esp32s3_line_following.ino",
        "firmware/esp32s3_line_following/classic_line_perception.h",
        "firmware/reference/README.md",
        "firmware/reference/line_following_policy.h",
        "firmware/reference/line_following_policy_manifest.json",
        "firmware/reference/line_following_policy_vectors.csv",
        "firmware/generated/README.md",
        "firmware/tests/verify_policy_vectors.cpp",
        "firmware/tests/verify_reference_policy.py",
        "linefollowingrobot_cad/urdf/linefollowingrobot.urdf",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")
    if not (ROOT / "LICENSE").is_file():
        errors.append("missing LICENSE")
    if not (ROOT / "THIRD_PARTY_NOTICES.md").is_file():
        errors.append("missing THIRD_PARTY_NOTICES.md")

    forbidden_suffixes = {".pt", ".pth", ".onnx", ".usd", ".usda", ".usdc", ".pem", ".key"}
    allowed_csvs = {
        Path("firmware/reference/line_following_policy_vectors.csv"),
        Path("firmware/generated/line_following_policy_vectors.csv"),
    }
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if (
            ".git" in relative.parts
            or "__pycache__" in relative.parts
            or relative.parts[:2] in {("isaac_sim", "assets"), ("isaac_sim", "output")}
            or relative.parts[:2] == ("firmware", "policies")
            or relative.parts[:1] in {("logs",), ("runs",), ("build",), (".pio",), (".venv",)}
        ):
            continue
        if path.suffix.lower() in forbidden_suffixes:
            errors.append(f"generated/private artifact must not be committed: {relative}")
        if path.suffix.lower() == ".csv" and relative not in allowed_csvs:
            errors.append(f"unexpected CSV artifact: {relative}")
        if path.name == ".env" or path.name.startswith(".env."):
            errors.append(f"local environment file must not be committed: {relative}")

    link_pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
    for markdown in ROOT.rglob("*.md"):
        if ".git" in markdown.parts:
            continue
        for target in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (markdown.parent / target).exists():
                errors.append(f"broken Markdown link: {markdown.relative_to(ROOT)} -> {target}")

    parsed_json: dict[str, object] = {}
    for relative in ("isaac_sim/config/default.json", "firmware/reference/line_following_policy_manifest.json"):
        try:
            parsed_json[relative] = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON: {relative}: {exc}")

    manifest = parsed_json.get("firmware/reference/line_following_policy_manifest.json")
    if isinstance(manifest, dict):
        for key in ("config", "onnx", "header", "vectors"):
            value = manifest.get(key)
            if isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):
                errors.append(f"manifest path must be portable: {key}={value}")
        reference = ROOT / "firmware/reference"
        expected_paths = {
            "config": ROOT / "isaac_sim/config/default.json",
            "header": reference / "line_following_policy.h",
            "vectors": reference / "line_following_policy_vectors.csv",
        }
        for key, expected_path in expected_paths.items():
            value = manifest.get(key)
            if not isinstance(value, str):
                errors.append(f"manifest field must be a path string: {key}")
                continue
            actual_path = ROOT / value if key == "config" else reference / value
            if actual_path.resolve() != expected_path.resolve():
                errors.append(f"manifest {key} must name {expected_path.relative_to(ROOT)}, found {value}")

        onnx_name = manifest.get("onnx")
        if isinstance(onnx_name, str) and Path(onnx_name).name != onnx_name:
            errors.append(f"manifest onnx must contain only a portable filename, found {onnx_name}")

        config_path = expected_paths["config"]
        header_path = expected_paths["header"]
        vectors_path = expected_paths["vectors"]
        if config_path.is_file():
            config_digest = file_sha256(config_path)
            if manifest.get("config_sha256") != config_digest:
                errors.append("manifest config_sha256 does not match default.json")
        else:
            config_digest = ""
        if header_path.is_file():
            header_digest = file_sha256(header_path)
            if manifest.get("header_sha256") != header_digest:
                errors.append("reference manifest header_sha256 does not match its header")
            header_text = header_path.read_text(encoding="utf-8")
            if config_digest and config_digest not in header_text:
                errors.append("reference header does not contain the current config SHA-256")
            onnx_digest = manifest.get("onnx_sha256")
            if isinstance(onnx_digest, str) and onnx_digest not in header_text:
                errors.append("reference header does not contain the manifest ONNX SHA-256")
        if vectors_path.is_file():
            if manifest.get("vectors_sha256") != file_sha256(vectors_path):
                errors.append("manifest vectors_sha256 does not match the golden vectors")
            with vectors_path.open(encoding="utf-8") as handle:
                vector_count = max(0, sum(1 for _ in handle) - 1)
            if manifest.get("vector_count") != vector_count:
                errors.append(
                    f"manifest vector_count is {manifest.get('vector_count')}, file contains {vector_count}"
                )

    for path in sorted((ROOT / "isaac_sim").rglob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError) as exc:
            errors.append(f"invalid Python: {path.relative_to(ROOT)}: {exc}")

    if errors:
        for message in errors:
            print(f"[FAIL] {message}")
        raise SystemExit(1)
    print("[PASS] Repository files, JSON, Markdown links and Python syntax")

    if importlib.util.find_spec("numpy") is None:
        print("[SKIP] Reference Python policy check (install NumPy to enable it)")
    else:
        run([PYTHON, str(ROOT / "firmware/tests/verify_reference_policy.py")])

    if skip_cpp:
        print("[SKIP] Desktop C++ checks")
        return
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        print("[SKIP] Desktop C++ checks (install g++ or clang++ to enable them)")
        return
    with tempfile.TemporaryDirectory(prefix="line_following_check_") as directory:
        temporary = Path(directory)
        checks = (
            (
                ROOT / "firmware/tests/verify_policy_vectors.cpp",
                temporary / "verify_policy",
                [str(ROOT / "firmware/reference/line_following_policy_vectors.csv")],
            ),
            (
                ROOT / "firmware/tests/classic_perception_smoke.cpp",
                temporary / "verify_perception",
                [],
            ),
        )
        for source, executable, arguments in checks:
            run([compiler, "-std=gnu++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)])
            run([str(executable), *arguments])


def release_check(*, skip_cpp: bool = False) -> None:
    """Check the physical folder for public-release residue and secrets."""
    source_check(skip_cpp=skip_cpp)
    errors: list[str] = []
    if not (ROOT / "LICENSE").is_file():
        errors.append("add the selected LICENSE before publication")
    if (ROOT / "LICENSE_PENDING.md").exists():
        errors.append("remove LICENSE_PENDING.md before publication")

    for relative in ("isaac_sim/assets", "isaac_sim/output", "firmware/policies"):
        directory = ROOT / relative
        if directory.is_dir() and any(path.is_file() for path in directory.rglob("*")):
            errors.append(f"remove generated files under {relative} before publication")

    for cache in sorted(ROOT.rglob("__pycache__")):
        if ".git" not in cache.parts:
            errors.append(f"remove Python cache directory: {cache.relative_to(ROOT)}")
    for bytecode in sorted(ROOT.rglob("*.pyc")):
        if ".git" not in bytecode.parts:
            errors.append(f"remove Python bytecode: {bytecode.relative_to(ROOT)}")

    excluded_roots = {
        ("isaac_sim", "assets"),
        ("isaac_sim", "output"),
        ("firmware", "policies"),
    }
    text_suffixes = {".c", ".cpp", ".h", ".ino", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
    forbidden_text = (
        (re.compile(r"/(?:home|Users)/[^/\s]+/"), "private absolute home path"),
        (re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"), "private Windows profile path"),
        (re.compile(r"\bgithub_" + r"staging\b"), "staging-directory reference"),
        (re.compile("Sla" + "ck", re.IGNORECASE), "private collaboration reference"),
        (re.compile("arch" + "ive/", re.IGNORECASE), "internal archive reference"),
        (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
        (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"), "credential-like token"),
    )
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if any(relative.parts[: len(prefix)] == prefix for prefix in excluded_roots):
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for pattern, label in forbidden_text:
            if pattern.search(content):
                errors.append(f"{label}: {relative}")

    if (ROOT / ".git").is_dir():
        # Audit exactly the ancestry that a push of the current branch sends.
        # Remote-tracking refs may still point at a history being replaced by
        # an intentional, lease-protected public-history rewrite.
        history = subprocess.run(
            ["git", "log", "HEAD", "-p", "--format=%H %s"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        ).stdout
        history_patterns = (
            (
                re.compile(r"\b" + "v" + r"(?:1[0-9]|20)[a-z]?\b", re.IGNORECASE),
                "internal experiment-version residue",
            ),
            (re.compile(r"/(?:home|Users)/[^/\s]+/"), "private absolute home path"),
            (re.compile(r"\bgithub_" + r"staging\b"), "staging-directory reference"),
            (re.compile("Sla" + "ck", re.IGNORECASE), "private collaboration reference"),
            (re.compile("arch" + "ive/", re.IGNORECASE), "internal archive reference"),
        )
        for pattern, label in history_patterns:
            if pattern.search(history):
                errors.append(f"Git history contains {label}; rewrite unpublished history before pushing")

    if errors:
        for message in errors:
            print(f"[FAIL] {message}")
        raise SystemExit(1)
    print("[PASS] Public-release hygiene")


def doctor(*, skip_gpu: bool = False) -> None:
    """Check the active Python environment and common path mistakes."""
    source_check(skip_cpp=True)
    failures: list[str] = []
    warnings: list[str] = []

    system = platform.system()
    if system not in {"Linux", "Windows"}:
        failures.append(f"unsupported operating system: {system}")
    else:
        print(f"[PASS] Operating system: {platform.platform()}")

    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        failures.append(f"this release is validated for Intel/AMD 64-bit machines, found {machine}")

    if sys.version_info[:2] != (3, 11):
        failures.append(f"Python 3.11 is required, found {platform.python_version()}")
    else:
        print(f"[PASS] Python {platform.python_version()}: {PYTHON}")

    cpu_count = os.cpu_count() or 0
    if cpu_count < 4:
        failures.append(f"Isaac Sim requires at least 4 CPU cores, found {cpu_count}")
    else:
        print(f"[PASS] CPU cores: {cpu_count}")

    if system == "Linux":
        libc_name, libc_version = platform.libc_ver()
        if libc_name != "glibc" or numeric_version(libc_version) < (2, 35):
            failures.append(f"Isaac Sim pip requires GLIBC 2.35 or newer, found {libc_name} {libc_version}")
        else:
            print(f"[PASS] GLIBC {libc_version}")

    in_conda = (Path(sys.prefix) / "conda-meta").is_dir() or bool(os.environ.get("CONDA_PREFIX"))
    in_virtualenv = sys.prefix != sys.base_prefix
    if not in_conda and not in_virtualenv:
        failures.append("Python is not running inside a conda or virtual environment")
    else:
        print(f"[PASS] Isolated Python environment: {sys.prefix}")

    if system == "Windows":
        for label, candidate, replacement in (
            ("repository", str(ROOT), "C:\\robotics\\line-following-robot-pai"),
            ("Python environment", str(Path(sys.prefix)), "C:\\Miniconda3\\envs\\env_isaaclab"),
        ):
            if (
                " " in candidate
                or "onedrive" in candidate.lower()
                or any(ord(character) > 127 for character in candidate)
            ):
                failures.append(
                    f"the Windows {label} path contains spaces, OneDrive or non-ASCII characters; "
                    f"use {replacement}"
                )
            if len(candidate) > 80:
                failures.append(f"the Windows {label} path is too long; use {replacement}")

        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\FileSystem",
            ) as key:
                windows_long_paths = int(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
        except (ImportError, OSError, ValueError):
            windows_long_paths = 0
        if windows_long_paths != 1:
            failures.append("Windows long-path support is disabled; enable it as shown in README section 3.1 and restart")
        else:
            print("[PASS] Windows long-path support enabled")

    if shutil.which("git") is None:
        failures.append("Git is not on PATH")
    else:
        print(f"[PASS] Git: {shutil.which('git')}")
        if system == "Windows":
            long_paths = subprocess.run(
                ["git", "config", "--global", "--get", "core.longpaths"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            ).stdout.strip().lower()
            if long_paths != "true":
                failures.append("enable Windows Git long paths: git config --global core.longpaths true")
            else:
                print("[PASS] Git long paths enabled")

    expected_versions = {
        "isaacsim": ("5.1.0",),
        "torch": ("2.7.0",),
        "torchvision": ("0.22.0",),
        "torchaudio": ("2.7.0",),
        "click": ("8.1.7",),
        "psutil": ("5.9.8",),
        "typing_extensions": ("4.12.2",),
        "numpy": ("1.26.0",),
        "rsl-rl-lib": ("3.1.2",),
        "onnx": ("1.20.1",),
        "pillow": ("11.3.0",),
    }
    for distribution, accepted in expected_versions.items():
        version = package_version(distribution)
        if version is None:
            failures.append(f"missing package: {distribution}")
        elif not any(version == item or version.startswith(item + "+") or version.startswith(item + ".") for item in accepted):
            failures.append(f"{distribution} must be {accepted[0]}, found {version}")
        else:
            print(f"[PASS] {distribution} {version}")

    for module in ("isaaclab", "isaaclab_rl", "rsl_rl"):
        if importlib.util.find_spec(module) is None:
            failures.append(f"missing Python module: {module}; install Isaac Lab v2.3.2 with rsl_rl")
        else:
            print(f"[PASS] import {module}")

    try:
        import psutil

        ram_gib = psutil.virtual_memory().total / (1024 ** 3)
        if ram_gib < 31.0:
            failures.append(f"Isaac Sim requires at least 32 GB RAM, found {ram_gib:.1f} GiB")
        else:
            print(f"[PASS] System RAM: {ram_gib:.1f} GiB")
    except Exception as exc:
        failures.append(f"RAM check failed: {exc}")

    if skip_gpu:
        print("[SKIP] NVIDIA driver and CUDA checks")
    else:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            failures.append("nvidia-smi is not on PATH; install or repair the NVIDIA driver")
        else:
            completed = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if completed.returncode != 0:
                failures.append(f"nvidia-smi failed: {completed.stdout.strip()}")
            else:
                print(f"[PASS] NVIDIA GPU: {completed.stdout.strip()}")
                gpu_rows = [row for row in completed.stdout.splitlines() if row.strip()]
                parsed_rows = [[field.strip() for field in row.split(",")] for row in gpu_rows]
                minimum_driver = (580, 88) if system == "Windows" else (580, 65, 6)
                if not parsed_rows or any(len(row) != 3 for row in parsed_rows):
                    failures.append("could not parse NVIDIA GPU, driver and VRAM information")
                else:
                    driver_versions = [numeric_version(row[1]) for row in parsed_rows]
                    vram_mib = [float(row[2]) for row in parsed_rows]
                    if any(version < minimum_driver for version in driver_versions):
                        required = "580.88" if system == "Windows" else "580.65.06"
                        failures.append(f"Isaac Sim 5.1 requires NVIDIA driver {required} or newer")
                    if max(vram_mib) < 15_000.0:
                        failures.append(
                            f"Isaac Sim 5.1 lists 16 GB VRAM as minimum, found {max(vram_mib) / 1024:.1f} GiB"
                        )
        try:
            import torch

            if not torch.cuda.is_available():
                failures.append("PyTorch cannot see CUDA; check the NVIDIA driver and cu128 PyTorch build")
            else:
                print(f"[PASS] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            failures.append(f"PyTorch CUDA check failed: {exc}")

    free_gib = shutil.disk_usage(ROOT).free / (1024 ** 3)
    if free_gib < 50.0:
        warnings.append(f"only {free_gib:.1f} GiB free; Isaac Sim installation and outputs may need more space")
    else:
        print(f"[PASS] Free disk space: {free_gib:.1f} GiB")

    for message in warnings:
        print(f"[WARN] {message}")
    if failures:
        for message in failures:
            print(f"[FAIL] {message}")
        print("Environment check failed. Fix the first FAIL, reopen the terminal, and run doctor again.")
        raise SystemExit(1)
    print("Environment check PASS. This terminal is ready for the project commands.")


def script(relative: str, *arguments: object) -> list[str]:
    return [PYTHON, str(ROOT / relative), *(str(value) for value in arguments)]


def checked_camera_episode(output_relative: str, *arguments: object) -> None:
    """Run one camera episode and reject Kit's occasional false zero exit code."""
    output_dir = ROOT / output_relative
    summary_path = output_dir / "episode_summary.json"
    summary_path.unlink(missing_ok=True)
    run(script(
        "isaac_sim/scripts/run_line_following.py", *arguments,
        "--output-dir", output_dir,
    ))
    if not summary_path.is_file():
        raise RuntimeError(
            f"Camera episode did not write {summary_path.relative_to(ROOT)}; "
            "inspect the first renderer or GPU error above."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("success"):
        raise RuntimeError(
            f"Camera episode failed: reason={summary.get('reason')}, "
            f"progress={summary.get('progress_m')}."
        )
    print(
        f"[PASS] Camera episode: backend={summary.get('policy_backend')}, "
        f"policy={summary.get('policy_id') or 'n/a'}, "
        f"completion={summary.get('completion_time_s')} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_parser = subparsers.add_parser("source-check", help="Check repository files and optional reference artifacts.")
    source_parser.add_argument("--skip-cpp", action="store_true")
    release_parser = subparsers.add_parser(
        "release-check", help="Run source checks plus public-folder and license checks."
    )
    release_parser.add_argument("--skip-cpp", action="store_true")
    doctor_parser = subparsers.add_parser("doctor", help="Check Python, packages, paths, driver and CUDA.")
    doctor_parser.add_argument("--skip-gpu", action="store_true", help=argparse.SUPPRESS)

    subparsers.add_parser("parity", help="Compare scalar and vectorized simulation lanes.")
    subparsers.add_parser("import-robot", help="Generate the local USD asset from URDF/STL.")
    subparsers.add_parser("smoke", help="Run nominal and randomized analytical camera episodes.")
    subparsers.add_parser(
        "reference-smoke", help="Run the optional reference policy in nominal and randomized camera episodes."
    )
    subparsers.add_parser(
        "reference-gate", help="Evaluate the optional reference policy on randomized seeds 0-19."
    )
    subparsers.add_parser(
        "reference-holdout", help="Evaluate the optional reference policy on randomized seeds 20-39."
    )
    subparsers.add_parser(
        "deployed-smoke", help="Run the learner policy exported to firmware/generated in two camera episodes."
    )
    subparsers.add_parser(
        "deployed-gate", help="Evaluate the learner header in firmware/generated on randomized seeds 0-19."
    )
    subparsers.add_parser(
        "deployed-holdout", help="Evaluate the learner header in firmware/generated on randomized seeds 20-39."
    )
    subparsers.add_parser("gui", help="Open the nominal scene in Isaac Sim.")

    bc_parser = subparsers.add_parser("train-bc", help="Train the behavior-cloning warm start.")
    bc_parser.add_argument("--num-envs", type=int, default=1024)
    bc_parser.add_argument("--output-dir", type=Path, default=Path("isaac_sim/output/rl/bc_candidate"))

    ppo_parser = subparsers.add_parser("train-ppo", help="Train PPO from a BC checkpoint.")
    ppo_parser.add_argument("--num-envs", type=int, default=1024)
    ppo_parser.add_argument("--iterations", type=int, default=600)
    ppo_parser.add_argument("--bc-run", type=Path, default=Path("isaac_sim/output/rl/bc_candidate"))
    ppo_parser.add_argument("--output-dir", type=Path, default=Path("isaac_sim/output/rl/ppo_candidate"))

    gate_parser = subparsers.add_parser("gate", help="Evaluate rendered seeds 0-19.")
    gate_parser.add_argument("--checkpoint", type=Path, required=True)
    gate_parser.add_argument("--output-dir", type=Path, default=Path("isaac_sim/output/evaluation_candidate_gate"))

    holdout_parser = subparsers.add_parser("holdout", help="Evaluate untouched seeds 20-39.")
    holdout_parser.add_argument("--checkpoint", type=Path, required=True)
    holdout_parser.add_argument("--output-dir", type=Path, default=Path("isaac_sim/output/evaluation_candidate_holdout"))

    onnx_parser = subparsers.add_parser("export-onnx", help="Export one accepted checkpoint to ONNX.")
    onnx_parser.add_argument("--checkpoint", type=Path, required=True)
    onnx_parser.add_argument("--onnx", type=Path, required=True)

    header_parser = subparsers.add_parser("export-header", help="Export and deploy ONNX as a C header.")
    header_parser.add_argument("--onnx", type=Path, required=True)
    header_parser.add_argument("--version-name", required=True)

    arguments = parser.parse_args()
    if arguments.command == "source-check":
        source_check(skip_cpp=arguments.skip_cpp)
    elif arguments.command == "release-check":
        release_check(skip_cpp=arguments.skip_cpp)
    elif arguments.command == "doctor":
        doctor(skip_gpu=arguments.skip_gpu)
    elif arguments.command == "parity":
        run(script("isaac_sim/scripts/check_rl_parity.py", "--samples", 1500))
    elif arguments.command == "import-robot":
        run(script("isaac_sim/scripts/import_robot.py", "--headless"))
    elif arguments.command == "smoke":
        checked_camera_episode(
            "isaac_sim/output/smoke/nominal", "--headless", "--seed", 0, "--save-debug-frame",
        )
        checked_camera_episode(
            "isaac_sim/output/smoke/randomized_seed_03", "--headless", "--seed", 3, "--randomize",
        )
    elif arguments.command == "reference-smoke":
        checked_camera_episode(
            "isaac_sim/output/reference_smoke/nominal", "--headless", "--policy-backend", "reference",
            "--seed", 0, "--save-debug-frame",
        )
        checked_camera_episode(
            "isaac_sim/output/reference_smoke/randomized_seed_03", "--headless",
            "--policy-backend", "reference", "--seed", 3, "--randomize",
        )
    elif arguments.command == "reference-gate":
        run(script(
            "isaac_sim/scripts/evaluate_line_following.py", "--policy-backend", "reference",
            "--seeds", 20, "--save-perception-debug",
            "--output-dir", "isaac_sim/output/reference_gate",
        ))
    elif arguments.command == "reference-holdout":
        run(script(
            "isaac_sim/scripts/evaluate_line_following.py", "--policy-backend", "reference",
            "--seed-start", 20, "--seeds", 20, "--save-perception-debug",
            "--output-dir", "isaac_sim/output/reference_holdout",
        ))
    elif arguments.command == "deployed-smoke":
        checked_camera_episode(
            "isaac_sim/output/deployed_smoke/nominal", "--headless", "--policy-backend", "deployed",
            "--seed", 0, "--save-debug-frame",
        )
        checked_camera_episode(
            "isaac_sim/output/deployed_smoke/randomized_seed_03", "--headless",
            "--policy-backend", "deployed", "--seed", 3, "--randomize",
        )
    elif arguments.command == "deployed-gate":
        run(script(
            "isaac_sim/scripts/evaluate_line_following.py", "--policy-backend", "deployed",
            "--seeds", 20, "--save-perception-debug",
            "--output-dir", "isaac_sim/output/deployed_gate",
        ))
    elif arguments.command == "deployed-holdout":
        run(script(
            "isaac_sim/scripts/evaluate_line_following.py", "--policy-backend", "deployed",
            "--seed-start", 20, "--seeds", 20, "--save-perception-debug",
            "--output-dir", "isaac_sim/output/deployed_holdout",
        ))
    elif arguments.command == "gui":
        run(script("isaac_sim/scripts/run_line_following.py", "--gui", "--seed", 0))
    elif arguments.command == "train-bc":
        run(script(
            "isaac_sim/scripts/train_policy_bc.py", "--headless", "--num-envs", arguments.num_envs,
            "--rollout-steps", 512, "--uniform-samples", 2097152, "--uniform-heading-max-rad", 0.8,
            "--epochs", 60, "--batch-size", 8192, "--learning-rate", 0.001, "--seed", 0,
            "--output-dir", arguments.output_dir,
        ))
    elif arguments.command == "train-ppo":
        run(script(
            "isaac_sim/scripts/train_policy_rl.py", "--headless", "--num_envs", arguments.num_envs,
            "--max_iterations", arguments.iterations, "--seed", 0,
            "--init-actor", arguments.bc_run / "model_bc.pt", "--bc-anchor-weight", 0.2,
            "--output-dir", arguments.output_dir,
        ))
    elif arguments.command == "gate":
        run(script(
            "isaac_sim/scripts/evaluate_line_following.py", "--policy-backend", "rl",
            "--checkpoint", arguments.checkpoint, "--seeds", 20, "--save-perception-debug",
            "--output-dir", arguments.output_dir,
        ))
    elif arguments.command == "holdout":
        run(script(
            "isaac_sim/scripts/evaluate_line_following.py", "--policy-backend", "rl",
            "--checkpoint", arguments.checkpoint, "--seed-start", 20, "--seeds", 20,
            "--save-perception-debug", "--output-dir", arguments.output_dir,
        ))
    elif arguments.command == "export-onnx":
        run(script(
            "isaac_sim/scripts/play_policy_rl.py", "--headless", "--num_envs", 1, "--steps", 0,
            "--checkpoint", arguments.checkpoint, "--export-onnx", arguments.onnx,
        ))
    elif arguments.command == "export-header":
        run(script(
            "isaac_sim/scripts/export_policy.py", "--onnx-model", arguments.onnx,
            "--version-name", arguments.version_name, "--vectors", 512, "--deploy",
        ))


if __name__ == "__main__":
    main()
