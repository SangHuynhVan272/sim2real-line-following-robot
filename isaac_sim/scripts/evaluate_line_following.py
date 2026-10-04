#!/usr/bin/env python3
"""Run deterministic nominal/randomized PhysX line-following evaluations.

The default command executes one nominal episode and 20 randomized seeds in
one headless application, recreating the scene/context for each episode.
Use ``--fresh-process`` for independent application startup per episode.
Use ``--tune`` to evaluate the small P-controller grid and update the JSON
only when one candidate passes every required episode.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    """Replace JSON atomically so readers never see a partly written report."""
    serialized = json.dumps(value, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False, encoding="utf-8") as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def archive_previous_report(path: Path) -> None:
    """Remove stale final results from the current run's path without losing evidence."""
    if not path.is_file():
        return
    history_dir = path.parent / "report_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    archived_path = history_dir / f"{path.stem}_{uuid.uuid4().hex}{path.suffix}"
    path.replace(archived_path)
    print(f"[REPORT] Previous report archived: {archived_path}. "
          "A new final report is written only after this run finishes.", flush=True)


def run_episode(
    config_path: Path,
    output_dir: Path,
    seed: int,
    randomized: bool,
    duration_s: float | None,
    algorithm: str | None,
    policy_backend: str,
    checkpoint: Path | None,
    save_perception_debug: bool,
    *,
    label: str = "episode",
    stop_event: threading.Event | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "runner.log"
    summary_path = output_dir / "episode_summary.json"
    previous_summary = summary_path.stat().st_mtime_ns if summary_path.is_file() else None
    command = [sys.executable, str(Path(__file__).with_name("run_line_following.py")), "--config", str(config_path),
               "--output-dir", str(output_dir), "--headless", "--seed", str(seed)]
    if randomized:
        command.append("--randomize")
    if duration_s is not None:
        command.extend(["--duration-s", str(duration_s)])
    if algorithm is not None:
        command.extend(["--algorithm", algorithm])
    command.extend(["--policy-backend", policy_backend])
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    if save_perception_debug:
        command.append("--save-perception-debug")
    started = time.monotonic()
    print(f"[START] {label}; log: {log_path}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment) as process:
            heartbeat_at = started + 15.0
            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        raise InterruptedError("Evaluation cancelled")
                    try:
                        return_code = process.wait(timeout=1.0)
                        break
                    except subprocess.TimeoutExpired:
                        now = time.monotonic()
                        if now >= heartbeat_at:
                            print(f"[RUNNING] {label}; elapsed={now - started:.0f}s (Isaac Sim startup/rendering)",
                                  flush=True)
                            heartbeat_at = now + 15.0
            finally:
                # Stop only the child created by this invocation if interrupted.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    elapsed = time.monotonic() - started
    if (return_code == 0 and summary_path.is_file()
            and summary_path.stat().st_mtime_ns != previous_summary):
        result = load(summary_path)
        result["wall_time_s"] = elapsed
        result["runner_log"] = str(log_path)
        return result
    with log_path.open("rb") as log:
        log.seek(max(0, log_path.stat().st_size - 4000))
        tail = log.read().decode("utf-8", errors="replace")
    return {"success": False, "reason": "runner_error", "seed": seed,
            "wall_time_s": elapsed, "runner_log": str(log_path), "runner_output": tail}


def evaluate(
    config_path: Path,
    output_dir: Path,
    seeds: int,
    seed_start: int,
    duration_s: float | None,
    algorithm: str | None,
    policy_backend: str,
    checkpoint: Path | None,
    save_perception_debug: bool,
    reuse_app: bool = True,
) -> dict:
    if seeds < 1:
        raise ValueError("Evaluation needs positive seeds.")
    started = time.monotonic()
    total = seeds + 1
    print(f"Evaluation: nominal + {seeds} randomized seeds ({seed_start}-{seed_start + seeds - 1}); "
          f"mode={'reused app, fresh scenes' if reuse_app else 'fresh processes'}. "
          f"Checkpoint: {checkpoint or policy_backend}", flush=True)
    scenarios = [("nominal", 0, False)] + [
        (f"seed_{seed:02d}", seed, True) for seed in range(seed_start, seed_start + seeds)
    ]
    results: dict[str, dict] = {}
    stop_event = threading.Event()

    def execute(index: int, name: str, seed: int, randomized: bool) -> dict:
        return run_episode(config_path, output_dir / name, seed, randomized, duration_s, algorithm,
                           policy_backend, checkpoint, save_perception_debug,
                           label=f"{name} [task {index}/{total}]", stop_event=stop_event)

    def record(name: str, result: dict) -> None:
        results[name] = result
        status = "PASS" if result.get("success") else "FAIL"
        wall_time = result.get("wall_time_s", 0.0)
        elapsed = time.monotonic() - started
        # A running average, not a promised completion time (failures can take longer).
        eta = elapsed / len(results) * (total - len(results))
        print(f"[DONE {len(results)}/{total}] {name}: {status}, reason={result.get('reason')}, "
              f"episode_wall={wall_time:.1f}s; total_elapsed={elapsed:.0f}s, rough_ETA={eta:.0f}s", flush=True)

    if reuse_app:
        run_session(config_path, output_dir, scenarios, duration_s, algorithm,
                    policy_backend, checkpoint, save_perception_debug, record)
    else:
        for index, (name, seed, randomized) in enumerate(scenarios, start=1):
            record(name, execute(index, name, seed, randomized))
    nominal = results["nominal"]
    # Keep report order deterministic, regardless of completion order.
    episodes = [results[f"seed_{seed:02d}"] for seed in range(seed_start, seed_start + seeds)]
    passed = sum(bool(item.get("success")) for item in episodes)
    return {
        "nominal": nominal,
        "episodes": episodes,
        "randomized_passes": passed,
        "randomized_total": seeds,
        "algorithm": algorithm,
        "policy_backend": policy_backend,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "wall_time_s": time.monotonic() - started,
        "execution_mode": "reused_app" if reuse_app else "fresh_process",
        "success": bool(nominal.get("success")) and passed == seeds,
        "max_line_loss_s": max((float(item.get("max_line_loss_s", 0.0)) for item in episodes), default=0.0),
        "mean_completion_time_s": sum(float(item["completion_time_s"]) for item in episodes if item.get("completion_time_s") is not None)
        / max(1, sum(item.get("completion_time_s") is not None for item in episodes)),
    }


def run_session(
    config_path: Path,
    output_dir: Path,
    scenarios: list[tuple[str, int, bool]],
    duration_s: float | None,
    algorithm: str | None,
    policy_backend: str,
    checkpoint: Path | None,
    save_perception_debug: bool,
    record: Callable[[str, dict], None],
) -> None:
    """Watch a sequential reusable-app worker without hiding progress/errors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    job = {"run_id": run_id, "config": str(config_path.resolve()),
           "output_dir": str(output_dir.resolve()), "scenarios": scenarios,
           "duration_s": duration_s, "algorithm": algorithm, "policy_backend": policy_backend,
           "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
           "save_perception_debug": save_perception_debug}
    job_path = output_dir / "session_job.json"
    progress_path = output_dir / "session_progress.json"
    log_path = output_dir / "session_runner.log"
    write(job_path, job)
    command = [sys.executable, str(Path(__file__).with_name("evaluate_session_worker.py")),
               "--job", str(job_path)]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    seen: set[str] = set()
    current = None
    heartbeat_at = time.monotonic() + 15.0
    current_started = time.monotonic()

    def update_progress() -> None:
        nonlocal current, current_started, heartbeat_at
        if not progress_path.is_file():
            return
        state = load(progress_path)
        if state.get("run_id") != run_id:
            return  # Never reuse a previous run's progress/results.
        for name, result in state.get("completed", {}).items():
            if name not in seen:
                record(name, result)
                seen.add(name)
        if state.get("current") != current:
            current = state.get("current")
            current_started = time.monotonic()
            heartbeat_at = current_started + 15.0
            if current is not None:
                print(f"[START] {current}; session log: {log_path}", flush=True)

    print(f"[START] Opening one Isaac Sim app; session log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment) as process:
            try:
                while True:
                    update_progress()
                    try:
                        return_code = process.wait(timeout=1.0)
                        break
                    except subprocess.TimeoutExpired:
                        now = time.monotonic()
                        if now >= heartbeat_at:
                            print(f"[RUNNING] {current or 'Isaac Sim startup/shutdown'}; "
                                  f"elapsed={now - current_started:.0f}s", flush=True)
                            heartbeat_at = now + 15.0
                update_progress()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    if return_code != 0 or len(seen) != len(scenarios):
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 4000))
            tail = log.read().decode("utf-8", errors="replace")
        # Invalidate the entire batch after a process error, even if some
        # completed episodes had PASS summaries before shutdown failed.
        raise RuntimeError(f"Reusable-app evaluation failed; see {log_path}\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root() / "isaac_sim/config/default.json")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "isaac_sim/output/evaluation")
    parser.add_argument("--seeds", type=int, help="Override evaluation.seeds; use 20 for the standard score.")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First randomized seed to evaluate (default: 0).")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fresh-process", dest="reuse_app", action="store_false",
                       help="Start an independent Isaac Sim process for every episode.")
    modes.add_argument("--reuse-app", dest="reuse_app", action="store_true",
                       help="Keep one headless app open with fresh scenes (already the default).")
    parser.set_defaults(reuse_app=True)
    parser.add_argument("--duration-s", type=float, help="Override episode duration for a quick smoke test.")
    parser.add_argument("--algorithm", choices=("two_roi", "centerline_lookahead"), help="Evaluate this controller instead of the configured default.")
    parser.add_argument(
        "--policy-backend", choices=("analytic", "reference", "deployed", "rl"), default="analytic",
    )
    parser.add_argument("--checkpoint", type=Path, help="RSL-RL .pt checkpoint required for --policy-backend rl.")
    parser.add_argument(
        "--save-perception-debug", action="store_true",
        help="Pass the debug flag to every episode so lower-scoring runs have RGB/mask evidence.",
    )
    parser.add_argument("--tune", action="store_true", help="Run the configured speed/Kp grid and persist a full-pass winner.")
    args = parser.parse_args()
    config = load(args.config)
    seeds = int(args.seeds if args.seeds is not None else config["evaluation"]["seeds"])
    if seeds < 1:
        raise SystemExit("--seeds must be positive")
    if args.seed_start < 0:
        raise SystemExit("--seed-start must be non-negative")
    if args.policy_backend == "rl" and args.checkpoint is None:
        raise SystemExit("--policy-backend rl requires --checkpoint <model.pt>.")
    if args.policy_backend != "rl" and args.checkpoint is not None:
        raise SystemExit("--checkpoint is valid only with --policy-backend rl.")
    if args.tune and args.policy_backend != "analytic":
        raise SystemExit("--tune only applies to the analytical baseline, never to an RL policy.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / ("tuning_report.json" if args.tune else "evaluation_report.json")
    # Invalidate before launching any episodes, including runs later interrupted
    # by a worker error or Ctrl+C. Keep the old result as historical evidence.
    archive_previous_report(report_path)

    if not args.tune:
        report = evaluate(
            args.config, args.output_dir, seeds, args.seed_start, args.duration_s, args.algorithm,
            args.policy_backend, args.checkpoint, args.save_perception_debug,
            args.reuse_app,
        )
        write(report_path, report)
        runner_errors = [
            item for item in [report["nominal"], *report["episodes"]]
            if item.get("reason") == "runner_error"
        ]
        if runner_errors:
            print(
                f"Evaluation could not complete: {len(runner_errors)} episode(s) "
                "ended with a runner/runtime error."
            )
            first_output = str(runner_errors[0].get("runner_output", "")).strip()
            if first_output:
                print(first_output)
            raise SystemExit(1)

        nominal_status = "PASS" if report["nominal"].get("success") else "FAIL"
        print(
            f"Evaluation score: {report['randomized_passes']}/{seeds} "
            f"randomized scenarios passed; nominal={nominal_status}", flush=True,
        )
        print(f"Evaluation time: {report['wall_time_s'] / 60.0:.1f} min "
              f"({report['wall_time_s']:.1f}s); mode={report['execution_mode']}", flush=True)
        raise SystemExit(0)

    tuning = config["evaluation"]["tuning"]
    candidates = list(itertools.product(tuning["target_speed_mps"], tuning["kp_lateral"], tuning["kp_heading"]))
    candidate_reports: list[dict] = []
    for index, (speed, lateral, heading) in enumerate(candidates):
        candidate = json.loads(json.dumps(config))
        candidate["controller"].update({"target_speed_mps": speed, "kp_lateral": lateral, "kp_heading": heading})
        candidate_dir = args.output_dir / f"candidate_{index:02d}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=candidate_dir.parent, delete=False, encoding="utf-8") as handle:
            json.dump(candidate, handle)
            candidate_path = Path(handle.name)
        try:
            report = evaluate(
                candidate_path, candidate_dir, seeds, args.seed_start, args.duration_s, args.algorithm,
                args.policy_backend, args.checkpoint, args.save_perception_debug,
                args.reuse_app,
            )
        finally:
            candidate_path.unlink(missing_ok=True)
        report["controller"] = candidate["controller"]
        candidate_reports.append(report)
        print(f"candidate {index + 1}/{len(candidates)}: {report['randomized_passes']}/{seeds}, nominal={report['nominal'].get('success')}")

    candidate_reports.sort(key=lambda item: (-item["randomized_passes"], item["max_line_loss_s"], item["mean_completion_time_s"]))
    winner = candidate_reports[0]
    report = {"success": winner["success"], "winner": winner, "candidates": candidate_reports}
    if winner["success"]:
        config["controller"].update({key: winner["controller"][key] for key in ("target_speed_mps", "kp_lateral", "kp_heading")})
        write(args.config, config)
        report["config_updated"] = True
    else:
        report["config_updated"] = False
    write(report_path, report)
    print(f"Tune PASS={report['success']} config_updated={report['config_updated']}")
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
