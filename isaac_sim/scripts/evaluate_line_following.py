#!/usr/bin/env python3
"""Run deterministic nominal/randomized PhysX line-following evaluations.

The default command executes one nominal episode and 20 randomized seeds.
Use ``--tune`` to evaluate the small P-controller grid and update the JSON
only when one candidate passes every required episode.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


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
) -> dict:
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
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    summary_path = output_dir / "episode_summary.json"
    if completed.returncode == 0 and summary_path.is_file():
        return load(summary_path)
    return {"success": False, "reason": "runner_error", "seed": seed, "runner_output": completed.stdout[-4000:]}


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
) -> dict:
    nominal = run_episode(
        config_path, output_dir / "nominal", 0, False, duration_s, algorithm,
        policy_backend, checkpoint, save_perception_debug,
    )
    episodes = [
        run_episode(
            config_path, output_dir / f"seed_{seed:02d}", seed, True, duration_s, algorithm,
            policy_backend, checkpoint, save_perception_debug,
        )
        for seed in range(seed_start, seed_start + seeds)
    ]
    passed = sum(bool(item.get("success")) for item in episodes)
    return {
        "nominal": nominal,
        "episodes": episodes,
        "randomized_passes": passed,
        "randomized_total": seeds,
        "algorithm": algorithm,
        "policy_backend": policy_backend,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "success": bool(nominal.get("success")) and passed == seeds,
        "max_line_loss_s": max((float(item.get("max_line_loss_s", 0.0)) for item in episodes), default=0.0),
        "mean_completion_time_s": sum(float(item["completion_time_s"]) for item in episodes if item.get("completion_time_s") is not None)
        / max(1, sum(item.get("completion_time_s") is not None for item in episodes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root() / "isaac_sim/config/default.json")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "isaac_sim/output/evaluation")
    parser.add_argument("--seeds", type=int, help="Override evaluation.seeds; use 20 for Simulation PASS.")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First randomized seed to evaluate (default: 0).")
    parser.add_argument("--duration-s", type=float, help="Override episode duration for a quick smoke test.")
    parser.add_argument("--algorithm", choices=("two_roi", "centerline_lookahead"), help="Evaluate this controller instead of the configured default.")
    parser.add_argument(
        "--policy-backend", choices=("analytic", "reference", "deployed", "rl"), default="analytic",
    )
    parser.add_argument("--checkpoint", type=Path, help="RSL-RL .pt checkpoint required for --policy-backend rl.")
    parser.add_argument(
        "--save-perception-debug", action="store_true",
        help="Pass the debug flag to every episode so a failed gate has RGB/mask evidence.",
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

    if not args.tune:
        report = evaluate(
            args.config, args.output_dir, seeds, args.seed_start, args.duration_s, args.algorithm,
            args.policy_backend, args.checkpoint, args.save_perception_debug,
        )
        write(args.output_dir / "evaluation_report.json", report)
        nominal_status = "PASS" if report["nominal"].get("success") else "FAIL"
        print(
            f"Evaluation score: {report['randomized_passes']}/{seeds} "
            f"randomized scenarios passed; nominal={nominal_status}"
        )
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
        report = evaluate(
            candidate_path, candidate_dir, seeds, args.seed_start, args.duration_s, args.algorithm,
            args.policy_backend, args.checkpoint, args.save_perception_debug,
        )
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
    write(args.output_dir / "tuning_report.json", report)
    print(f"Tune PASS={report['success']} config_updated={report['config_updated']}")
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
