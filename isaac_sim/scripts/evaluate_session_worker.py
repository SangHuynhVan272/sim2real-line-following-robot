#!/usr/bin/env python3
"""Reuse one headless app while the frozen scalar runner recreates every scene.

Internal evaluator worker, not a training or export command. A complete fresh
stage, SimulationContext, policy load, scenario RNG and sensor pipeline are
created per episode; only the Isaac Sim application stays open.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
import time
from typing import Any


def episode_arguments(job: dict[str, Any], name: str, seed: int, randomized: bool) -> list[str]:
    """Build exactly the scalar evaluator arguments for one scenario."""
    arguments = [str(Path(__file__).with_name("run_line_following.py")),
                 "--headless", "--config", job["config"], "--output-dir",
                 str(Path(job["output_dir"]) / name), "--seed", str(seed),
                 "--policy-backend", job["policy_backend"]]
    if randomized:
        arguments.append("--randomize")
    for key, flag in (("checkpoint", "--checkpoint"), ("duration_s", "--duration-s"),
                      ("algorithm", "--algorithm")):
        if job.get(key) is not None:
            arguments.extend((flag, str(job[key])))
    if job["save_perception_debug"]:
        arguments.append("--save-perception-debug")
    return arguments


def main() -> None:
    """Run an immutable job manifest and publish atomic progress snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    progress_path = Path(job["output_dir"]) / "session_progress.json"
    state: dict[str, Any] = {"run_id": job["run_id"], "current": "Isaac Sim startup",
                             "completed": {}, "finished": False}

    def publish() -> None:
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(progress_path)

    publish()
    namespace = runpy.run_path(str(Path(__file__).with_name("run_line_following.py")),
                              run_name="line_following_session_episode")
    import isaacsim

    launch_config = {"headless": True, "width": 1280, "height": 720, "renderer": "RayTracedLighting"}
    original_factory = isaacsim.SimulationApp
    app = original_factory(launch_config)
    from isaacsim.core.api.simulation_context import SimulationContext

    class EpisodeApp:
        def __getattr__(self, name: str) -> Any:
            return getattr(app, name)

        def close(self) -> None:
            # The runner stops physics and detaches its annotator as usual.
            # The owning worker closes the application after the whole job.
            pass

    def shared_app(config: dict[str, Any]) -> EpisodeApp:
        if config != launch_config:
            raise ValueError("Session runner must use the identical headless launch configuration")
        return EpisodeApp()

    try:
        isaacsim.SimulationApp = shared_app
        for name, seed, randomized in job["scenarios"]:
            state["current"] = name
            publish()
            started = time.monotonic()
            sys.argv = episode_arguments(job, name, seed, randomized)
            summary_path = Path(job["output_dir"]) / name / "episode_summary.json"
            previous_mtime = summary_path.stat().st_mtime_ns if summary_path.is_file() else None
            namespace["main"]()
            if not summary_path.is_file() or summary_path.stat().st_mtime_ns == previous_mtime:
                raise RuntimeError(f"Episode did not write a fresh summary: {name}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["wall_time_s"] = time.monotonic() - started
            summary["runner_log"] = str(Path(job["output_dir"]) / "session_runner.log")
            # Destroy old callbacks/singleton before the next run creates its
            # new stage. Do not add any physics/render steps between episodes.
            SimulationContext.clear_instance()
            state["completed"][name] = summary
            publish()
        state["current"] = None
        state["finished"] = True
        publish()
    finally:
        isaacsim.SimulationApp = original_factory
        app.close()


if __name__ == "__main__":
    main()
