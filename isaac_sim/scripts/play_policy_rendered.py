#!/usr/bin/env python3
# Copyright (c) 2026, Huynh Van Sang.
# SPDX-License-Identifier: BSD-3-Clause

"""Preview a checkpoint through the unchanged rendered-camera episode runner.

GUI-only presentation fixes bind the initial camera and pause at the saved
episode result. No policy, perception, physics-step, or evaluation logic is
replaced. Headless execution passes through without these display hooks.
"""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Any
from unittest.mock import patch


def bind_robot_camera() -> bool:
    """Bind once when the camera exists; leave later user camera choices alone."""
    import omni.kit.commands
    import omni.usd
    from omni.kit.viewport.utility import get_active_viewport

    stage = omni.usd.get_context().get_stage()
    viewport = get_active_viewport()
    camera_path = "/World/RobotCamera"
    if stage is None or viewport is None or not stage.GetPrimAtPath(camera_path).IsValid():
        return False
    omni.kit.commands.execute("SetViewportCamera", camera_path=camera_path, viewport_api=viewport)
    return True


def pause_at_result(summary: dict[str, Any]) -> None:
    """Freeze the completed episode before the runner enters its GUI hold loop."""
    import omni.timeline

    omni.timeline.get_timeline_interface().pause()
    print(f"Preview finished: success={summary['success']}, reason={summary['reason']}. "
          "GUI paused at the final state; close it before continuing.", flush=True)


def main() -> None:
    """Reuse the scalar runner, with display hooks only for an explicit GUI run."""
    namespace = runpy.run_path(str(Path(__file__).with_name("run_line_following.py")),
                              run_name="line_following_rendered_preview")
    original_episode = namespace["run_episode"]

    def preview_episode(config: dict[str, Any], args: Any, scenario: dict[str, Any]) -> Any:
        if not args.gui:
            return original_episode(config, args, scenario)

        import isaacsim

        original_app = isaacsim.SimulationApp
        original_write = namespace["write_json"]

        def gui_app(launch_config: dict[str, Any]) -> Any:
            app = original_app(launch_config)
            original_update = app.update
            initial_camera_pending = True

            def update() -> None:
                nonlocal initial_camera_pending
                original_update()
                if initial_camera_pending and bind_robot_camera():
                    initial_camera_pending = False

            app.update = update
            return app

        def write_result(path: Path, payload: Any) -> None:
            original_write(path, payload)
            if path.name == "episode_summary.json":
                pause_at_result(payload)

        # Scope both hooks to this preview invocation, not other apps/evaluators.
        with patch.object(isaacsim, "SimulationApp", gui_app), \
                patch.dict(original_episode.__globals__, {"write_json": write_result}):
            return original_episode(config, args, scenario)

    with patch.dict(namespace["main"].__globals__, {"run_episode": preview_episode}):
        namespace["main"]()


if __name__ == "__main__":
    main()
