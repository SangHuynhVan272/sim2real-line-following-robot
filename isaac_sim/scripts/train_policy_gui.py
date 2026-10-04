#!/usr/bin/env python3
# Copyright (c) 2026, Huynh Van Sang.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the existing PPO trainer with standard Isaac Lab/Isaac Sim viewports.

This adapter adds collision-free display geometry and two USD cameras only.
It does not supply RGB observations, change rewards, or advance extra physics
steps. The frozen training modules stay unchanged, so existing checkpoints
retain their implementation contract. Use --num_envs to size a GUI run.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import runpy
import sys
from typing import Any


def tape_mesh(track: dict[str, Any]) -> tuple[list[tuple[float, float, float]], list[int]]:
    """Build display-only tape quads, including the open track's finish extension."""
    vertices = [tuple(float(value) for value in point) for point in track["points_xy_m"]]
    if len(vertices) < 2 or any(len(point) != 2 for point in vertices):
        raise ValueError("Display track needs at least two XY points.")
    half_width = float(track["tape_width_m"]) / 2.0
    if half_width <= 0.0:
        raise ValueError("Display tape width must be positive.")
    segments = list(zip(vertices, vertices[1:]))
    if track.get("closed", False):
        segments.append((vertices[-1], vertices[0]))
    else:
        extension = float(track.get("finish_tape_extension_m", 0.0))
        if extension > 0.0:
            # A duplicated last vertex must not turn the finish extension into NaN.
            last_segment = next((pair for pair in reversed(segments) if pair[0] != pair[1]), None)
            if last_segment is None:
                raise ValueError("Display track has no non-zero segment.")
            start, end = last_segment
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            tip = (vertices[-1][0] + extension * (end[0] - start[0]) / length,
                   vertices[-1][1] + extension * (end[1] - start[1]) / length)
            segments.append((vertices[-1], tip))
    segments = [pair for pair in segments if math.dist(*pair) > 1.0e-12]
    normals = []
    for start, end in segments:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        normals.append((-dy / length, dx / length))

    def join(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
        denominator = 1.0 + left[0] * right[0] + left[1] * right[1]
        if denominator < 1.0e-6:
            return half_width * right[0], half_width * right[1]
        scale = half_width / denominator
        return scale * (left[0] + right[0]), scale * (left[1] + right[1])

    points: list[tuple[float, float, float]] = []
    indices: list[int] = []
    for index, (start, end) in enumerate(segments):
        nx, ny = normals[index]
        before, after = (index - 1) % len(segments), (index + 1) % len(segments)
        # Shared miter vertices avoid artificial white cracks between quads
        # in the close-up robot view. All geometry remains display-only.
        sx, sy = (join(normals[before], normals[index]) if segments[before][1] == start
                  else (half_width * nx, half_width * ny))
        ex, ey = (join(normals[index], normals[after]) if segments[after][0] == end
                  else (half_width * nx, half_width * ny))
        offset = len(points)
        points.extend(((start[0] - sx, start[1] - sy, 0.002),
                       (end[0] - ex, end[1] - ey, 0.002),
                       (end[0] + ex, end[1] + ey, 0.002),
                       (start[0] + sx, start[1] + sy, 0.002)))
        indices.extend(range(offset, offset + 4))
    if not points:
        raise ValueError("Display track has no non-zero segment.")
    return points, indices


class TrainingView:
    """Display the live training scene without becoming a sensor/control lane."""

    def __init__(self, env: Any, env_index: int) -> None:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade
        from run_line_following import look_at_matrix

        self.env = env
        self.env_index = env_index
        self.stage = omni.usd.get_context().get_stage()
        self.look_at_matrix = look_at_matrix
        self.Gf = Gf
        root = "/World/TrainingView"
        UsdGeom.Xform.Define(self.stage, root)
        config = env.cfg.line_config
        track = config["track"]
        origins = env.scene.env_origins.detach().cpu().tolist()

        def material(name: str, color: tuple[float, float, float]) -> Any:
            value = UsdShade.Material.Define(self.stage, f"{root}/Looks/{name}")
            shader = UsdShade.Shader.Define(self.stage, f"{value.GetPath()}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            from pxr import Sdf
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
            value.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return value

        floor_material = material("WhiteFloor", (1.0, 1.0, 1.0))
        tape_material = material("BlackTape", (0.015, 0.015, 0.015))
        points, indices = tape_mesh(track)
        half_x, half_y = (float(value) / 2.0 for value in track["floor_size_m"])

        def mesh(path: str, positions: list, faces: list[int], surface: Any) -> None:
            value = UsdGeom.Mesh.Define(self.stage, path)
            value.CreatePointsAttr(positions)
            value.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))
            value.CreateFaceVertexIndicesAttr(faces)
            value.CreateSubdivisionSchemeAttr("none")
            value.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI.Apply(value.GetPrim()).Bind(surface)
            # Deliberately no CollisionAPI/RigidBodyAPI: contact stays on the
            # original training ground plane with the original friction draw.

        for index, origin in enumerate(origins):
            path = f"{root}/env_{index}"
            transform = UsdGeom.Xform.Define(self.stage, path)
            transform.AddTranslateOp().Set(Gf.Vec3d(*origin))
            mesh(f"{path}/Floor", [(-half_x, -half_y, 0.0002),
                                   (half_x, -half_y, 0.0002),
                                   (half_x, half_y, 0.0002),
                                   (-half_x, half_y, 0.0002)], [0, 1, 2, 3], floor_material)
            mesh(f"{path}/Tape", points, indices, tape_material)

        origin = Gf.Vec3d(*origins[env_index])
        overview = UsdGeom.Camera.Define(self.stage, f"{root}/TeachingOverviewCamera")
        overview.CreateFocalLengthAttr(18.0)
        overview.AddTransformOp().Set(look_at_matrix(
            origin + Gf.Vec3d(half_x, -1.7 * half_y, 1.2 * max(half_x, half_y)), origin))
        self.camera = UsdGeom.Camera.Define(self.stage, f"{root}/RobotCamera")
        self.camera.CreateHorizontalApertureAttr(20.955)
        width, height = config["robot"]["camera_resolution_px"]
        self.camera.CreateVerticalApertureAttr(20.955 * float(height) / float(width))
        self.camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 100.0))
        self.camera_transform = self.camera.AddTransformOp()
        self.update()
        self.overview_path = str(overview.GetPath())
        self.initial_view_pending = True
        self.activate_overview()
        print(f"GUI: live PPO training, observing env_{env_index}.", flush=True)
        print(f"GUI cameras: {overview.GetPath()} and {self.camera.GetPath()}.", flush=True)
        print("Use the standard Camera menu; add a second viewport for side-by-side views.", flush=True)
        print("Camera images are display-only; PPO still receives the existing 7-value geometric observation.", flush=True)

    def activate_overview(self) -> None:
        """Select the camera after Kit has synchronized newly authored prims."""
        import omni.kit.commands
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            # Use Kit's standard camera-selection command, including its
            # boundCamera metadata, so later viewport initialization retains
            # the choice. Do not keep forcing it once the user picks a view.
            omni.kit.commands.execute(
                "SetViewportCamera", camera_path=self.overview_path, viewport_api=viewport)

    def update(self) -> None:
        """Follow the selected robot using the training lane's current camera pose."""
        from envs.line_following_rl.mdp.scene_state import camera_pose, get_scenario

        eye, forward, _ = camera_pose(self.env)
        index = self.env_index
        world_eye = eye[index] + self.env.scene.env_origins[index]
        position = self.Gf.Vec3d(*world_eye.detach().cpu().tolist())
        direction = self.Gf.Vec3d(*forward[index].detach().cpu().tolist())
        scenario = get_scenario(self.env)
        roll = math.radians(float(scenario["camera_roll_deg"][index]))
        hfov = math.radians(float(scenario["camera_hfov_deg"][index]))
        self.camera.CreateFocalLengthAttr().Set(20.955 / (2.0 * math.tan(hfov / 2.0)))
        self.camera_transform.Set(self.look_at_matrix(position, position + direction, roll))


def main() -> None:
    """Reuse the frozen trainer, adding a display-only ManagerBasedRLEnv adapter."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--view-env", type=int, default=0)
    visual_args, training_args = parser.parse_known_args()
    if "--headless" in training_args:
        raise SystemExit("GUI trainer cannot use --headless; use train_policy_rl.py for headless training.")
    sys.argv = [str(Path(__file__).with_name("train_policy_rl.py")), *training_args]
    namespace = runpy.run_path(sys.argv[0], run_name="line_following_gui_trainer")
    app = namespace["simulation_app"]
    try:
        if not app.is_running() or namespace["app_launcher"]._headless:
            raise RuntimeError("GUI training needs a desktop display and non-headless AppLauncher.")
        base_env = namespace["ManagerBasedRLEnv"]

        class GuiEnv(base_env):
            def __init__(self, cfg: Any, **kwargs: Any) -> None:
                if not 0 <= visual_args.view_env < cfg.scene.num_envs:
                    raise ValueError("--view-env must index one of the training environments.")
                cfg.viewer.origin_type = "env"
                cfg.viewer.env_index = visual_args.view_env
                super().__init__(cfg=cfg, **kwargs)
                self.training_view = TrainingView(self, visual_args.view_env)

            def step(self, action: Any) -> Any:
                result = super().step(action)
                # Read-only display update. No new simulation steps, RNG draws,
                # action changes, or changes to the returned training tensors.
                self.training_view.update()
                if self.training_view.initial_view_pending:
                    self.training_view.activate_overview()
                    self.training_view.initial_view_pending = False
                return result

        # Patch only this invocation's function globals, not Isaac Lab classes
        # or files. Reuse exactly the trainer's BC/PPO config/save/learn logic.
        namespace["main"].__globals__["ManagerBasedRLEnv"] = GuiEnv
        namespace["main"]()
    finally:
        app.close()


if __name__ == "__main__":
    main()
