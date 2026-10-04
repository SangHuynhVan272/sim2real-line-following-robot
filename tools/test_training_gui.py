"""CPU-only checks for display geometry and GUI/headless command routing."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gui = load_module("training_gui_tests_target", "isaac_sim/scripts/train_policy_gui.py")
rendered = load_module("rendered_play_tests_target", "isaac_sim/scripts/play_policy_rendered.py")
project = load_module("project_tests_target", "tools/project.py")


class TapeMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = {"points_xy_m": [[0, 0], [1, 0]], "tape_width_m": 0.02,
                      "finish_tape_extension_m": 0.2, "closed": False}

    def test_open_track_extension_and_width(self) -> None:
        original = copy.deepcopy(self.track)
        points, indices = gui.tape_mesh(self.track)
        self.assertEqual(len(points), 8)
        self.assertEqual(indices, list(range(8)))
        self.assertAlmostEqual(max(point[0] for point in points), 1.2)
        self.assertAlmostEqual(max(point[1] for point in points), 0.01)
        self.assertEqual(self.track, original)

    def test_closed_track_has_no_finish_extension(self) -> None:
        self.track["closed"] = True
        points, _ = gui.tape_mesh(self.track)
        self.assertEqual(len(points), 8)
        self.assertEqual(max(point[0] for point in points), 1.0)

    def test_duplicate_final_point(self) -> None:
        self.track["points_xy_m"].append([1, 0])
        points, _ = gui.tape_mesh(self.track)
        self.assertEqual(len(points), 8)
        self.assertAlmostEqual(max(point[0] for point in points), 1.2)

    def test_invalid_geometry(self) -> None:
        self.track["tape_width_m"] = 0
        with self.assertRaises(ValueError):
            gui.tape_mesh(self.track)

    def test_curved_track_has_shared_edges_without_white_cracks(self) -> None:
        self.track["points_xy_m"] = [[0, 0], [1, 0], [1, 1]]
        self.track["finish_tape_extension_m"] = 0
        points, _ = gui.tape_mesh(self.track)
        self.assertEqual(points[1], points[4])
        self.assertEqual(points[2], points[7])


class TrainingCommandTests(unittest.TestCase):
    def command(self, *arguments: str) -> list[str]:
        with patch.object(sys, "argv", ["project.py", "train-ppo", *arguments]), \
                patch.object(project, "run") as run:
            project.main()
        return run.call_args.args[0]

    def test_default_is_gui_with_unchanged_training_recipe(self) -> None:
        command = self.command()
        self.assertTrue(command[1].endswith("train_policy_gui.py"))
        self.assertNotIn("--headless", command)
        self.assertEqual(command[command.index("--view-env") + 1], "0")
        self.assertEqual(command[command.index("--num_envs") + 1], "1024")
        self.assertEqual(command[command.index("--max_iterations") + 1], "600")
        self.assertEqual(command[command.index("--seed") + 1], "0")
        self.assertEqual(command[command.index("--bc-anchor-weight") + 1], "0.2")

    def test_explicit_headless_retains_original_recipe(self) -> None:
        command = self.command("--headless")
        self.assertTrue(command[1].endswith("train_policy_rl.py"))
        self.assertIn("--headless", command)
        self.assertNotIn("--view-env", command)
        self.assertEqual(command[command.index("--num_envs") + 1], "1024")
        self.assertEqual(command[command.index("--max_iterations") + 1], "600")
        self.assertEqual(command[command.index("--bc-anchor-weight") + 1], "0.2")

    def test_gui_reuses_recipe_and_honors_explicit_overrides(self) -> None:
        command = self.command("--num-envs", "16", "--iterations", "8", "--view-env", "2")
        self.assertTrue(command[1].endswith("train_policy_gui.py"))
        self.assertNotIn("--headless", command)
        self.assertEqual(command[command.index("--view-env") + 1], "2")
        self.assertEqual(command[command.index("--num_envs") + 1], "16")
        self.assertEqual(command[command.index("--max_iterations") + 1], "8")
        self.assertEqual(command[command.index("--bc-anchor-weight") + 1], "0.2")

    def test_optional_gui_flag_is_compatible_with_previous_commands(self) -> None:
        self.assertEqual(self.command("--gui"), self.command())

    def test_conflicting_display_flags_are_rejected(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.command("--gui", "--headless")
        self.assertEqual(error.exception.code, 2)

    def test_nonzero_view_env_is_rejected_in_headless_mode(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.command("--headless", "--view-env", "2")
        self.assertEqual(error.exception.code, 2)


class PlaybackCommandTests(unittest.TestCase):
    def command(self, *arguments: str) -> tuple[list[str], object]:
        with patch.object(sys, "argv", ["project.py", "play", *arguments]), \
                patch.object(project, "run") as run, \
                patch.object(project, "resolve_checkpoint", return_value=Path("selected/model_500.pt")) as resolve, \
                redirect_stdout(io.StringIO()):
            project.main()
        return run.call_args.args[0], resolve

    def test_default_uses_selected_checkpoint_and_rendered_camera_gui(self) -> None:
        command, resolve = self.command()
        resolve.assert_called_once_with()
        self.assertTrue(command[1].endswith("play_policy_rendered.py"))
        self.assertIn("--gui", command)
        self.assertNotIn("--headless", command)
        self.assertEqual(command[command.index("--policy-backend") + 1], "rl")
        self.assertEqual(command[command.index("--checkpoint") + 1], "selected/model_500.pt")
        self.assertEqual(command[command.index("--output-dir") + 1], "isaac_sim/output/play")
        self.assertNotIn("--randomize", command)

    def test_explicit_checkpoint_is_not_replaced_by_selection(self) -> None:
        command, resolve = self.command("--checkpoint", "my_run/model_100.pt")
        resolve.assert_not_called()
        self.assertEqual(command[command.index("--checkpoint") + 1], "my_run/model_100.pt")

    def test_gui_and_headless_differ_only_in_display_flag(self) -> None:
        gui_command, _ = self.command("--checkpoint", "my_run/model_100.pt")
        headless_command, _ = self.command("--checkpoint", "my_run/model_100.pt", "--headless")
        self.assertEqual(["--headless" if value == "--gui" else value for value in gui_command], headless_command)

    def test_randomized_preview_forwards_seed_and_output_without_changing_model(self) -> None:
        command, _ = self.command("--randomize", "--seed", "3", "--output-dir", "isaac_sim/output/play_test")
        self.assertIn("--randomize", command)
        self.assertEqual(command[command.index("--seed") + 1], "3")
        self.assertEqual(command[command.index("--output-dir") + 1], "isaac_sim/output/play_test")
        self.assertEqual(command[command.index("--checkpoint") + 1], "selected/model_500.pt")


class RenderedPreviewAdapterTests(unittest.TestCase):
    def exercise(self, gui_enabled: bool, *, fail: bool = False) -> None:
        original_update = Mock()
        app_factory = Mock(return_value=SimpleNamespace(update=original_update))
        fake_isaacsim = SimpleNamespace(SimulationApp=app_factory)
        original_write = Mock()
        args = SimpleNamespace(gui=gui_enabled)
        summary = {"success": True, "reason": "completed"}

        def episode(config, episode_args, scenario):
            self.assertIs(episode_args, args)
            if episode_args.gui:
                app = fake_isaacsim.SimulationApp({"headless": False})
                for _ in range(3):
                    app.update()
            if fail:
                raise RuntimeError("test episode failure")
            # The adapter temporarily patches this function's globals.
            if episode_args.gui:
                globals()["write_json"](Path("scenario.json"), {})
                globals()["write_json"](Path("episode_summary.json"), summary)
            return summary

        def main():
            self.assertIs(globals()["run_episode"]({}, args, {}), summary)

        namespace = {"main": main, "run_episode": episode, "write_json": original_write}
        with patch.object(rendered.runpy, "run_path", return_value=namespace), \
                patch.dict(sys.modules, {"isaacsim": fake_isaacsim}), \
                patch.object(rendered, "bind_robot_camera", side_effect=[False, True]) as bind, \
                patch.object(rendered, "pause_at_result") as pause:
            if fail:
                with self.assertRaisesRegex(RuntimeError, "test episode failure"):
                    rendered.main()
            else:
                rendered.main()
            self.assertIs(fake_isaacsim.SimulationApp, app_factory)
            self.assertNotIn("run_episode", globals())
            self.assertNotIn("write_json", globals())
            if gui_enabled:
                self.assertEqual(original_update.call_count, 3)
                self.assertEqual(bind.call_count, 2)  # Stop forcing camera after binding.
                if fail:
                    pause.assert_not_called()
                else:
                    pause.assert_called_once_with(summary)
                    self.assertEqual(original_write.call_count, 2)
            else:
                app_factory.assert_not_called()
                bind.assert_not_called()
                pause.assert_not_called()

    def test_headless_reuses_runner_without_display_hooks(self) -> None:
        self.exercise(False)

    def test_gui_binds_once_pauses_only_result_and_restores_hooks(self) -> None:
        self.exercise(True)

    def test_gui_hooks_are_restored_even_on_episode_failure(self) -> None:
        self.exercise(True, fail=True)


if __name__ == "__main__":
    unittest.main()
