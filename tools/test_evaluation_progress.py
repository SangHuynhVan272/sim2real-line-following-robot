"""CPU-only tests for evaluation scheduling, progress, and failure handling."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


evaluator = module("evaluation_progress_test_target", "isaac_sim/scripts/evaluate_line_following.py")
project = module("evaluation_project_test_target", "tools/project.py")
session_worker = module("evaluation_session_test_target", "isaac_sim/scripts/evaluate_session_worker.py")


class SchedulingTests(unittest.TestCase):
    def evaluate(self, seed_start=10):
        calls = []
        lock = threading.Lock()

        def episode(config, output, seed, randomized, duration, algorithm, backend, checkpoint, debug, **kwargs):
            with lock:
                calls.append((seed, randomized, output.name, kwargs["label"]))
            return {"success": not (randomized and seed == 11),
                    "reason": "line_lost" if randomized and seed == 11 else "completed",
                    "scenario": {"seed": seed}, "completion_time_s": 14.0,
                    "max_line_loss_s": 0.6 if randomized and seed == 11 else 0.0,
                    "wall_time_s": 0.1}

        stream = io.StringIO()
        with patch.object(evaluator, "run_episode", side_effect=episode), redirect_stdout(stream):
            report = evaluator.evaluate(Path("config.json"), Path("out"), 3, seed_start, None, None,
                                        "rl", Path("model_599.pt"), True, reuse_app=False)
        return report, calls, stream.getvalue()

    def test_serial_progress_counts_nominal_and_seeds_without_changing_score(self):
        report, calls, output = self.evaluate()
        self.assertEqual([(seed, random) for seed, random, _, _ in calls],
                         [(0, False), (10, True), (11, True), (12, True)])
        self.assertEqual(report["randomized_passes"], 2)
        self.assertFalse(report["success"])
        self.assertIn("[DONE 4/4]", output)
        self.assertIn("seed_11: FAIL, reason=line_lost", output)

    def test_holdout_seed_range_does_not_overlap_validation(self):
        report, calls, output = self.evaluate(20)
        self.assertEqual([(seed, random) for seed, random, _, _ in calls],
                         [(0, False), (20, True), (21, True), (22, True)])
        self.assertEqual([item["scenario"]["seed"] for item in report["episodes"]], [20, 21, 22])
        self.assertEqual(report["randomized_passes"], 3)
        self.assertIn("[DONE 4/4]", output)

    def test_invalid_seed_count_does_not_launch_a_child(self):
        with patch.object(evaluator, "run_episode") as run, self.assertRaises(ValueError):
            evaluator.evaluate(Path("c"), Path("out"), 0, 0, None, None, "rl", Path("m"), False)
        run.assert_not_called()

    def test_default_evaluation_uses_one_session_and_keeps_seed_order(self):
        def session(config, output, scenarios, duration, algorithm, backend, checkpoint, debug, record):
            self.assertEqual(scenarios, [("nominal", 0, False), ("seed_20", 20, True), ("seed_21", 21, True)])
            for name, seed, randomized in scenarios:
                record(name, {"success": True, "reason": "completed", "wall_time_s": 1,
                              "scenario": {"seed": seed}})

        with patch.object(evaluator, "run_session", side_effect=session) as run_session, \
                patch.object(evaluator, "run_episode") as run_episode, redirect_stdout(io.StringIO()):
            report = evaluator.evaluate(Path("c"), Path("out"), 2, 20, None, None, "rl", Path("m"), False)
        run_session.assert_called_once()
        run_episode.assert_not_called()
        self.assertEqual(report["execution_mode"], "reused_app")
        self.assertEqual(report["randomized_passes"], 2)


class EpisodeProcessTests(unittest.TestCase):
    def exercise(self, *, fresh=True, return_code=0, heartbeat=False, cancelled=False):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = output / "episode_summary.json"
            if not fresh:
                evaluator.write(summary, {"success": True, "reason": "completed"})
            state = {"calls": 0, "finished": False, "terminated": False}

            class Process:
                def __enter__(self):
                    if fresh:
                        evaluator.write(summary, {"success": True, "reason": "completed"})
                    return self

                def __exit__(self, *args):
                    return False

                def wait(self, timeout=None):
                    state["calls"] += 1
                    if heartbeat and state["calls"] == 1:
                        raise subprocess.TimeoutExpired("test", 1)
                    state["finished"] = True
                    return return_code

                def poll(self):
                    return return_code if state["finished"] else None

                def terminate(self):
                    state["terminated"] = True

                def kill(self):
                    raise AssertionError("Test child should terminate without kill")

            def spawn(command, **kwargs):
                kwargs["stdout"].write("test child output\n")
                self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
                self.assertIn("--headless", command)
                return Process()

            event = threading.Event()
            if cancelled:
                event.set()
            stream = io.StringIO()
            clock = [0.0, 16.0, 17.0] if heartbeat else [0.0, 1.0]
            with patch.object(evaluator.subprocess, "Popen", side_effect=spawn), \
                    patch.object(evaluator.time, "monotonic", side_effect=clock), redirect_stdout(stream):
                if cancelled:
                    with self.assertRaises(InterruptedError):
                        evaluator.run_episode(Path("c"), output, 11, True, None, None,
                                              "rl", Path("m"), False, stop_event=event)
                    self.assertTrue(state["terminated"])
                    return None, stream.getvalue()
                result = evaluator.run_episode(Path("c"), output, 11, True, None, None,
                                               "rl", Path("m"), False, label="seed_11")
            self.assertTrue((output / "runner.log").is_file())
            return result, stream.getvalue()

    def test_start_heartbeat_and_fresh_summary(self):
        result, output = self.exercise(heartbeat=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["wall_time_s"], 17)
        self.assertIn("[START] seed_11", output)
        self.assertIn("[RUNNING] seed_11", output)

    def test_stale_summary_cannot_hide_runner_failure(self):
        result, _ = self.exercise(fresh=False)
        self.assertEqual(result["reason"], "runner_error")
        self.assertFalse(result["success"])
        self.assertIn("test child output", result["runner_output"])

    def test_nonzero_exit_rejects_even_a_written_success_summary(self):
        result, _ = self.exercise(return_code=1)
        self.assertEqual(result["reason"], "runner_error")

    def test_cancellation_terminates_only_the_created_child(self):
        self.exercise(cancelled=True)


class CommandRoutingTests(unittest.TestCase):
    def command(self, name, *arguments):
        with patch.object(sys, "argv", ["project.py", name, "--checkpoint", "m.pt", *arguments]), \
                patch.object(project, "run") as run, patch.object(project, "enforce_teaching_gate"):
            project.main()
        return run.call_args.args[0]

    def test_default_gate_remains_twenty_scenarios(self):
        command = self.command("gate")
        self.assertEqual(command[command.index("--seeds") + 1], "20")
        self.assertNotIn("--workers", command)

    def test_holdout_keeps_seeds_twenty_through_thirty_nine(self):
        command = self.command("holdout")
        self.assertEqual(command[command.index("--seed-start") + 1], "20")
        self.assertEqual(command[command.index("--seeds") + 1], "20")


    def test_reuse_gate_changes_only_the_execution_mode(self):
        normal = self.command("gate")
        reused = self.command("gate", "--reuse-app")
        self.assertEqual(reused, normal)
        self.assertNotIn("--fresh-process", normal)

    def test_fresh_process_changes_only_execution_mode_for_gate_and_holdout(self):
        for name in ("gate", "holdout"):
            with self.subTest(command=name):
                normal = self.command(name)
                fresh = self.command(name, "--fresh-process")
                self.assertEqual([arg for arg in fresh if arg != "--fresh-process"], normal)

    def test_selection_defaults_to_reuse_and_can_request_fresh_process(self):
        for flags, expected in (([], True), (["--fresh-process"], False), (["--reuse-app"], True)):
            with self.subTest(flags=flags), \
                    patch.object(sys, "argv", ["project.py", "select-checkpoint", *flags]), \
                    patch.object(project, "select_best_checkpoint") as select:
                project.main()
                self.assertIs(select.call_args.args[2], expected)

    def test_conflicting_modes_are_rejected_before_launch(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                patch.object(sys, "argv", ["project.py", "gate", "--checkpoint", "m.pt",
                                           "--reuse-app", "--fresh-process"]), \
                patch.object(project, "run") as run, self.assertRaises(SystemExit):
            project.main()
        run.assert_not_called()

    def test_evaluator_cli_defaults_to_reuse_and_honors_explicit_modes(self):
        for flags, expected in (([], True), (["--fresh-process"], False), (["--reuse-app"], True)):
            report = {"nominal": {"success": True}, "episodes": [],
                      "randomized_passes": 20, "wall_time_s": 1,
                      "execution_mode": "reused_app" if expected else "fresh_process"}
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory, \
                    patch.object(sys, "argv", ["evaluate", "--policy-backend", "rl", "--checkpoint", "m.pt",
                                               "--output-dir", directory, "--seeds", "20", *flags]), \
                    patch.object(evaluator, "evaluate", return_value=report) as evaluate, \
                    patch.object(evaluator, "write"), redirect_stdout(io.StringIO()), \
                    self.assertRaises(SystemExit) as exited:
                evaluator.main()
            self.assertEqual(exited.exception.code, 0)
            self.assertIs(evaluate.call_args.args[-1], expected)


class ReportLifecycleTests(unittest.TestCase):
    def invoke(self, output, *, report=None, error=None, tune=False):
        arguments = ["evaluate", "--output-dir", str(output), "--seeds", "20"]
        arguments += ["--tune"] if tune else ["--policy-backend", "rl", "--checkpoint", "m.pt"]
        with patch.object(sys, "argv", arguments), \
                patch.object(evaluator, "evaluate", return_value=report, side_effect=error), \
                redirect_stdout(io.StringIO()):
            evaluator.main()

    def test_failed_or_interrupted_run_archives_previous_pass_before_evaluation(self):
        for error in (RuntimeError("worker failed"), KeyboardInterrupt(), SystemExit(1)):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                previous = {"success": True, "randomized_passes": 20, "checkpoint": "old.pt"}
                report_path = output / "evaluation_report.json"
                evaluator.write(report_path, previous)

                def fail(*args):
                    self.assertFalse(report_path.exists())
                    raise error

                with self.assertRaises(type(error)):
                    self.invoke(output, error=fail)
                self.assertFalse(report_path.exists())
                archived = list((output / "report_history").glob("evaluation_report_*.json"))
                self.assertEqual(len(archived), 1)
                self.assertEqual(evaluator.load(archived[0]), previous)

    def test_successful_run_publishes_only_new_report_and_retains_history(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report_path = output / "evaluation_report.json"
            previous = {"success": True, "randomized_passes": 20, "checkpoint": "old.pt"}
            current = {"nominal": {"success": True}, "episodes": [], "randomized_passes": 19,
                       "wall_time_s": 1, "execution_mode": "reused_app", "checkpoint": "new.pt"}
            evaluator.write(report_path, previous)
            for _ in range(2):
                with self.assertRaises(SystemExit) as exited:
                    self.invoke(output, report=current)
                self.assertEqual(exited.exception.code, 0)
                self.assertEqual(evaluator.load(report_path), current)
            archived = list((output / "report_history").glob("evaluation_report_*.json"))
            self.assertEqual(len(archived), 2)
            self.assertCountEqual([evaluator.load(path) for path in archived], [previous, current])

    def test_first_failed_run_does_not_create_a_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(RuntimeError):
                self.invoke(output, error=RuntimeError("worker failed"))
            self.assertFalse((output / "evaluation_report.json").exists())
            self.assertFalse((output / "report_history").exists())

    def test_failed_tuning_archives_previous_tuning_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous = {"success": True, "config_updated": True}
            evaluator.write(output / "tuning_report.json", previous)
            with self.assertRaises(RuntimeError):
                self.invoke(output, error=RuntimeError("worker failed"), tune=True)
            self.assertFalse((output / "tuning_report.json").exists())
            archived = list((output / "report_history").glob("tuning_report_*.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(evaluator.load(archived[0]), previous)

    def test_atomic_write_failure_preserves_existing_report_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            evaluator.write(path, {"old": True})
            with patch.object(evaluator.Path, "replace", side_effect=OSError("disk error")), \
                    self.assertRaises(OSError):
                evaluator.write(path, {"new": True})
            self.assertEqual(evaluator.load(path), {"old": True})
            self.assertEqual(list(path.parent.iterdir()), [path])


class SessionTests(unittest.TestCase):
    def test_worker_arguments_preserve_checkpoint_seed_and_debug(self):
        job = {"config": "config.json", "output_dir": "out", "policy_backend": "rl",
               "checkpoint": "model_599.pt", "duration_s": None, "algorithm": None,
               "save_perception_debug": True}
        args = session_worker.episode_arguments(job, "seed_11", 11, True)
        self.assertIn("--headless", args)
        self.assertIn("--randomize", args)
        self.assertIn("--save-perception-debug", args)
        self.assertEqual(args[args.index("--checkpoint") + 1], "model_599.pt")
        self.assertEqual(args[args.index("--seed") + 1], "11")
        self.assertNotIn("--duration-s", args)
        nominal = session_worker.episode_arguments(job, "nominal", 0, False)
        self.assertNotIn("--randomize", nominal)

    def exercise(self, return_code):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            progress = output / "session_progress.json"
            recorded = []

            class Process:
                def __enter__(self):
                    # An old run's progress must never be accepted.
                    evaluator.write(progress, {"run_id": "old", "completed": {"stale": {}}})
                    return self

                def __exit__(self, *args):
                    return False

                def poll(self):
                    return return_code

                def wait(self, timeout=None):
                    job = evaluator.load(output / "session_job.json")
                    result = {"success": True, "reason": "completed", "wall_time_s": 1}
                    evaluator.write(progress, {"run_id": job["run_id"], "current": None,
                                               "completed": {"nominal": result, "seed_11": result}})
                    return return_code

            with patch.object(evaluator.subprocess, "Popen", return_value=Process()), redirect_stdout(io.StringIO()):
                if return_code:
                    with self.assertRaisesRegex(RuntimeError, "Reusable-app evaluation failed"):
                        evaluator.run_session(Path("c"), output, [("nominal", 0, False), ("seed_11", 11, True)],
                                              None, None, "rl", Path("m"), True,
                                              lambda name, result: recorded.append(name))
                else:
                    evaluator.run_session(Path("c"), output, [("nominal", 0, False), ("seed_11", 11, True)],
                                          None, None, "rl", Path("m"), True,
                                          lambda name, result: recorded.append(name))
            self.assertEqual(recorded, ["nominal", "seed_11"])

    def test_session_rejects_stale_progress_and_records_every_episode(self):
        self.exercise(0)

    def test_session_shutdown_error_invalidates_batch_even_after_success_summaries(self):
        self.exercise(1)


class SessionLifecycleTests(unittest.TestCase):
    def exercise(self, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job_path = output / "job.json"
            job = {"run_id": "test", "config": "c.json", "output_dir": str(output),
                   "scenarios": [("nominal", 0, False), ("seed_11", 11, True)],
                   "duration_s": None, "algorithm": None, "policy_backend": "rl",
                   "checkpoint": "model_599.pt", "save_perception_debug": False}
            evaluator.write(job_path, job)
            app = SimpleNamespace(close=Mock(), update=Mock())
            factory = Mock(return_value=app)
            fake_isaacsim = ModuleType("isaacsim")
            fake_isaacsim.SimulationApp = factory
            fake_context = ModuleType("isaacsim.core.api.simulation_context")
            fake_context.SimulationContext = SimpleNamespace(clear_instance=Mock())
            calls = []

            def episode():
                config = {"headless": True, "width": 1280, "height": 720, "renderer": "RayTracedLighting"}
                proxy = fake_isaacsim.SimulationApp(config)
                proxy.close()
                app.close.assert_not_called()  # Per-episode close must not end the whole job.
                calls.append(list(sys.argv))
                if fail:
                    raise RuntimeError("test episode failure")
                destination = Path(sys.argv[sys.argv.index("--output-dir") + 1])
                destination.mkdir()
                evaluator.write(destination / "episode_summary.json", {"success": True, "reason": "completed"})

            with patch.object(sys, "argv", ["worker", "--job", str(job_path)]), \
                    patch.dict(sys.modules, {"isaacsim": fake_isaacsim,
                                             "isaacsim.core.api.simulation_context": fake_context}), \
                    patch.object(session_worker.runpy, "run_path", return_value={"main": episode}):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "test episode failure"):
                        session_worker.main()
                else:
                    session_worker.main()
            factory.assert_called_once()
            self.assertIs(fake_isaacsim.SimulationApp, factory)
            app.close.assert_called_once()
            app.update.assert_not_called()  # No added physics/render steps in the adapter.
            state = evaluator.load(output / "session_progress.json")
            if not fail:
                self.assertTrue(state["finished"])
                self.assertEqual(list(state["completed"]), ["nominal", "seed_11"])
                self.assertEqual(fake_context.SimulationContext.clear_instance.call_count, 2)
                self.assertNotIn("--randomize", calls[0])
                self.assertIn("--randomize", calls[1])
            else:
                self.assertFalse(state["finished"])
                self.assertEqual(state["completed"], {})

    def test_one_app_per_job_context_reset_each_episode_no_extra_ticks(self):
        self.exercise()

    def test_application_closed_and_factory_restored_on_failure(self):
        self.exercise(fail=True)


if __name__ == "__main__":
    unittest.main()
