# Train and export a policy

> [!NOTE]
> Copyable code blocks in this guide are commands intended for Terminal. Status
> messages, example results, interface definitions, and file paths are shown as
> normal text, tables, lists, or notes.

This guide is the detailed training companion to the root README. The root
`README.md` is the single source of truth for the hands-on teaching workflow.
This document expands the training, evaluation, export and diagnostic steps.

Training is stochastic. Every run produces a new candidate; do not expect its
weights to match another person's run. Validation seeds 0-19 form the teaching
deployment gate: `20/20` with `nominal=PASS` is the target, while `19/20`
with `nominal=PASS` is also accepted. Holdout seeds 20-39 are recorded as a
robustness score and do not block deployment after the validation gate passes.
Nothing in `firmware/reference/` is loaded by the train, evaluation, export or
firmware commands below.

## 1. Keep the deployment contract unchanged

| Interface   | Values                                                                                        |
| ----------- | --------------------------------------------------------------------------------------------- |
| Observation | `[e_y, e_theta_rad, line_confidence, rpm_left, rpm_right, duty_left_prev, duty_right_prev]` |
| Action      | `[duty_left, duty_right]`                                                                   |

- `e_y` and both duties are in `[-1, 1]`.
- `e_theta_rad` is in radians; confidence is in `[0, 1]`.
- Front-wheel encoders provide the left/right RPM observations.
- All four motors are driven by side.
- The actor outputs duty directly; do not add a wheel-speed PID.
- Physics, policy and camera rates are 120, 60 and 30 Hz.

Changing this contract, the robot, camera, timing or network invalidates the
old checkpoint and requires a full retrain.

## 2. Validate the simulator

```bash
conda activate env_isaaclab
python tools/project.py doctor
python tools/project.py source-check
python tools/project.py import-robot
python tools/project.py parity
python tools/project.py smoke
```

Fix simulator or perception failures before training. A tape outside the
camera frame cannot be fixed by running PPO longer.

## 3. Train the behavior-cloning warm start

The default recipe uses 1,024 environments, seed 0, 512 rollout steps and
2,097,152 uniform ABI samples:

```bash
python tools/project.py train-bc --output-dir isaac_sim/output/rl/bc_candidate
```

On a 16-31 GB RAM machine, the project is still supported. If host memory is
tight, reduce the environment count, for example:

```bash
python tools/project.py train-bc --num-envs 256 --output-dir isaac_sim/output/rl/bc_candidate
```

Using fewer environments may reduce throughput, but it is preferable to
treating a 16-31 GB machine as an invalid installation.

This creates:

- `isaac_sim/output/rl/bc_candidate/model_bc.pt`
- `isaac_sim/output/rl/bc_candidate/bc_summary.json`

Behavior cloning fits the actor in pre-tanh space. Do not replace it with a
post-tanh fit; that previously produced saturated actors.

## 4. Train PPO from behavior cloning

PPO training opens the standard Isaac Lab/Isaac Sim GUI by default. No
`--gui` flag is required. Run this from the Ubuntu desktop:

```bash
python tools/project.py train-ppo --bc-run isaac_sim/output/rl/bc_candidate --output-dir isaac_sim/output/rl/ppo_candidate --iterations 600
```

When you do not need the GUI (including a workstation without a display),
add `--headless`:

```bash
python tools/project.py train-ppo --headless --bc-run isaac_sim/output/rl/bc_candidate --output-dir isaac_sim/output/rl/ppo_candidate --iterations 600
```

If host RAM is constrained, use the same lower environment count for PPO, for
example `--num-envs 256`. The command uses seed 0 and a BC anchor weight of
`0.2`.

For the standard teaching workflow, no manual checkpoint filename is needed.
The next step automatically uses the newest saved checkpoint, regardless of the
iteration number.

Choose your preferred camera and scene view in the standard GUI: observe an
individual environment or the full scene. No camera-selection flag is needed
in the training command. These are live training robots, not checkpoint replay. See
[README Step 8.2](../README.md#82-train-ppo--gui-by-default)
for the camera/observation distinction. Reducing the environment count
changes the sample budget and can change
the resulting policy; `600` counts PPO iterations, not episodes.

The GUI and headless commands use the same recipe: 1,024 environments, seed 0,
the BC actor, anchor weight `0.2`, and 600 PPO iterations. Each iteration collects
32 steps per environment before updating the policy. The BC warm start means
the robot may already follow the line before PPO's first update.

### Live training views

Use the Camera menu to choose `Perspective`, `TeachingOverviewCamera` or
`RobotCamera`. **Window > Viewport > Viewport 2** opens a second view. You can
inspect one robot or navigate around the scene; no camera flag is required.
The wrapper's robot camera follows environment 0, while Perspective lets you
inspect the other environments freely.

The white floor, black tape and camera images are display-only. PPO receives
the frozen seven-value geometric observation, not RGB images. Evaluation then
uses rendered camera images and the actual perception pipeline, which is why
visually successful training is not sufficient evidence for export.

Watch rewards and episode statistics in Terminal. Wait for
`training complete; checkpoints in ...` before closing Terminal or the GUI.
Rendering adds overhead; add `--num-envs 256` if memory is tight, but record the
change because it alters the sample budget. Keep the documented output directory
for the copy-paste workflow. For separate experiments, choose a different output
directory and pass the resulting checkpoint explicitly to evaluation/export.
Both commands start a fresh run, not a resume, and can overwrite checkpoints
with the same names in an existing output directory.

> [!TIP]
> **Optional — search for a stronger saved checkpoint**
>
> Run the following only when you want to compare all saved PPO checkpoints:
>
> ```bash
> python tools/project.py select-checkpoint
> ```
>
> The selector evaluates every saved `model_*.pt` checkpoint on validation
> seeds 0-19. It first prefers checkpoints that pass the teaching deployment
> gate, then prefers the higher validation score, nominal PASS, and the later
> iteration. If no saved checkpoint passes the gate, the best remaining
> candidate is still recorded for diagnosis. This can take substantially longer
> than evaluating one checkpoint.

## 5. Evaluate the selected checkpoint

Use one command to resolve the checkpoint for the remaining workflow:

```bash
CHECKPOINT="$(python tools/project.py checkpoint-path)"
echo "Using checkpoint: $CHECKPOINT"
```

If `select-checkpoint` was skipped, this resolves the newest saved checkpoint.
If it was run, this resolves the checkpoint selected on validation seeds 0-19.
An older selection is ignored if the latest checkpoint is newer than the
selection file. Keep this Terminal and `CHECKPOINT` unchanged until export.

### 5.1 Deployment gate: seeds 0-19

Run the validation gate:

```bash
python tools/project.py gate --checkpoint "$CHECKPOINT"
```

Evaluation prints `[START]`, a `[RUNNING]` heartbeat every 15 seconds, and
`[DONE n/21]` with the seed, result, elapsed time, and an approximate ETA.
The 21 tasks include nominal plus 20 randomized seeds. The default reuses one
headless app and saves full Isaac Sim output to `session_runner.log`. With
`--fresh-process`, each scenario has its own `runner.log`.
A heartbeat confirms that the evaluator is waiting, not that a potentially
stalled simulator is making forward progress.

For the teaching workflow, the gate passes only when the nominal episode passes
and at least 19 of the 20 randomized validation scenarios pass:

- `20/20` + `nominal=PASS`: target result.
- `19/20` + `nominal=PASS`: accepted teaching gate PASS.
- `18/20` or lower, or `nominal=FAIL`: gate not passed.

Continue only after the final `[PASS] Deployment gate` message. This is a
workflow requirement: the export commands do not read or enforce the gate
report automatically. Do not skip the gate or export a failed candidate.

If the gate does not pass, first compare the saved PPO checkpoints on the same
validation set:

```bash
python tools/project.py select-checkpoint
CHECKPOINT="$(python tools/project.py checkpoint-path)"
python tools/project.py gate --checkpoint "$CHECKPOINT"
```

If the selected checkpoint still does not pass, retrain or continue to
Section 7 for diagnostics. Do not use holdout seeds to select or tune the
checkpoint.

### 5.2 Holdout score: seeds 20-39

After the gate passes, run the untouched holdout on the same checkpoint:

```bash
python tools/project.py holdout --checkpoint "$CHECKPOINT"
```

The holdout is a robustness score, not a deployment gate. Record the result even
when it is below 20/20, and do not repeatedly use seeds 20-39 for checkpoint
selection or tuning.

> [!NOTE]
> Example: **Gate 19/20 with nominal PASS; Holdout 20/20.** This checkpoint
> passes the teaching deployment gate and may continue to export.

`--tune` applies only to the analytical baseline. Never use it to tune an RL
policy on holdout seeds.

### Execution modes and logs

Gate and holdout reuse one headless app per set by default and
recreate the scene/context, camera pipeline, scenario and policy per episode.
No extra flag is needed. For diagnostics, add `--fresh-process` to start an
independent app per episode. Choose one mode, not both; fresh-process evaluation
is slower and is not an additional required step.

```bash
python tools/project.py gate --checkpoint "$CHECKPOINT" --fresh-process
```

For a holdout runtime problem, rerun only the holdout with `--fresh-process`
after confirming the gate passed for that same checkpoint. Both modes retain
the full seed sets, episode duration and scoring thresholds.

| Output | Default directory |
|---|---|
| Gate report and per-seed results | `isaac_sim/output/evaluation_candidate_gate/` |
| Holdout report and per-seed results | `isaac_sim/output/evaluation_candidate_holdout/` |
| Checkpoint-selection results | `isaac_sim/output/checkpoint_selection/` |

Each evaluation set writes `evaluation_report.json`, a combined
`session_runner.log`, and a folder per scenario containing `episode_summary.json`
and `observations.csv`. In fresh-process mode, each scenario instead has its
own `runner.log`. Gate and holdout also save perception diagnostics. Watch a
log from a second Terminal if a heartbeat continues without a DONE message;
the heartbeat means the parent is waiting, not proof that physics is advancing.

Before a new run starts, the previous final report is moved to `report_history/`.
If the run crashes or is interrupted, no old PASS report remains at
`evaluation_report.json`; a new final report is written atomically after the set finishes.

If the worker errors or stops before completing the set, the batch is invalid;
rerun that set with `--fresh-process` rather than exporting from partial results.
Reports record the execution mode. Rendered trajectories and scores are not
guaranteed to be bit-identical across runs or modes, despite identical seeds and
criteria. Confirm unexpected differences in fresh-process mode;
neither mode guarantees a `20/20` checkpoint.
`select-checkpoint` also keeps all validation seeds and starts one app per
checkpoint by default; `--fresh-process` is available for diagnostics.
The old explicit `--reuse-app` flag is still accepted, but is unnecessary.
Do not use holdout for checkpoint selection.

### 5.3 Watch the checkpoint to be exported

After evaluation, optionally watch the same checkpoint:

```bash
python tools/project.py play --checkpoint "$CHECKPOINT"
```

GUI is the default. **RobotCamera** shows the line from the robot; select
**TeachingOverviewCamera** in the viewport Camera menu for the overview.
This is inference only, with rendered-camera perception through the same
simulation path as the gate. It does not train, select, export, or overwrite
a model. One episode runs, then the GUI stays open, paused at the final state;
close it before continuing or rerun the command to replay. Add `--headless`
to run without the GUI. Preview diagnostics go to `isaac_sim/output/play/`,
not the gate/holdout folders.

`isaac_sim/scripts/play_policy_rl.py` is the lower-level Isaac Lab training-lane
player and ONNX exporter. It uses geometric perception rather than rendered
camera images, so use `project.py play` for this visible-track preview.
Keep the same `CHECKPOINT` for gate, holdout, preview, and export. If you select
a different model, resolve the path and evaluate it before preview/export.
A successful preview is not a substitute for evaluation or measured hardware
validation.

### Keep the evaluated checkpoint through export

Gate, holdout and preview do not overwrite the `.pt` file or change the selected
checkpoint. `select-checkpoint` records a path; it does not replace the last
training checkpoint. If you run selection again, resolve `CHECKPOINT` again and
evaluate the new choice before export.

If you already know which checkpoint to use, set `CHECKPOINT` to its actual `.pt`
path instead of searching all saved models, then run the gate on that path.
Do not resolve the path again between gate and export: a new training run,
selection, or manual choice could otherwise make those steps use different models.
If you open a new Terminal, activate `env_isaaclab`, return to the project root,
and restore the exact path printed by the completed gate before continuing.

## 6. Export and verify the selected checkpoint

Use the same `CHECKPOINT` evaluated in Section 5, not a newly resolved model:

```bash
python tools/project.py export-onnx --checkpoint "$CHECKPOINT" --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx

POLICY_VERSION="$(date +%Y%m%d_%H%M%S)_ppo_student"
python tools/project.py export-header --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx --version-name "$POLICY_VERSION"

python tools/project.py source-check
python tools/project.py deployed-smoke
```

`export-header` creates a versioned export under `firmware/policies/` and
deploys the same header, manifest and vectors to `firmware/generated/`, which
is the directory compiled by the sketch. Never edit generated weights by hand.

`deployed-smoke` checks that the generated C policy can be loaded and executed
before flashing the firmware.

The `.pt`, `.onnx`, versioned `firmware/policies/` export and active
`firmware/generated/` files remain local and are not committed to Git. Record
the policy ID and evaluation reports with the learner's results.

## 7. Diagnose a lower-scoring candidate

Run the failed seed with perception diagnostics:

```bash
python isaac_sim/scripts/run_line_following.py --headless --seed 11 --randomize --save-perception-debug --policy-backend rl --checkpoint "$CHECKPOINT"
```

Replace seed `11` with a failed validation seed when diagnosing a different
scenario, keeping the failed checkpoint path in `CHECKPOINT`.

Inspect `episode_summary.json`, `observations.csv`, the RGB frame and its
binary `_mask.png`. Classify the cause before changing code or configuration.
After a contract change, repeat parity, smoke and training, then gate; reserve
holdout for the final candidate instead of using it to tune the fix.

| Failure                                | Check first                                                |
| -------------------------------------- | ---------------------------------------------------------- |
| Tape missing or wrong`e_y` sign      | Camera mount, HFOV, orientation and perception             |
| Training lane differs from camera lane | Shared track/perception implementation                     |
| RPM differs from hardware              | Motor, encoder and randomization values in`default.json` |
| Reward exploit or wrong termination    | Reward and termination implementation                      |
| ABI, rate or action meaning changed    | Simulator, exporter and firmware together                  |

## 8. Optional reference actor

`firmware/reference/` contains a small public actor only for checking the
repository's inference and deployment contract. It is never copied into
`firmware/generated/` automatically and is not an input to BC or PPO.

```bash
python tools/project.py reference-smoke
```

Use this only as a troubleshooting comparison. A project result should use the
policy ID produced by the learner's own selected checkpoint and export.
