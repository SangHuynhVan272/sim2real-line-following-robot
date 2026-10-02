# Train and export a policy

> [!NOTE]
> Copyable code blocks in this guide are commands intended for Terminal. Status
> messages, example results, interface definitions, and file paths are shown as
> normal text, tables, lists, or notes.

This is the primary project workflow. It takes a learner from a validated
simulator to a new `.pt` checkpoint, an ONNX export and the C header compiled by
that learner's ESP32-S3 firmware.

Training is stochastic. Every run produces a new candidate; do not expect its
weights to match another person's run. The rendered-camera evaluations quantify
how robust the selected checkpoint is. A perfect 20/20 score on each set is the
best result, but a lower score is still a valid measured outcome and does not
block export or deployment. Nothing in `firmware/reference/` is loaded by the
train, evaluation, export or firmware commands below.

## 1. Keep the deployment contract unchanged

| Interface | Values |
|---|---|
| Observation | `[e_y, e_theta_rad, line_confidence, rpm_left, rpm_right, duty_left_prev, duty_right_prev]` |
| Action | `[duty_left, duty_right]` |

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

This creates:

- `isaac_sim/output/rl/bc_candidate/model_bc.pt`
- `isaac_sim/output/rl/bc_candidate/bc_summary.json`

Behavior cloning fits the actor in pre-tanh space. Do not replace it with a
post-tanh fit; that previously produced saturated actors.

## 4. Train PPO from behavior cloning

```bash
python tools/project.py train-ppo --bc-run isaac_sim/output/rl/bc_candidate --output-dir isaac_sim/output/rl/ppo_candidate --iterations 600
```

The command uses seed 0 and a BC anchor weight of `0.2`.

For the standard teaching workflow, no manual checkpoint filename is needed.
The next step automatically uses the newest saved checkpoint, regardless of the
iteration number.

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

### 5.1 Deployment gate: seeds 0-19

Run the validation gate:

```bash
python tools/project.py gate --checkpoint "$CHECKPOINT"
```

For the teaching workflow, the gate passes only when the nominal episode passes
and at least 19 of the 20 randomized validation scenarios pass:

- `20/20` + `nominal=PASS`: target result.
- `19/20` + `nominal=PASS`: accepted teaching gate PASS.
- `18/20` or lower, or `nominal=FAIL`: gate not passed.

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
CHECKPOINT="$(python tools/project.py checkpoint-path)"
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

## 6. Export and verify the selected checkpoint

```bash
CHECKPOINT="$(python tools/project.py checkpoint-path)"
python tools/project.py export-onnx --checkpoint "$CHECKPOINT" --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx

POLICY_VERSION="$(date +%Y%m%d_%H%M%S)_ppo_student"
python tools/project.py export-header --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx --version-name "$POLICY_VERSION"

python tools/project.py source-check
python tools/project.py deployed-smoke
python tools/project.py deployed-gate
python tools/project.py deployed-holdout
```

`export-header` creates a versioned export under `firmware/policies/` and
deploys the same header, manifest and vectors to `firmware/generated/`, which
is the directory compiled by the sketch. Never edit generated weights by hand.

`deployed-smoke` checks that the generated C policy can be loaded and executed.
`deployed-gate` then applies the same teaching deployment rule as Section 5.1:
`20/20` is the target and `19/20` with `nominal=PASS` is accepted. If the
deployed gate does not pass, do not flash the firmware. `deployed-holdout` is
recorded as a robustness score and does not block deployment.

The `.pt`, `.onnx`, versioned `firmware/policies/` export and active
`firmware/generated/` files remain local and are not committed to Git. Record
the policy ID and evaluation reports with the learner's results.

## 7. Diagnose a lower-scoring candidate

Run the failed seed with perception diagnostics:

```bash
CHECKPOINT="$(python tools/project.py checkpoint-path)"
python isaac_sim/scripts/run_line_following.py --headless --seed 11 --randomize --save-perception-debug --policy-backend rl --checkpoint "$CHECKPOINT"
```

Replace seed `11` with a failed validation seed when diagnosing a different
scenario.

Inspect `episode_summary.json`, `observations.csv`, the RGB frame and its
binary `_mask.png`. Change the classified cause, then repeat parity, smoke,
gate and holdout.

| Failure | Check first |
|---|---|
| Tape missing or wrong `e_y` sign | Camera mount, HFOV, orientation and perception |
| Training lane differs from camera lane | Shared track/perception implementation |
| RPM differs from hardware | Motor, encoder and randomization values in `default.json` |
| Reward exploit or wrong termination | Reward and termination implementation |
| ABI, rate or action meaning changed | Simulator, exporter and firmware together |

## 8. Optional reference actor

`firmware/reference/` contains a small public actor only for checking the
repository's inference and deployment contract. It is never copied into
`firmware/generated/` automatically and is not an input to BC or PPO.

```bash
python tools/project.py reference-smoke
```

Use this only as a troubleshooting comparison. A project result should use the
policy ID produced by the learner's own selected checkpoint and export.
