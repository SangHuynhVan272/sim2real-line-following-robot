# Train and export a policy

This is the primary project workflow. It takes a learner from a validated
simulator to a new `.pt` checkpoint, an ONNX export and the C header compiled by
that learner's ESP32-S3 firmware.

Training is stochastic. Every run produces a new candidate; do not expect its
weights to match another person's run. Accept it only after it passes all 40
rendered-camera seeds. Nothing in `firmware/reference/` is loaded by the train,
evaluation, export or firmware commands below.

## 1. Keep the deployment contract unchanged

```text
observation = [e_y, e_theta_rad, line_confidence,
               rpm_left, rpm_right, duty_left_prev, duty_right_prev]
action      = [duty_left, duty_right]
```

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

```text
isaac_sim/output/rl/bc_candidate/model_bc.pt
isaac_sim/output/rl/bc_candidate/bc_summary.json
```

Behavior cloning fits the actor in pre-tanh space. Do not replace it with a
post-tanh fit; that previously produced saturated actors.

## 4. Train PPO from behavior cloning

```bash
python tools/project.py train-ppo --bc-run isaac_sim/output/rl/bc_candidate --output-dir isaac_sim/output/rl/ppo_candidate --iterations 600
```

The command uses seed 0 and a BC anchor weight of `0.2`. Choose one
checkpoint from training metrics before opening the rendered evaluation set.
Do not search for a checkpoint by repeatedly trying evaluation seeds.

## 5. Run both acceptance sets

`<N>` is the checkpoint iteration you selected. Pick it from the **task
metrics** in TensorBoard -- `Episode_Termination/finished` high and
`Episode_Termination/line_lost` low -- not from the reward curve, and not simply
the last one saved. Reward can keep climbing while the policy learns to hold
both wheels saturated, which may score well while completing fewer routes.

Selecting among checkpoints using seeds 0-19 is legitimate -- that is what the
gate is for. Seeds 20-39 must stay untouched until you have a candidate that
already passes 20/20, otherwise the holdout stops being evidence.

For the candidate you selected:

```bash
python tools/project.py gate --checkpoint isaac_sim/output/rl/ppo_candidate/model_<N>.pt
python tools/project.py holdout --checkpoint isaac_sim/output/rl/ppo_candidate/model_<N>.pt
```

Acceptance requires:

- nominal episode PASS;
- seeds 0-19: **20/20**;
- untouched seeds 20-39: **20/20**;
- no code, config or checkpoint change between the two sets.

`--tune` applies only to the analytical baseline. Never use it to make an RL
candidate pass, and never tune on seeds 20-39.

## 6. Export the accepted checkpoint

```bash
python tools/project.py export-onnx --checkpoint isaac_sim/output/rl/ppo_candidate/model_<N>.pt --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx
python tools/project.py export-header --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx --version-name YYYYMMDD_ppo_candidate
python tools/project.py source-check
python tools/project.py deployed-smoke
```

`export-header` creates a versioned export under `firmware/policies/` and
deploys the same header, manifest and vectors to `firmware/generated/`, which
is the directory compiled by the sketch. Never edit generated weights by hand.

`deployed-smoke` loads the arrays back from `firmware/generated/`, checks their
fingerprints and runs nominal plus randomized camera episodes. Before flashing
a final result, repeat the full rendered sets against the exported header:

```bash
python tools/project.py deployed-gate
python tools/project.py deployed-holdout
```

The `.pt`, `.onnx`, versioned `firmware/policies/` export and active
`firmware/generated/` files remain local and are not committed to Git. Record
the policy ID and evaluation reports with the learner's results.

## 7. Diagnose a failed candidate

Run the failed seed with perception diagnostics:

```text
python isaac_sim/scripts/run_line_following.py --headless --seed 11 --randomize --save-perception-debug --policy-backend rl --checkpoint isaac_sim/output/rl/ppo_candidate/model_<N>.pt
```

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
policy ID produced by the learner's own accepted checkpoint and export.
