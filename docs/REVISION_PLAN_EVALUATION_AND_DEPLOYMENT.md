# Revision plan: evaluation scoring and deployment flow

## Goal

Change the project so model evaluation is a **quality score**, not a hard deployment gate.

A trained checkpoint may continue to ONNX export, C-header export, firmware testing, and robot deployment even when it scores below 20/20. A higher score is better, and 20/20 remains the best possible result on each 20-scenario evaluation set.

At the same time, fix the deployed-policy path bug so `--policy-backend deployed` reads the learner artifact from `firmware/generated/`, which is where `export_policy.py --deploy` actually writes it.

Do **not** add an Ubuntu-version block. Ubuntu 24.04 remains the validated reference platform, but newer Linux/Ubuntu releases may proceed when the dependency and `doctor` checks pass.

## Required changes

### 1. README.md

- Keep two installation paths:
  - existing NVIDIA Isaac Sim/Isaac Lab users start at Step 6;
  - new users follow Steps 1-5.
- Keep Ubuntu 24.04 as the validated reference platform, but do not make the exact Ubuntu version a hard requirement.
- Replace the hard 20/20 gate language in Steps 8-11:
  - Step 9 becomes a performance evaluation.
  - 20/20 is the maximum/best score.
  - 19/20 is stronger than 18/20, etc.
  - scores below 20/20 do not block export or deployment.
- If comparing several checkpoints, use seeds 0-19 as the validation set for checkpoint selection. Keep seeds 20-39 as holdout/reporting data after the checkpoint is selected.
- Replace “accepted checkpoint” with “selected checkpoint”.
- In Step 11, compare the exported C policy against the original checkpoint, but do not block firmware upload solely because the score is below 20/20.
- Update Troubleshooting so a score below 20/20 is described as a model-performance result rather than a software failure.

### 2. docs/TRAINING.md

- Remove the requirement that a learner must pass all 40 rendered-camera seeds before export.
- Describe validation and holdout as quantitative evaluation sets.
- Preserve the distinction between validation and holdout:
  - validation seeds 0-19 may be used to compare/select checkpoints;
  - holdout seeds 20-39 should be used after selection when the user wants an unbiased report.
- Allow either the latest checkpoint or a deliberately selected checkpoint to continue through export.
- Rename “accepted checkpoint” wording to “selected checkpoint”.
- Change “failed candidate” wording to “lower-scoring candidate” where appropriate.

### 3. isaac_sim/scripts/evaluate_line_following.py

Current behavior:
- prints `Simulation PASS=False randomized=19/20`;
- exits with status 1 whenever the policy does not achieve a perfect score;
- `tools/project.py` then shows a traceback even though evaluation itself completed correctly.

New behavior:
- always write `evaluation_report.json`;
- print an informational score such as:
  `Evaluation score: 19/20 randomized scenarios passed; nominal=PASS`;
- return exit status 0 when evaluation ran successfully, regardless of score;
- still return a real error/non-zero status for invalid arguments, missing files, crashes, or other runtime/software errors;
- keep the report’s existing fields for compatibility;
- `--tune` may retain its stricter full-pass behavior because it is a separate analytical-baseline tuning operation.

### 4. isaac_sim/scripts/run_line_following.py

Fix the deployed artifact mapping.

Current incorrect mapping:
- `reference` -> `firmware/reference/`
- `deployed` -> `firmware/deployed/`  (wrong; exporter does not write here)

Required mapping:
- `reference` -> `firmware/reference/`
- `deployed` -> `firmware/generated/`

Apply the same mapping both:
- when loading the header/manifest in `run_episode()`;
- when checking that the required files exist in `main()`.

This fixes the observed error:
`No learner policy is deployed...`
after `export-header` had already successfully written to `firmware/generated/`.

### 5. tools/project.py

Keep strict smoke checks for the simulator before training, but make **deployed-smoke** informational with respect to model performance.

- Extend `checked_camera_episode()` with a mode that still requires a valid summary file but does not raise an exception solely because the policy failed to complete the route.
- Use the strict/default mode for normal simulator smoke and reference smoke.
- Use the non-blocking scoring mode for `deployed-smoke`.
- A missing summary, renderer failure, malformed artifact, or other runtime problem must remain a real error.
- Change the `export-onnx` help text from “accepted checkpoint” to “selected checkpoint”.

### 6. isaac_sim/scripts/policy_header.py

Documentation-only wording:
- replace “learner’s accepted checkpoint” with “learner’s selected checkpoint”.

## Intended workflow after the revision

```text
train BC
  -> train PPO
  -> select latest checkpoint or another chosen checkpoint
  -> validation score on seeds 0-19
  -> optional checkpoint comparison/retraining
  -> holdout score on seeds 20-39
  -> export selected checkpoint to ONNX
  -> export C header to firmware/generated
  -> deployed smoke/evaluation scores
  -> upload firmware to ESP32-S3
```

Example:

```text
Validation: 19/20
Holdout:    20/20
```

This is a valid result and may continue to export/deployment. It is simply weaker on the validation set than a 20/20 model.

## Non-goals

- Do not require Ubuntu 24.04 exactly in code.
- Do not block Ubuntu 26.x or another recent Linux release solely because of its distro version.
- Do not weaken real software/runtime errors into scores.
- Do not remove validation or holdout reporting.
- Do not change the observation/action ABI or PPO network architecture.
- Do not rename the existing `gate` / `holdout` CLI commands in this revision; keep them for compatibility.
