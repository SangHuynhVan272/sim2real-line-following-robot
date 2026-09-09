# Optional reference policy

This directory contains one previously trained policy only as a reproducible
example and test fixture. It is **not** the policy learners are expected to
deploy, and the firmware does not include it automatically.

The three files travel together:

- `line_following_policy.h` contains the frozen MLP weights;
- `line_following_policy_manifest.json` records the ABI and fingerprints;
- `line_following_policy_vectors.csv` verifies matching Python and C++ output.

Run `python tools/project.py reference-smoke` if you want to confirm that the
simulator can execute this example before spending time on training. For the
actual project, follow [`docs/TRAINING.md`](../../docs/TRAINING.md), export your
own actor to `firmware/generated/`, then run
`python tools/project.py deployed-smoke`.

Do not copy this header to `firmware/generated/` for a claimed student result.
Results should identify the checkpoint and policy ID produced by that student's
own training run.
