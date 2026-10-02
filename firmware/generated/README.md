# Learner-generated policy

This directory is intentionally empty in Git. It is populated only after you
train, evaluate and export your own policy by following
[`docs/TRAINING.md`](../../docs/TRAINING.md).

The firmware compiles these local files:

```text
line_following_policy.h
line_following_policy_manifest.json
line_following_policy_vectors.csv
SELECTED.txt
```

Generate them with:

```bash
POLICY_VERSION="$(date +%Y%m%d_%H%M%S)_ppo_student"
python tools/project.py export-header \
  --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx \
  --version-name "$POLICY_VERSION"
```

Do not edit the header manually or commit these generated files. The optional
example in `firmware/reference/` is separate and is never selected by the
firmware automatically.
