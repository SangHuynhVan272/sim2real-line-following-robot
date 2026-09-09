# Public release checklist

Run this checklist from the repository root before creating a GitHub release.
It is intentionally separate from simulation acceptance: a policy can pass in
simulation while the repository still contains a private path, stale artifact
or incomplete deployment guide.

## 1. Legal and repository hygiene

- Confirm `LICENSE` and `THIRD_PARTY_NOTICES.md` are present and current.
- Confirm every CAD mesh, reference policy and third-party snippet may be
  redistributed under that license.
- Run `python tools/project.py release-check`.
- Review `git status --short` and `git ls-files`; do not publish checkpoints,
  ONNX files, USD scenes, logs, debug images, local environment files or Python
  caches.
- Inspect the commits that will be pushed, not only the current working tree.
  Removing text in a later commit does not remove it from Git history.

## 2. Reproduce the public workflow

On a fresh clone or clean machine:

```bash
conda activate env_isaaclab
python tools/project.py doctor
python tools/project.py import-robot
python tools/project.py parity
python tools/project.py smoke
```

The optional public fixture may be checked separately with `reference-smoke`.
It must not appear in `firmware/generated/` or become a substitute for a
learner result.

If the policy or its contract changed, follow `docs/TRAINING.md` from behavior
cloning through both rendered evaluation sets and export. Then rerun:

```bash
python tools/project.py source-check
python tools/project.py deployed-smoke
```

For a promoted learner policy, also run `deployed-gate` and
`deployed-holdout`. They execute the actor exported to the local C header, not
the `.pt` checkpoint. Do not commit that learner-specific header to the public
teaching repository.

## 3. Verify firmware deployment

- Confirm `firmware/generated/line_following_policy.h` was produced from the
  learner's accepted checkpoint. A fresh clone intentionally has no active
  policy and should not be compiled for deployment before this step.
- Open `firmware/esp32s3_line_following/esp32s3_line_following.ino` with the
  documented Arduino-ESP32 version and compile for the target ESP32-S3 board.
- For a repeatable command-line compile, install Arduino CLI and run:

  ```bash
  arduino-cli core update-index --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
  arduino-cli core install esp32:esp32@3.3.11 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
  arduino-cli compile --fqbn esp32:esp32:esp32s3 firmware/esp32s3_line_following
  ```

- Flash with motor power disconnected or the robot physically restrained.
- Confirm the serial policy/config IDs match `firmware/generated/`.
- Verify camera orientation, encoder signs, both motors on each side,
  line-loss stop and stall protection before an autonomous floor test.

Record the operating system, GPU, driver, exact commands and resulting policy
ID in the GitHub release notes. Do not claim real-robot readiness from
simulation evidence alone.
