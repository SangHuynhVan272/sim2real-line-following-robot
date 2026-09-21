# Sim2Real Line-Following Robot

An end-to-end reinforcement-learning project for a four-motor skid-steer robot
with an OV2640 camera, two wheel encoders and an ESP32-S3.

![Dual-view Isaac Sim line-following demo](docs/media/simulation_rollout.gif)

*A short dual-view Isaac Sim replay: rendered `RobotCamera` frames on the left
and the corresponding track progress on the right. This is a visualization from
simulator data, not policy-validation evidence or footage of the physical robot.*

The intended learner workflow is:

```text
validate simulator -> train BC -> train PPO -> select .pt checkpoint
       -> rendered gate + untouched holdout -> export .onnx
       -> generate firmware/generated/line_following_policy.h -> ESP32-S3
```

The policy receives seven values and directly commands left/right PWM duty:

```text
[e_y, e_theta, line confidence, left/right RPM, previous left/right duty]
                              -> MLP 7 -> 64 -> 64 -> 2
```

There is no wheel-speed PID in the simulator or firmware.

## 1. What is included

| Path                                  | Purpose                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `isaac_sim/config/default.json`     | Robot, camera, motor, randomization and training settings    |
| `isaac_sim/envs/line_following_rl/` | Vectorized Isaac Lab training environment                    |
| `isaac_sim/scripts/`                | Import, simulation, training, evaluation and export programs |
| `linefollowingrobot_cad/`           | URDF and STL files used to create the Isaac Sim robot        |
| `firmware/esp32s3_line_following/`  | Arduino ESP32-S3 firmware                                    |
| `firmware/generated/`               | Local output for the learner's exported policy; ignored by Git |
| `firmware/reference/`               | Optional example actor and golden vectors; never selected automatically |
| `firmware/tests/`                   | Two small host checks for perception and policy parity       |
| `tools/project.py`                  | Project command runner and environment checks                |
| `docs/TRAINING.md`                  | Complete retraining and export procedure                     |

This repository teaches you to produce and deploy **your own policy**. Training
checkpoints (`.pt`), ONNX exports and the active files in `firmware/generated/`
are local outputs and are not committed to Git. After training, the exporter
copies your accepted actor into `firmware/generated/`; that is the only policy
the ESP32-S3 sketch includes.

`firmware/reference/` contains one optional example actor so a learner can
verify the inference ABI and simulator before a long training run. It is a test
fixture, not a required starting checkpoint, not a claimed student result and
not a firmware fallback. A fresh clone intentionally has no active firmware
policy until the learner completes the export step.

Simulation PASS is not a Real Robot PASS claim. Every physical robot still
needs camera, encoder and motor-direction checks before autonomous driving.

## 2. Validated platform

This project is developed and validated on Ubuntu 24.04 LTS (x86-64).
Use a PC meeting NVIDIA's minimum specification:

- Ubuntu 24.04 LTS;
- at least 4 CPU cores, 32 GB RAM and 50 GB free SSD space;
- an NVIDIA RTX GPU with 16 GB VRAM or more;
- NVIDIA driver 580.65.06 or newer;
- a stable internet connection for the first Isaac Sim launch.

Windows, WSL, macOS and Ubuntu 22.04 have not been validated for this
repository. They may work, but they are outside the currently supported and
tested workflow.

NVIDIA publishes the current GPU, VRAM and driver requirements in the
[Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
Run `nvidia-smi` before installing. If it fails, repair the NVIDIA driver
before continuing; do not install a random CUDA Toolkit to work around a
driver problem.

This project uses exactly:

| Component | Version |
|---|---:|
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.2 |
| PyTorch / torchvision / torchaudio | 2.7.0 / 0.22.0 / 2.7.0, CUDA 12.8 build |
| RSL-RL | 3.1.2 |
| NumPy | 1.26.0 |
| ONNX | 1.20.1 |
| Pillow | 11.3.0 |
| click / psutil / typing_extensions | 8.1.7 / 5.9.8 / 4.12.2 |
| Arduino-ESP32 | 3.3.11 |

These are compatibility pins, not suggestions to install the newest release.
NVIDIA now classifies Isaac Sim 5.1 as an older unsupported release; moving to
Isaac Sim 6.x or Isaac Lab 3.x is a migration that this repository has not
validated.

Do not use the current Isaac Lab `main` branch or Python 3.12. Those versions
use different APIs and will produce import or RSL-RL configuration errors.

## 3. Install on Ubuntu 24.04 LTS

### 3.1 Install system tools and Miniconda

Open Terminal:

```bash
sudo apt update
sudo apt install -y git wget cmake build-essential
mkdir -p "$HOME/robotics"
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
```

Close Terminal and open it again. Verify:

```bash
git --version
conda --version
nvidia-smi
```

Isaac Sim pip packages require GLIBC 2.35 or newer. Ubuntu 24.04 meets that
requirement; confirm with:

```bash
ldd --version
```

### 3.2 Create the Python environment

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
```

The prompt must begin with `(env_isaaclab)`.

### 3.3 Install Isaac Sim and PyTorch

```bash
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install --upgrade --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

Start Isaac Sim once and accept the EULA:

```bash
isaacsim
```

Wait for the first extension download and cache build to finish, then close
Isaac Sim.

### 3.4 Install Isaac Lab

```bash
cd "$HOME/robotics"
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh --install rsl_rl
```

Re-apply the project package versions:

```bash
python -m pip install --upgrade --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.0 onnx==1.20.1 pillow==11.3.0 click==8.1.7 psutil==5.9.8 typing_extensions==4.12.2
```

Verify Isaac Lab from `$HOME/robotics/IsaacLab`:

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

A black Isaac Sim viewport should open. Stop it with `Ctrl+C`.

### 3.5 Download this project

Run the following commands to download the project:

```bash
cd "$HOME/robotics"
git clone https://github.com/SangHuynhVan272/sim2real-line-following-robot.git line-following-robot-pai
cd line-following-robot-pai
python tools/project.py doctor
```

For a ZIP download, extract it to
`$HOME/robotics/line-following-robot-pai`, activate `env_isaaclab`, enter that
directory and run the same `doctor` command.

Continue only when there is no `[FAIL]` and the last line says
`Environment check PASS`. A `[SKIP]` for an optional check is normal.

## 4. Run the project on Ubuntu

Every time you open a new terminal:

```text
1. conda activate env_isaaclab
2. enter the line-following-robot-pai directory
3. run python tools/project.py doctor
```

The project runner finds the repository from its own file location, so its
paths do not depend on your username or installation directory.

### 4.1 Check the downloaded source

```bash
python tools/project.py source-check
```

This also compiles and runs the C++ policy/perception tests when `g++` is
installed. Firmware compilation is checked separately in Arduino IDE.

### 4.2 Import the robot and compare both simulation implementations

```bash
python tools/project.py import-robot
python tools/project.py parity
```

`import-robot` creates local URDF/USD output under `isaac_sim/assets/`.
`parity` compares 1,500 samples between the rendered and vectorized lanes and
must end with `PARITY OK`.

### 4.3 Run the first camera simulation

```bash
python tools/project.py smoke
```

`smoke` runs one nominal and one randomized camera episode using the analytical
baseline. This checks the camera, perception, physics and motor model before
training.

### 4.4 Optional reference-policy check

```bash
python tools/project.py reference-smoke
```

This optional check reads the example arrays from
`firmware/reference/line_following_policy.h`. It confirms that Python and the
firmware header share one inference contract; it does not train or select a
policy for the learner.

To repeat the complete acceptance sets for that optional reference fixture:

```bash
python tools/project.py reference-gate
python tools/project.py reference-holdout
```

Each command also runs a nominal episode. The gate covers randomized seeds
0-19 and the untouched holdout covers seeds 20-39. These rendered-camera runs
take substantially longer than the two-episode smoke check.

Open the interactive scene with:

```bash
python tools/project.py gui
```

Use `RobotCamera` for the perception image and `TeachingOverviewCamera` for
the whole track.

## 5. Train and export your policy

Training is the main project activity. The reference actor is not used by these
commands:

```bash
python tools/project.py train-bc --output-dir isaac_sim/output/rl/bc_candidate
python tools/project.py train-ppo --bc-run isaac_sim/output/rl/bc_candidate --output-dir isaac_sim/output/rl/ppo_candidate --iterations 600
```

A finished job is only a candidate. It must pass seeds 0-19 and the untouched
20-39 holdout before export. After selecting checkpoint `<N>` from the training
metrics, the remaining pipeline is:

```bash
python tools/project.py gate --checkpoint isaac_sim/output/rl/ppo_candidate/model_<N>.pt
python tools/project.py holdout --checkpoint isaac_sim/output/rl/ppo_candidate/model_<N>.pt
python tools/project.py export-onnx --checkpoint isaac_sim/output/rl/ppo_candidate/model_<N>.pt --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx
python tools/project.py export-header --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx --version-name YYYYMMDD_ppo_candidate
python tools/project.py deployed-smoke
```

Read [`docs/TRAINING.md`](docs/TRAINING.md) before running these commands; it
defines checkpoint selection, the untouched holdout rule and the full
post-export validation.

## 6. Build and flash the ESP32-S3

Complete Section 5 first. Confirm that your export created
`firmware/generated/line_following_policy.h`; the firmware deliberately refuses
to compile without a learner-generated header. Then:

1. Install [Arduino IDE 2](https://www.arduino.cc/en/software).
2. Open Arduino IDE Preferences and add this Board Manager URL:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`.
3. Open Boards Manager, search for `esp32 by Espressif Systems`, select version
   **3.3.11**, and install it.
4. Select an ESP32-S3 board matching the actual module. For a custom board,
   start with `ESP32S3 Dev Module` and configure its flash/PSRAM from the board
   specification.
5. Open
   `firmware/esp32s3_line_following/esp32s3_line_following.ino`.
6. Select the USB/serial port, click Verify, then Upload.
7. Open Serial Monitor at 115200 baud and record the policy and config IDs. They
   must match your `firmware/generated/line_following_policy_manifest.json`.

The sketch was compile-tested with Arduino-ESP32 3.3.11 and the
`ESP32S3 Dev Module` board profile.

The sketch targets this wiring contract:

| Function | Pins / setting |
|---|---|
| OV2640 clock/control | XCLK = GPIO 15, SIOD/SDA = 4, SIOC/SCL = 5, VSYNC = 6, HREF = 7, PCLK = 13; PWDN/reset unused |
| OV2640 data D0-D7 | GPIO 11, 9, 8, 10, 12, 18, 17, 16; QVGA grayscale; horizontal mirror and vertical flip enabled |
| Right front encoder | A = GPIO 1, B = GPIO 2 |
| Left front encoder | A = GPIO 3, B = GPIO 46 |
| Right-side motor driver | direction = GPIO 41, PWM = GPIO 42 |
| Left-side motor driver | direction = GPIO 48, PWM = GPIO 47 |
| Encoder scale | 1,400 counts per wheel revolution |
| Serial monitor | 115200 baud |

Both motors on one side receive the same side command; only the front encoder
on each side supplies policy RPM. A different board, camera ribbon, motor
driver or encoder resolution requires matching changes in the sketch and
`default.json`, followed by retraining and export. Do not power the motors from
the USB port.

For the first motor test, disconnect motor power or physically restrain the
robot. Confirm camera direction, encoder sign, all four motor directions,
line-loss stop and stall protection before placing it on the floor.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `python` points outside `env_isaaclab` | Run `conda activate env_isaaclab`, then rerun `python tools/project.py doctor` |
| Python is 3.10 or 3.12 | Recreate the conda environment with Python 3.11 |
| `isaacsim` is not recognized | The conda environment is inactive or Isaac Sim installation failed |
| `ModuleNotFoundError: isaaclab` | From Isaac Lab v2.3.2, rerun `./isaaclab.sh --install rsl_rl` |
| RSL-RL reports `class_name`, `actor` or `critic` | Checkout Isaac Lab v2.3.2 and reinstall `rsl_rl` |
| `torch.cuda.is_available()` is false | Repair the NVIDIA driver, then reinstall the cu128 PyTorch command |
| First Isaac Sim launch appears frozen | Wait while extensions and shaders download; this can exceed ten minutes |
| Kit prints `Failed to create change watch ... No space left on device` although disk space is available | Close applications that consume many file watches; on Linux, increase the per-user inotify watch limit, then restart Isaac Sim |
| Robot USD is missing | Run `python tools/project.py import-robot` before smoke/gui |
| `PARITY` fails | Restore the unchanged source/config pair and rerun parity before training |
| PPO completes but evaluation fails | Diagnose the failed rendered seed; do not weaken the acceptance gate |
| Arduino cannot find `esp_camera.h` | Install/select Espressif's ESP32 board package |
| Robot steers the wrong way | Verify camera mirror/flip and the sign of `e_y` before enabling motors |

When asking for help, include the complete output of:

```bash
python tools/project.py doctor
python tools/project.py source-check --skip-cpp
```

Also include the operating system, GPU model, NVIDIA driver version and the
first error line. Do not send the entire Isaac Sim log unless requested.

## 8. License

The original project source, documentation, CAD and generated policy artifacts
are released under the [BSD 3-Clause License](LICENSE). Third-party frameworks,
tools and libraries remain under their respective terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Before pushing a release, also run the public-package checklist in
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).
