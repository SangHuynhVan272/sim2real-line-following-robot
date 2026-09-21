# Sim2Real Line-Following Robot

This project takes you from training a line-following policy in Isaac Lab to
deploying that policy on an ESP32-S3 robot with Arduino IDE.

![Dual-view Isaac Sim line-following demo](docs/media/simulation_rollout.gif)

You do not need a pretrained model. By following this README from top to
bottom, you will create:

| Output | Purpose |
|---|---|
| `model_*.pt` | PPO model trained in Isaac Lab |
| `policy.onnx` | Intermediate model used for firmware export |
| `line_following_policy.h` | C policy compiled into the ESP32-S3 firmware |

The complete workflow is:

```text
install -> validate simulation -> train BC -> train PPO
-> test 40 scenarios -> export policy -> Arduino Upload -> run the robot
```

## 1. Before you begin

This guide assumes that you are using a new computer and that the robot has
already been assembled, wired, and configured for this project.

Your computer needs:

- Ubuntu 24.04 LTS 64-bit;
- an NVIDIA RTX GPU with at least 16 GB VRAM;
- NVIDIA driver 580.65.06 or newer;
- at least 32 GB RAM and 50 GB of free SSD space;
- a stable Internet connection for the first installation.

This workflow has not been validated on Windows, WSL, macOS, or Ubuntu 22.04.

### How to use the command blocks

1. Open Terminal with `Ctrl + Alt + T`.
2. Copy and paste **one command block at a time**.
3. Press `Enter` and wait for the block to finish before continuing.
4. Do not copy a leading `$` from examples found elsewhere.
5. Stop if you see `[FAIL]`, `Error`, or `Traceback`, or if the command never
   returns to the terminal prompt. Do not continue by ignoring an error.

Check the GPU first:

```bash
nvidia-smi
```

The command must show the GPU name, driver version, and VRAM. If it fails or is
not found, install or repair the NVIDIA driver before continuing.

## 2. Install system tools and Miniconda

Run:

```bash
sudo apt update
sudo apt install -y git wget cmake build-essential
mkdir -p "$HOME/robotics"
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
```

Close Terminal, open a new Terminal, and verify the installation:

```bash
git --version
conda --version
nvidia-smi
```

All three commands must run successfully.

## 3. Create the Python environment

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
```

After `conda activate`, the terminal prompt must begin with:

```text
(env_isaaclab)
```

## 4. Install Isaac Sim 5.1 and PyTorch

```bash
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install --upgrade --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

Start Isaac Sim once:

```bash
isaacsim
```

Accept the NVIDIA EULA if prompted. The first launch may take more than ten
minutes while extensions are downloaded and the shader cache is built. When
Isaac Sim has opened completely, close it and return to Terminal.

## 5. Install Isaac Lab

```bash
cd "$HOME/robotics"
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh --install rsl_rl
```

Reinstall the package versions required by this project:

```bash
python -m pip install --upgrade --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.0 onnx==1.20.1 pillow==11.3.0 click==8.1.7 psutil==5.9.8 typing_extensions==4.12.2
```

Verify Isaac Lab:

```bash
cd "$HOME/robotics/IsaacLab"
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

An empty Isaac Sim window should open. Press `Ctrl + C` in Terminal to stop the
program, and then close the Isaac Sim window.

## 6. Download the robot project

```bash
cd "$HOME/robotics"
git clone https://github.com/SangHuynhVan272/sim2real-line-following-robot.git line-following-robot-pai
cd line-following-robot-pai
```

Run every remaining command from the `line-following-robot-pai` directory.

Whenever you open a new Terminal, run these three commands first:

```bash
conda activate env_isaaclab
cd "$HOME/robotics/line-following-robot-pai"
python tools/project.py doctor
```

Continue only when the final line is:

```text
Environment check PASS. This terminal is ready for the project commands.
```

An optional `[SKIP]` is normal. A `[FAIL]` is not normal and must be fixed
before you continue.

## 7. Validate the source and simulation

Check the downloaded project files:

```bash
python tools/project.py source-check
```

The output must include:

```text
[PASS] Repository files, JSON, Markdown links and Python syntax
```

Generate the robot USD and compare the two simulation implementations:

```bash
python tools/project.py import-robot
python tools/project.py parity
```

The `parity` command must end with:

```text
PARITY OK
```

Run two camera episodes before training:

```bash
python tools/project.py smoke
```

Both episodes must print `[PASS] Camera episode`.

To watch the robot in Isaac Sim before training, run:

```bash
python tools/project.py gui
```

Select `RobotCamera` to see the image used by the controller, or select
`TeachingOverviewCamera` to see the complete track. Press `Ctrl + C` in
Terminal when you want to close the simulation.

## 8. Train the model

Training has two stages. Behavior Cloning creates a reliable starting point,
and PPO then improves the policy in randomized environments.

### 8.1 Train the Behavior Cloning warm start

```bash
python tools/project.py train-bc --output-dir isaac_sim/output/rl/bc_candidate
```

Do not close Terminal while this command is running. When it finishes, the
following file must exist:

```text
isaac_sim/output/rl/bc_candidate/model_bc.pt
```

Check it with:

```bash
test -f isaac_sim/output/rl/bc_candidate/model_bc.pt && echo "BC MODEL OK"
```

Continue only when `BC MODEL OK` is printed.

### 8.2 Train PPO

```bash
python tools/project.py train-ppo --bc-run isaac_sim/output/rl/bc_candidate --output-dir isaac_sim/output/rl/ppo_candidate --iterations 600
```

Training is complete when Terminal prints:

```text
training complete; checkpoints in ...
```

This recipe uses the newest saved checkpoint as its candidate. Find it
automatically with:

```bash
CHECKPOINT="$(find isaac_sim/output/rl/ppo_candidate -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
if [ -z "$CHECKPOINT" ]; then
  echo "ERROR: no PPO checkpoint was found"
else
  echo "PPO candidate: $CHECKPOINT"
fi
```

You must see a path ending in `.pt`. This is only a candidate model and must
not be deployed yet.

## 9. Test the trained model

First, run the gate on 20 randomized scenarios using seeds 0 through 19:

```bash
CHECKPOINT="$(find isaac_sim/output/rl/ppo_candidate -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
if [ -z "$CHECKPOINT" ]; then
  echo "ERROR: no PPO checkpoint was found"
else
  python tools/project.py gate --checkpoint "$CHECKPOINT"
fi
```

Continue only when the final line is:

```text
Simulation PASS=True randomized=20/20
```

Next, run the holdout on 20 untouched scenarios using seeds 20 through 39:

```bash
CHECKPOINT="$(find isaac_sim/output/rl/ppo_candidate -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
if [ -z "$CHECKPOINT" ]; then
  echo "ERROR: no PPO checkpoint was found"
else
  python tools/project.py holdout --checkpoint "$CHECKPOINT"
fi
```

Continue only when the final line is also:

```text
Simulation PASS=True randomized=20/20
```

If either command does not pass 20/20, stop here. Do not export or deploy that
model.

## 10. Convert the model into an ESP32-S3 policy

Export the accepted checkpoint to ONNX:

```bash
CHECKPOINT="$(find isaac_sim/output/rl/ppo_candidate -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
if [ -z "$CHECKPOINT" ]; then
  echo "ERROR: no PPO checkpoint was found"
else
  python tools/project.py export-onnx --checkpoint "$CHECKPOINT" --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx
fi
```

Check the ONNX file:

```bash
test -f isaac_sim/output/rl/ppo_candidate/policy.onnx && echo "ONNX OK"
```

Continue only when `ONNX OK` is printed.

Convert the ONNX model into a C header and place it in the directory used by
the Arduino firmware:

```bash
POLICY_VERSION="$(date +%Y%m%d_%H%M%S)_ppo_student"
python tools/project.py export-header --onnx isaac_sim/output/rl/ppo_candidate/policy.onnx --version-name "$POLICY_VERSION"
```

Check the firmware policy:

```bash
test -f firmware/generated/line_following_policy.h && echo "FIRMWARE POLICY OK"
```

Continue only when `FIRMWARE POLICY OK` is printed.

Do not copy the `.pt` or `.onnx` file into Arduino. Arduino uses only the C
header generated at:

```text
firmware/generated/line_following_policy.h
```

## 11. Test the exact policy that will be uploaded

Run a short test using the generated C header:

```bash
python tools/project.py deployed-smoke
```

Both episodes must print `[PASS] Camera episode`.

Run all 40 scenarios again using the generated C header:

```bash
python tools/project.py deployed-gate
python tools/project.py deployed-holdout
```

Each command must end with:

```text
Simulation PASS=True randomized=20/20
```

If either test fails, do not upload the firmware. The generated C policy must
pass the same tests as the `.pt` model.

## 12. Upload the policy with Arduino IDE

### 12.1 Install Arduino IDE and ESP32 support

1. Download and install [Arduino IDE 2](https://www.arduino.cc/en/software).
2. Open **File > Preferences**.
3. Add this URL to **Additional Boards Manager URLs**:

   ```text
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```

4. Open **Tools > Board > Boards Manager**.
5. Search for `esp32 by Espressif Systems`.
6. Select version **3.3.11** and click **Install**.

### 12.2 Open the firmware

In Arduino IDE, select **File > Open** and open:

```text
$HOME/robotics/line-following-robot-pai/firmware/esp32s3_line_following/esp32s3_line_following.ino
```

Arduino IDE will compile this sketch together with the policy in
`firmware/generated/`.

### 12.3 Select the board and upload

1. Keep the robot battery power switched off.
2. Connect the robot to the computer with its USB data cable.
3. Select **Tools > Board > esp32 > ESP32S3 Dev Module**.
4. Select the robot USB port under **Tools > Port**.
5. Click **Upload** and wait for `Done uploading`.

The Serial Monitor is optional. If you want to inspect the loaded policy before
disconnecting USB, open **Tools > Serial Monitor**, select `115200 baud`, and
look for a line in this form:

```text
policy_id=... config_id=... mirror=... flip=...
```

Serial Monitor is not required for normal operation.

After the upload:

1. Close Serial Monitor if it is open.
2. Disconnect the USB cable from the robot.
3. Place the robot correctly over the line.
4. Switch on the robot battery power.

The firmware waits for three valid camera frames containing the line and then
starts the trained policy automatically.

## 13. Your generated result files

```text
isaac_sim/output/rl/bc_candidate/model_bc.pt
isaac_sim/output/rl/ppo_candidate/model_*.pt
isaac_sim/output/rl/ppo_candidate/policy.onnx
firmware/policies/<policy-version>/
firmware/generated/line_following_policy.h
firmware/generated/line_following_policy_manifest.json
firmware/generated/line_following_policy_vectors.csv
```

These files are generated locally and are not uploaded to Git automatically.
Never edit the weights in `line_following_policy.h` by hand.

## 14. Troubleshooting

| Symptom | Action |
|---|---|
| The prompt does not show `(env_isaaclab)` | Run `conda activate env_isaaclab` |
| `python: command not found` | Activate `env_isaaclab`, then try again |
| `ModuleNotFoundError: isaaclab` | Enter `$HOME/robotics/IsaacLab` and rerun `./isaaclab.sh --install rsl_rl` |
| `torch.cuda.is_available()` is `false` | Repair the NVIDIA driver and reinstall the cu128 PyTorch packages from Step 4 |
| The first Isaac Sim launch takes a long time | Wait for extensions and shaders to finish downloading |
| `No space left on device` | Check free SSD space and close applications using many file watchers |
| `PARITY` does not end with `PARITY OK` | Do not train; restore the correct source and configuration first |
| Gate or holdout does not pass 20/20 | Do not export the model; save the output and contact the instructor |
| Arduino cannot find `line_following_policy.h` | Complete Step 10 before verifying the sketch |
| Arduino cannot find `esp_camera.h` | Install ESP32 by Espressif Systems 3.3.11 and select the ESP32-S3 board |
| The USB port is not listed | Use a USB data cable, try another port, or correct Ubuntu serial-port permissions |

When requesting help, run the following commands and provide their complete
output together with the first error line:

```bash
conda activate env_isaaclab
cd "$HOME/robotics/line-following-robot-pai"
python tools/project.py doctor
python tools/project.py source-check --skip-cpp
```

## 15. License

The project source, documentation, CAD files, and generated policy artifacts
are released under the [BSD 3-Clause License](LICENSE). Third-party frameworks
and packages retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
