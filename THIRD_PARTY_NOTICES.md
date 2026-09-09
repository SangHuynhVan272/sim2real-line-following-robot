# Third-party notices

The repository's original source code, documentation, CAD files and generated
policy artifacts are licensed under the BSD 3-Clause License in `LICENSE`.
Third-party software is not relicensed by that file.

## Isaac Lab

The environment and training integration use Isaac Lab v2.3.2. The following
files retain the upstream copyright and BSD-3-Clause identifier because their
manager/configuration or launcher structure is based on Isaac Lab examples and
templates:

- `isaac_sim/envs/line_following_rl/env_cfg.py`
- `isaac_sim/envs/line_following_rl/agents/rsl_rl_ppo_cfg.py`
- `isaac_sim/scripts/train_policy_rl.py`
- `isaac_sim/scripts/play_policy_rl.py`

Isaac Lab is copyright its contributors and is distributed under the
[BSD 3-Clause License](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/LICENSE).

## RSL-RL

RSL-RL v3.1.2 is installed separately and is not copied into this repository.
It is copyright ETH Zurich and NVIDIA Corporation & Affiliates and is
distributed under the
[BSD 3-Clause License](https://github.com/leggedrobotics/rsl_rl/blob/v3.1.2/LICENSE).

## Isaac Sim

Isaac Sim and Omniverse Kit are installed separately and are not distributed
by this repository. Users must obtain them from NVIDIA and accept the
[applicable NVIDIA license terms](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/legal.html).
The project license does not grant rights to NVIDIA software or assets.

## Arduino-ESP32

Arduino-ESP32 v3.3.11 is installed separately and is not copied into this
repository. It is distributed under the
[GNU Lesser General Public License v2.1](https://github.com/espressif/arduino-esp32/blob/3.3.11/LICENSE.md).
Anyone distributing a compiled firmware binary is responsible for complying
with the licenses of the Arduino core and all linked libraries.

## CAD export tool

`linefollowingrobot_cad/urdf/linefollowingrobot.urdf` was generated using the
SolidWorks to URDF Exporter and retains its generated provenance comment. The
exporter itself is not included in this repository.
