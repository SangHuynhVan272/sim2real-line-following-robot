# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Huynh Van Sang.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO hyperparameters.

The actor is deliberately tiny.  It has to run at 60 Hz on an ESP32-S3 from a
generated C header with no ML runtime, so 7 -> 64 -> 64 -> 2 with ELU is roughly
4.8k parameters and a few microseconds per inference.  Widening it is not free:
it is a firmware cost.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LineFollowingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # The actor sees only the exported ABI; the critic additionally sees the
    # privileged group.  RSL-RL builds each network from the groups named here,
    # so this is what keeps the asymmetry out of the exported policy.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    }
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "line_following"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        # This actor starts from a rendered-camera-validated BC policy.  A
        # 0.5-rad raw Gaussian immediately visits saturated, reverse-wheel
        # states and one PPO update can erase that behaviour.  Keep enough
        # exploration for the randomized training lane while making the first
        # trust-region update local to the deployed policy.
        init_noise_std=0.1,
        # The critic may be larger than the actor: it never ships.
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
