"""Actor ABI plus simulator-only state for an asymmetric critic.

The **policy** group must stay exactly what the ESP32-S3 can hand it.  Anything
richer -- track progress, distance to the tape, the true pose -- exists in
simulation and does not exist on the robot.  Putting a privileged value in the
policy group trains a policy that cannot be deployed, and the failure is
invisible until hardware.

The **critic** is never exported, so it may see the hidden state that actually
determines rewards and terminations.  Without that split two identical
camera/encoder observations can carry different remaining distance and
different terminal returns, which makes the value function ill-posed.
"""

from __future__ import annotations

import torch

from .scene_state import camera_pose, get_line_state, get_scenario, local_xy, measured_rpm

#: Shared with ``terminations.off_track`` so the critic's normalizer and the
#: termination boundary cannot drift apart.
OFF_TRACK_LIMIT_M = 0.25


def update_line_state(env):
    """Advance sensor and route state once for the current post-physics step.

    Two rates, one loop: encoders and ``duty_prev`` refresh every policy tick
    while the look-ahead is held between camera ticks.  That is what lets the
    actuation loop run at the configured policy rate behind the configured
    camera rate, and it is the behaviour the firmware has.

    Isaac Lab evaluates terminations and rewards *before* observations.  While
    this refresh lived only inside the observation term, both of those read the
    **preceding** physics step.  All three managers now call this idempotent
    helper: the first caller advances the state, later callers only read it.
    """
    state = get_line_state(env)
    scenario = get_scenario(env)

    # RSL-RL reads observations while it constructs the runner and again when
    # a rollout starts.  Those reads must be idempotent: no physics occurred,
    # so neither the camera clock nor encoder noise may advance.  ManagerBased
    # RLEnv increments this counter exactly once after every decimated policy
    # step, before it asks ObservationManager for the next observation.
    policy_step = int(env.common_step_counter)
    env_ids = torch.nonzero(state.last_observation_step != policy_step, as_tuple=False).squeeze(-1)
    if env_ids.numel() == 0:
        return state

    # Camera phase is global just like the physical 30 Hz sensor.  Step zero
    # is a valid initial frame; an environment reset on the alternate 60 Hz
    # policy phase correctly waits one tick for the next camera frame.
    if policy_step % state.camera_stride == 0:
        eye, forward, robot_xy = camera_pose(env)
        lens = {key: scenario[f"lens_{key}"] for key in ("k1", "k2", "p1", "p2")}
        state.update_camera(
            env_ids,
            eye[env_ids],
            forward[env_ids],
            robot_xy[env_ids],
            scenario["camera_hfov_deg"][env_ids],
            {key: value[env_ids] for key, value in lens.items()},
            scenario["camera_roll_deg"][env_ids],
            scenario["camera_delay_ticks"].round(),
        )
    state.refresh_confidence(env_ids)

    # Per-episode noise amplitude, not one global constant: the encoder scale is
    # still an unconfirmed 1400 count/rev and its noise floor has never been
    # measured, so a policy trained against a single figure depends on a number
    # nobody has checked.
    noise = (torch.randn_like(state.measured_rpm[env_ids])
             * scenario["encoder_noise_rpm"][env_ids, None])
    sampled_rpm = measured_rpm(env)[env_ids] * scenario["encoder_scale"][env_ids, None] + noise
    state.measured_rpm[env_ids] = state.delay_encoder(sampled_rpm, env_ids)

    projected_progress, distance = state.track.project(
        local_xy(env, env.scene["robot"].data.root_pos_w[env_ids, :2], env_ids))
    # Reward only genuinely new route completion.  Raw nearest-point progress
    # can decrease when the robot reverses; paying every later re-crossing of
    # that same segment lets PPO farm progress by oscillating near the start.
    # The camera evaluator already reports maximum travelled progress, so the
    # training lane must use the same monotonic definition.
    state.previous_progress[env_ids] = state.progress[env_ids]
    state.progress[env_ids] = state.advance_progress(state.progress[env_ids], projected_progress)
    # A direct-duty policy can otherwise settle forever with one side below the
    # randomized breakaway duty.  Track a small *windowed* progress increment,
    # rather than noisy per-tick motion, so the training termination can reject
    # that deadlock without penalizing ordinary low-speed cornering.
    stall_progress_m = float(env.cfg.line_config["rl"]["terminations"]["stall_progress_m"])
    made_progress = (
        state.progress[env_ids] - state.stall_reference_progress[env_ids]
    ) >= stall_progress_m
    state.stall_duration_s[env_ids] = torch.where(
        made_progress,
        torch.zeros_like(state.stall_duration_s[env_ids]),
        state.stall_duration_s[env_ids] + state.policy_dt,
    )
    state.stall_reference_progress[env_ids] = torch.where(
        made_progress,
        state.progress[env_ids],
        state.stall_reference_progress[env_ids],
    )
    state.distance_to_track[env_ids] = distance
    state.last_observation_step[env_ids] = policy_step
    return state


def line_observation(env) -> torch.Tensor:
    """Return the frozen ``[N, 7]`` firmware observation, in ABI order."""
    return update_line_state(env).observation()


def critic_observation(env) -> torch.Tensor:
    """Return normalized simulator-only state for the value function.

    RSL-RL joins this group to the critic only; the actor and the exported C
    policy stay 7 -> 64 -> 64 -> 2.  Every entry is a dimensionless ratio in
    roughly [0, 1], so the critic sees comparable magnitudes with no learned
    normalizer, and every entry is something the reward or a termination
    actually reads.  Route target is not among them: this lane always runs the
    whole polyline, so it would be a constant.
    """
    state = update_line_state(env)
    total = max(float(state.track.total_length), 1.0e-6)
    evaluation = env.cfg.line_config["evaluation"]
    stall_timeout_s = float(env.cfg.line_config["rl"]["terminations"]["stall_timeout_s"])
    episode_denominator = max(1.0, float(env.max_episode_length))
    return torch.stack(
        [
            state.progress / total,
            state.previous_progress / total,
            ((total - state.progress) / total).clamp(min=0.0),
            state.distance_to_track / OFF_TRACK_LIMIT_M,
            state.line_loss_s / max(float(evaluation["max_line_loss_s"]), 1.0e-6),
            state.stall_duration_s / max(stall_timeout_s, 1.0e-6),
            env.episode_length_buf.to(torch.float32) / episode_denominator,
        ],
        dim=-1,
    )
