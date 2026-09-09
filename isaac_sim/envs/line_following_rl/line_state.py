"""Per-environment perception and actuation state for the RL training lane.

The frozen observation is not memoryless: ``line_confidence`` measures how old
the look-ahead is, and ``duty_prev`` is what the slew limiter clamps against.
Both need buffers that survive between steps, so they live here rather than
being recomputed from whatever the simulator happens to expose.

Two rules from the validated simulator are reproduced exactly, because breaking
either produces a robot that looks fine in training and runs away on hardware:

* **Staleness is measured on the camera clock only.**  ``line_loss_s`` grows on
  a camera tick that found nothing and resets on one that did.  It must never be
  derived from ``line_confidence``, which is itself derived from ``line_loss_s``
  -- that circle pins both at zero and the policy then drives on a frozen
  observation forever instead of failing.
* **A missing detection is not a stop.**  The last valid observation is held
  while confidence decays.
"""

from __future__ import annotations

import torch

from .perception_batch import GeometricPerceptionBatch
from .track_batch import TrackBatch


class LineState:
    """Buffers the observation history for a batch of environments."""

    def __init__(
        self,
        num_envs: int,
        config: dict,
        device: torch.device | str,
        camera_stride: int,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.config = config
        self.camera_stride = int(camera_stride)
        self.max_line_loss_s = float(config["evaluation"]["max_line_loss_s"])
        self.policy_dt = 1.0 / float(config["physics"]["policy_hz"])
        self.camera_period_s = self.camera_stride * self.policy_dt
        def _max_ticks(key: str) -> int:
            """Buffer depth is the widest delay the randomization can draw."""
            value = config["randomization"][key]
            return int(max(value)) if isinstance(value, (list, tuple)) else int(value)

        self.encoder_delay_ticks = _max_ticks("encoder_delay_ticks")
        # Counted in *camera* ticks, not policy ticks. The rendered lane combines
        # an explicit frame queue with annotator delivery latency. Reproducing
        # that delay here prevents training on a more responsive sensor than
        # the rendered and physical cameras can provide.
        self.camera_delay_ticks = _max_ticks("camera_delay_ticks")
        self.actuation_delay_ticks = _max_ticks("actuation_delay_ticks")
        if self.encoder_delay_ticks < 0 or self.actuation_delay_ticks < 0:
            raise ValueError("encoder_delay_ticks and actuation_delay_ticks must be non-negative.")
        if self.camera_delay_ticks < 0:
            raise ValueError("camera_delay_ticks must be non-negative.")

        self.track = TrackBatch(
            config["track"]["points_xy_m"], device=self.device,
            finish_extension_m=(
                0.0 if config["track"].get("closed", False)
                else float(config["track"].get("finish_tape_extension_m", 0.0))
            ),
        )
        self.perception = GeometricPerceptionBatch(config, self.track, device=self.device)

        zeros = lambda *shape: torch.zeros(*shape, device=self.device)  # noqa: E731
        self.duty_prev = zeros(num_envs, 2)
        self.duty_delta = zeros(num_envs, 2)
        self.last_valid = zeros(num_envs, 2)
        self.ever_detected = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.line_loss_s = zeros(num_envs)
        self.max_line_loss_s_seen = zeros(num_envs)
        self.confidence = zeros(num_envs)
        self.measured_rpm = zeros(num_envs, 2)
        self.encoder_delay_buffer = zeros(num_envs, self.encoder_delay_ticks, 2)
        self.actuation_delay_buffer = zeros(num_envs, self.actuation_delay_ticks, 2)
        # e_y, e_theta and the detection flag travel together: the rendered
        # lane delays the frame, so line-loss accounting sees the delayed
        # detection too, not a fresh one.
        self.camera_delay_buffer = zeros(num_envs, self.camera_delay_ticks, 3)
        self.progress = zeros(num_envs)
        self.previous_progress = zeros(num_envs)
        self.stall_reference_progress = zeros(num_envs)
        self.stall_duration_s = zeros(num_envs)
        self.distance_to_track = zeros(num_envs)
        # ObservationManager may be asked for the current observations more
        # than once (for example while RSL-RL constructs its runner).  Camera
        # age and encoder noise are physical state, not properties of that
        # read, so record the policy step at which each environment was last
        # advanced.  ``-1`` makes the initial reset produce one sample at
        # policy step zero.
        self.last_observation_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=self.device,
        )

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Clear the history of the environments that just reset."""
        self.duty_prev[env_ids] = 0.0
        self.duty_delta[env_ids] = 0.0
        self.last_valid[env_ids] = 0.0
        self.ever_detected[env_ids] = False
        self.line_loss_s[env_ids] = 0.0
        self.max_line_loss_s_seen[env_ids] = 0.0
        self.confidence[env_ids] = 0.0
        self.measured_rpm[env_ids] = 0.0
        if self.encoder_delay_ticks:
            self.encoder_delay_buffer[env_ids] = 0.0
        if self.actuation_delay_ticks:
            self.actuation_delay_buffer[env_ids] = 0.0
        # A cleared buffer reads back as "not detected", which is what the
        # rendered lane reports on its first camera tick (queue_not_ready).
        if self.camera_delay_ticks:
            self.camera_delay_buffer[env_ids] = 0.0
        # reset_robot_on_track() immediately replaces these with the projected
        # start pose. Clearing them here prevents stale finish state from being
        # visible between reset hooks or in focused unit tests.
        self.progress[env_ids] = 0.0
        self.previous_progress[env_ids] = 0.0
        self.stall_reference_progress[env_ids] = 0.0
        self.stall_duration_s[env_ids] = 0.0
        self.distance_to_track[env_ids] = 0.0
        self.last_observation_step[env_ids] = -1

    @staticmethod
    def _delay_values(
        buffer: torch.Tensor,
        values: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Delay values by exactly ``buffer.shape[1]`` policy ticks."""
        if buffer.shape[1] == 0:
            return values
        delayed = buffer[env_ids, 0].clone()
        if buffer.shape[1] > 1:
            buffer[env_ids, :-1] = buffer[env_ids, 1:].clone()
        buffer[env_ids, -1] = values
        return delayed

    @staticmethod
    def _delay_values_variable(
        buffer: torch.Tensor,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        ticks: torch.Tensor,
    ) -> torch.Tensor:
        """Delay per environment by its own ``ticks``, up to the buffer depth.

        The buffer is always as deep as the widest draw; a shallower delay reads
        further along it.  Index ``depth - ticks`` is the entry written that many
        calls ago, because the read happens before this call's shift: index 0
        holds the oldest sample and index ``depth - 1`` the one written last.

        Real latency is not a constant. A dropped frame, slow image processing
        or a scheduler hiccup moves it by a whole period. Randomizing the delay
        prevents the policy from depending on timing that the hardware can only
        approximate.
        """
        depth = buffer.shape[1]
        if depth == 0:
            return values
        index = (depth - ticks[env_ids].to(torch.long)).clamp(0, depth - 1)
        delayed = buffer[env_ids, index].clone()
        if depth > 1:
            buffer[env_ids, :-1] = buffer[env_ids, 1:].clone()
        buffer[env_ids, -1] = values
        return delayed

    def delay_encoder(self, values: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Return RPM delayed by the configured number of policy ticks."""
        return self._delay_values(self.encoder_delay_buffer, values, env_ids)

    def delay_actuation(self, values: torch.Tensor, ticks: torch.Tensor) -> torch.Tensor:
        """Return duty delayed by each environment's own number of policy ticks."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self._delay_values_variable(
            self.actuation_delay_buffer, values, env_ids, ticks,
        )

    @staticmethod
    def advance_progress(previous_best: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
        """Keep route completion monotonic so reversing cannot earn it twice."""
        return torch.maximum(previous_best, projected)

    def update_camera(
        self,
        env_ids: torch.Tensor,
        eye: torch.Tensor,
        forward: torch.Tensor,
        robot_xy: torch.Tensor,
        hfov_deg: torch.Tensor,
        lens: dict[str, torch.Tensor],
        roll_deg: torch.Tensor,
        delay_ticks: torch.Tensor,
    ) -> None:
        """Take one camera sample and age the ones that found nothing.

        Call this only on a camera tick.  Calling it every policy tick would make
        the look-ahead refresh four times faster than the sensor can deliver,
        which is the single easiest way to train a policy that cannot work.
        """
        e_y, e_theta, detected = self.perception.evaluate(
            eye,
            forward,
            robot_xy,
            hfov_deg,
            lens,
            roll_deg,
        )
        # Hold the sample for the configured number of camera periods before it
        # becomes the observation, matching the rendered lane's frame queue.
        # This has to happen before line-loss accounting, not after: the gate
        # counts a loss when the *delivered* frame found nothing, so delaying
        # only e_y/e_theta while letting a fresh detection flag through would
        # leave the training lane recovering from a loss one period early.
        if self.camera_delay_ticks:
            sample = torch.stack([e_y, e_theta, detected.to(e_y.dtype)], dim=-1)
            delayed = self._delay_values_variable(
                self.camera_delay_buffer, sample, env_ids, delay_ticks,
            )
            e_y, e_theta = delayed[:, 0], delayed[:, 1]
            detected = delayed[:, 2] > 0.5
        # ``ever_detected`` is still the value from before this frame, which is
        # exactly the question being asked: had the tape been seen at all yet?
        # Firmware waits indefinitely before its first lock and only enforces
        # max_line_loss_s afterwards, so the pre-lock wait must not accumulate
        # here either -- otherwise the two lanes disagree about what counts as
        # a failure, and a marginal start pose is scored as a lost line.
        zero = torch.zeros_like(self.line_loss_s[env_ids])
        line_loss_s = torch.where(
            detected,
            zero,
            torch.where(
                self.ever_detected[env_ids],
                self.line_loss_s[env_ids] + self.camera_period_s,
                zero,
            ),
        )
        self.line_loss_s[env_ids] = line_loss_s
        self.max_line_loss_s_seen[env_ids] = torch.maximum(
            self.max_line_loss_s_seen[env_ids], line_loss_s,
        )
        fresh = detected[:, None]
        self.last_valid[env_ids] = torch.where(
            fresh,
            torch.stack([e_y, e_theta], dim=-1),
            self.last_valid[env_ids],
        )
        self.ever_detected[env_ids] |= detected

    def refresh_confidence(self, env_ids: torch.Tensor) -> None:
        """Recompute ``line_confidence`` from staleness, never the other way round."""
        decayed = 1.0 - self.line_loss_s[env_ids] / max(1e-6, self.max_line_loss_s)
        self.confidence[env_ids] = torch.where(
            self.ever_detected[env_ids], decayed.clamp(0.0, 1.0), torch.zeros_like(decayed),
        )

    def observation(self) -> torch.Tensor:
        """Return the frozen ``[N, 7]`` ABI vector.

        Order, units and normalisation are fixed by the firmware contract:
        ``e_y`` in [-1, 1], ``e_theta`` in radians, confidence in [0, 1], wheel
        speeds in RPM, previous duty in [-1, 1].  Do not reorder or rescale.
        """
        held = torch.where(self.ever_detected[:, None], self.last_valid, torch.zeros_like(self.last_valid))
        return torch.cat([held, self.confidence[:, None], self.measured_rpm, self.duty_prev], dim=-1)
