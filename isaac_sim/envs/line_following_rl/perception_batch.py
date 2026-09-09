"""Batched geometric perception -- the training-lane stand-in for the camera.

Training cannot render.  A 320x240 frame per environment would have to pass
through gain, motion blur, rolling shutter, a bilinear lens remap, a real JPEG
round-trip and then Otsu plus a per-row run tracer, all in single-threaded
numpy.  At 1024 environments and the configured policy rate that is hopeless, so the training lane
replaces the whole camera with the projection that already labels the
supervised dataset: ``image_observation_from_track_geometry``.

The semantics reproduced here are exactly that function's:

* project the known tape centreline into the pinhole image,
* reject samples behind the camera or beyond ``geometry_label_sensor_margin_px``,
* apply the forward Brown-Conrady lens model,
* trace the same contiguous bottom-to-top row span as the image algorithm,
* fit the same linear/quadratic centreline inside that visible span,
* clamp near/look-ahead rows to that span before producing ``e_y/e_theta``.

A pose whose near or far row has no crossing returns ``valid = False``.  That is
the training-lane equivalent of a camera tick that produced no detection, and it
is what lets staleness and ``line_confidence`` behave the same in both lanes.

``SAMPLE_COUNT`` is fixed for the whole batch, while the reference derives it
per pose from a 4 mm spacing. The chosen value matches the reference's
unclamped look-ahead span so mid-track poses use the same grid. Endpoint
clamping can still produce a small upper-tail difference, which
``check_rl_parity.py`` measures explicitly.
"""

from __future__ import annotations

import math

import torch

from .track_batch import TrackBatch

#: Track samples per environment.  ceil(0.71 / 0.004) + 1 -- the reference's
#: own count for an unclamped span.  Do not "round" this; see the module note.
SAMPLE_COUNT = 179
#: Look-ahead window around the robot, in metres of track arc length.
SPAN_BEHIND_M = 0.06
SPAN_AHEAD_M = 0.65
#: Height of the tape surface above the floor, as the simulator spawns it.
TAPE_HEIGHT_M = 0.002


class GeometricPerceptionBatch:
    """Evaluate ``[e_y, e_theta, valid]`` for a batch of camera poses."""

    def __init__(self, config: dict, track: TrackBatch, device: torch.device | str = "cpu") -> None:
        robot, vision = config["robot"], config["vision"]
        width, height = (int(value) for value in robot["camera_resolution_px"])
        self.device = torch.device(device)
        self.track = track
        self.width, self.height = width, height
        self.half_width = (width - 1) / 2.0
        self.center_x, self.center_y = (width - 1) / 2.0, (height - 1) / 2.0
        self.near_row = float(vision["roi_rows_fraction"][0]) * (height - 1)
        self.far_row = float(vision["lookahead_row_fraction"]) * (height - 1)
        self.trace_row_values = sorted(
            (float(value) * (height - 1) for value in vision["centerline_rows_fraction"]), reverse=True,
        )
        self.trace_rows = torch.tensor(self.trace_row_values, device=self.device)
        self.minimum_trace_points = int(vision["minimum_centerline_points"])
        self.sensor_margin = float(vision.get("geometry_label_sensor_margin_px", 16.0))
        self.scale_x = max(self.center_x, 1.0)
        self.scale_y = max(self.center_y, 1.0)

    def _focal_px(self, hfov_deg: torch.Tensor) -> torch.Tensor:
        return self.half_width / torch.tan(torch.deg2rad(hfov_deg) / 2.0)

    def _distort(
        self, pixel_x: torch.Tensor, pixel_y: torch.Tensor, lens: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward Brown-Conrady, matching ``distort_projected_pixel``."""
        k1, k2 = lens["k1"][:, None], lens["k2"][:, None]
        p1, p2 = lens["p1"][:, None], lens["p2"][:, None]
        x = (pixel_x - self.center_x) / self.scale_x
        y = (pixel_y - self.center_y) / self.scale_y
        radius_sq = x * x + y * y
        radial = 1.0 + k1 * radius_sq + k2 * radius_sq * radius_sq
        distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (radius_sq + 2.0 * x * x)
        distorted_y = y * radial + p1 * (radius_sq + 2.0 * y * y) + 2.0 * p2 * x * y
        return distorted_x * self.scale_x + self.center_x, distorted_y * self.scale_y + self.center_y

    def _crossing_x(
        self, pixel_x: torch.Tensor, pixel_y: torch.Tensor, valid: torch.Tensor, row: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First crossing of an image row, as ``(pixel_x, found)``.

        Sample index increases with track progress, so the earliest crossing
        along the track is the lowest index -- the batched form of the
        reference's ``min(candidates, key=progress)``.
        """
        y_a, y_b = pixel_y[:, :-1], pixel_y[:, 1:]
        x_a, x_b = pixel_x[:, :-1], pixel_x[:, 1:]
        pair_valid = valid[:, :-1] & valid[:, 1:]
        spans = (row >= torch.minimum(y_a, y_b)) & (row <= torch.maximum(y_a, y_b))
        separated = (y_b - y_a).abs() >= 1e-6
        candidate = pair_valid & spans & separated
        fraction = (row - y_a) / torch.where(separated, y_b - y_a, torch.ones_like(y_a))
        crossing = x_a + fraction * (x_b - x_a)
        # argmax over a boolean picks the first True; guard the all-False case.
        found = candidate.any(dim=1)
        first = candidate.to(torch.int8).argmax(dim=1)
        rows = torch.arange(pixel_x.shape[0], device=pixel_x.device)
        return crossing[rows, first], found

    def evaluate(
        self,
        eye: torch.Tensor,
        forward: torch.Tensor,
        robot_xy: torch.Tensor,
        hfov_deg: torch.Tensor,
        lens: dict[str, torch.Tensor],
        roll_deg: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(e_y, e_theta_rad, valid)`` for ``[N, ...]`` camera poses.

        ``eye`` and ``forward`` are the camera position and optical axis in world
        coordinates; ``robot_xy`` locates the robot on the track so the sampled
        span follows it.  ``e_y`` and ``e_theta_rad`` are meaningless where
        ``valid`` is False and are returned as zero there.
        """
        count = eye.shape[0]
        forward = forward / torch.linalg.norm(forward, dim=-1, keepdim=True).clamp(min=1e-9)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=eye.device).expand_as(forward)
        right = torch.cross(forward, world_up, dim=-1)
        right_norm = torch.linalg.norm(right, dim=-1, keepdim=True)
        # A camera pointing straight down has no stable image right vector.
        degenerate = right_norm.squeeze(-1) < 1e-6
        right = right / right_norm.clamp(min=1e-9)
        image_up = torch.cross(right, forward, dim=-1)
        image_up = image_up / torch.linalg.norm(image_up, dim=-1, keepdim=True).clamp(min=1e-9)
        if roll_deg is not None:
            roll = torch.deg2rad(roll_deg)[:, None]
            cosine, sine = torch.cos(roll), torch.sin(roll)
            right, image_up = right * cosine + image_up * sine, image_up * cosine - right * sine

        progress, _ = self.track.project(robot_xy)
        start = (progress - SPAN_BEHIND_M).clamp(min=0.0)
        stop = (progress + SPAN_AHEAD_M).clamp(
            max=self.track.total_length + self.track.finish_extension_m,
        )
        ramp = torch.linspace(0.0, 1.0, SAMPLE_COUNT, device=eye.device)
        samples = start[:, None] + ramp[None, :] * (stop - start)[:, None]   # [N, S]

        points = self.track.point_at(samples)                                 # [N, S, 2]
        tape = torch.full_like(points[..., :1], TAPE_HEIGHT_M)
        offset = torch.cat([points, tape], dim=-1) - eye[:, None, :]          # [N, S, 3]
        depth = (offset * forward[:, None, :]).sum(-1)                        # [N, S]
        in_front = depth > 1e-5
        safe_depth = torch.where(in_front, depth, torch.ones_like(depth))

        focal = self._focal_px(hfov_deg)[:, None]
        pixel_x = self.half_width + focal * (offset * right[:, None, :]).sum(-1) / safe_depth
        pixel_y = self.center_y - focal * (offset * image_up[:, None, :]).sum(-1) / safe_depth
        on_sensor = (
            (pixel_x >= -self.sensor_margin) & (pixel_x <= (self.width - 1) + self.sensor_margin)
            & (pixel_y >= -self.sensor_margin) & (pixel_y <= (self.height - 1) + self.sensor_margin)
        )
        valid = in_front & on_sensor & ~degenerate[:, None]
        pixel_x, pixel_y = self._distort(pixel_x, pixel_y, lens)

        # Match trace_centerline(): ignore empty rows below the first hit, then
        # stop at the first gap. A fixed near/far crossing incorrectly declared
        # corners lost even while the rendered tracer still had 3+ good rows.
        crossings: list[torch.Tensor] = []
        found_rows: list[torch.Tensor] = []
        for row in self.trace_row_values:
            crossing, found = self._crossing_x(pixel_x, pixel_y, valid, row)
            crossings.append(crossing)
            found_rows.append(found)
        crossing_x = torch.stack(crossings, dim=1)
        found = torch.stack(found_rows, dim=1)
        active = torch.zeros(count, dtype=torch.bool, device=eye.device)
        broken = torch.zeros_like(active)
        included_rows: list[torch.Tensor] = []
        for column in range(found.shape[1]):
            hit = found[:, column]
            broken |= active & ~hit
            included_rows.append(hit & ~broken)
            active |= hit
        included = torch.stack(included_rows, dim=1)
        trace_count = included.sum(dim=1)
        detected = trace_count >= self.minimum_trace_points

        rows = self.trace_rows.to(dtype=pixel_x.dtype)[None, :].expand(count, -1)
        positive_inf = torch.full_like(rows, torch.inf)
        negative_inf = torch.full_like(rows, -torch.inf)
        farthest = torch.where(included, rows, positive_inf).min(dim=1).values
        nearest = torch.where(included, rows, negative_inf).max(dim=1).values
        farthest = torch.where(detected, farthest, torch.zeros_like(farthest))
        nearest = torch.where(detected, nearest, torch.ones_like(nearest))
        near_row = torch.clamp(torch.full_like(nearest, self.near_row), min=farthest, max=nearest)
        far_row = torch.clamp(torch.full_like(nearest, self.far_row), min=farthest, max=nearest)

        # Batched normal equations reproduce numpy.polyfit's least-squares
        # centreline: quadratic for 4+ points, linear for exactly 3.
        weight = included.to(pixel_x.dtype)
        normalized_rows = (rows - self.center_y) / self.scale_y
        near_normalized = (near_row - self.center_y) / self.scale_y
        far_normalized = (far_row - self.center_y) / self.scale_y
        design_quadratic = torch.stack(
            [normalized_rows.square(), normalized_rows, torch.ones_like(rows)], dim=-1,
        )
        normal_quadratic = torch.einsum("nri,nrj,nr->nij", design_quadratic, design_quadratic, weight)
        rhs_quadratic = torch.einsum("nri,nr,nr->ni", design_quadratic, crossing_x, weight)
        use_quadratic = trace_count >= 4
        identity3 = torch.eye(3, device=eye.device, dtype=pixel_x.dtype)[None]
        normal_quadratic = torch.where(use_quadratic[:, None, None], normal_quadratic, identity3)
        rhs_quadratic = torch.where(use_quadratic[:, None], rhs_quadratic, torch.zeros_like(rhs_quadratic))
        coefficient_quadratic = torch.linalg.solve(normal_quadratic, rhs_quadratic)

        design_linear = torch.stack([normalized_rows, torch.ones_like(rows)], dim=-1)
        normal_linear = torch.einsum("nri,nrj,nr->nij", design_linear, design_linear, weight)
        rhs_linear = torch.einsum("nri,nr,nr->ni", design_linear, crossing_x, weight)
        use_linear = detected & ~use_quadratic
        identity2 = torch.eye(2, device=eye.device, dtype=pixel_x.dtype)[None]
        normal_linear = torch.where(use_linear[:, None, None], normal_linear, identity2)
        rhs_linear = torch.where(use_linear[:, None], rhs_linear, torch.zeros_like(rhs_linear))
        coefficient_linear = torch.linalg.solve(normal_linear, rhs_linear)

        near_quadratic = coefficient_quadratic[:, 0] * near_normalized.square() + coefficient_quadratic[:, 1] * near_normalized + coefficient_quadratic[:, 2]
        far_quadratic = coefficient_quadratic[:, 0] * far_normalized.square() + coefficient_quadratic[:, 1] * far_normalized + coefficient_quadratic[:, 2]
        near_linear = coefficient_linear[:, 0] * near_normalized + coefficient_linear[:, 1]
        far_linear = coefficient_linear[:, 0] * far_normalized + coefficient_linear[:, 1]
        near_x = torch.where(use_quadratic, near_quadratic, near_linear)
        far_x = torch.where(use_quadratic, far_quadratic, far_linear)

        lateral = ((far_x - self.half_width) / self.half_width).clamp(-1.0, 1.0)
        heading = torch.atan2(far_x - near_x, (near_row - far_row).clamp(min=1.0))
        zero = torch.zeros(count, device=eye.device)
        return torch.where(detected, lateral, zero), torch.where(detected, heading, zero), detected
