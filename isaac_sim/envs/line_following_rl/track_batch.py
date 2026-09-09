"""Batched track polyline maths for the RL training lane.

The validated simulator walks the track with per-episode numpy helpers in
``isaac_sim/scripts/run_line_following.py``.  Training runs 1024 environments
at the configured policy rate, so those scalar helpers are re-expressed here as torch tensor
operations over the whole batch.

This module is the *second* implementation of the same geometry, which is a
liability: the two can drift.  ``isaac_sim/scripts/check_rl_parity.py`` exists
to keep them honest and must be re-run whenever either side changes.

Nothing here imports Isaac Sim, so it can be unit-tested in plain Python.
"""

from __future__ import annotations

import torch


class TrackBatch:
    """An arc-length parameterised polyline evaluated for a batch of poses.

    ``points`` is the ``track.points_xy_m`` polyline from ``default.json``.
    Only open tracks are supported, matching the current configuration; a
    closed track would need the wrap-around handling that ``track_progress``
    already carries on the simulator side.
    """

    def __init__(
        self, points: list[list[float]], device: torch.device | str = "cpu",
        finish_extension_m: float = 0.0,
    ) -> None:
        vertices = torch.tensor(points, dtype=torch.float32, device=device)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 2:
            raise ValueError("track points must be an (N >= 2, 2) polyline")
        self.device = torch.device(device)
        self.vertices = vertices                                  # [V, 2]
        self.starts = vertices[:-1]                               # [S, 2]
        self.deltas = vertices[1:] - vertices[:-1]                # [S, 2]
        self.lengths = torch.linalg.norm(self.deltas, dim=-1)     # [S]
        # Guard against duplicated vertices so the projection never divides by
        # zero; a zero-length segment simply never wins the nearest-point test.
        self.safe_lengths = torch.clamp(self.lengths, min=1e-9)
        self.cumulative = torch.cat([
            torch.zeros(1, device=self.device),
            torch.cumsum(self.lengths, dim=0),
        ])                                                        # [S + 1]
        self.total_length = float(self.cumulative[-1])
        # The rendered scene lays finish_tape_extension_m of real tape past the
        # last vertex, so a camera near the finish still sees a line.  Without
        # the same run-out here the geometric stand-in goes blind over the final
        # look-ahead span and labels a valid finishing approach as line loss.
        self.finish_extension_m = max(0.0, float(finish_extension_m))

    def project(self, positions_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(progress_m, lateral_distance_m)`` for ``[N, 2]`` positions.

        ``progress_m`` is arc length along the polyline of the nearest point and
        ``lateral_distance_m`` is the unsigned distance to it -- the batched
        equivalent of ``track_progress`` plus ``nearest_track_segment``.
        """
        offset = positions_xy[:, None, :] - self.starts[None, :, :]          # [N, S, 2]
        fraction = (offset * self.deltas[None]).sum(-1) / (self.safe_lengths ** 2)[None]
        fraction = fraction.clamp(0.0, 1.0)                                   # [N, S]
        closest = self.starts[None] + fraction[..., None] * self.deltas[None]  # [N, S, 2]
        distance = torch.linalg.norm(positions_xy[:, None, :] - closest, dim=-1)
        best = distance.argmin(dim=1)                                          # [N]
        rows = torch.arange(positions_xy.shape[0], device=positions_xy.device)
        progress = self.cumulative[best] + fraction[rows, best] * self.lengths[best]
        return progress, distance[rows, best]

    def point_at(self, progress_m: torch.Tensor) -> torch.Tensor:
        """Return ``[..., 2]`` track points at the given arc lengths.

        Arc lengths clamp to the start and to the end of the run-out tape, so
        the sampled centreline covers exactly the tape the scene actually lays.
        """
        flat = progress_m.reshape(-1).clamp(0.0, self.total_length + self.finish_extension_m)
        # ``right - 1`` is the segment containing this arc length.
        index = torch.searchsorted(self.cumulative, flat, right=True) - 1
        index = index.clamp(0, self.lengths.shape[0] - 1)
        local = (flat - self.cumulative[index]) / self.safe_lengths[index]
        point = self.starts[index] + local[:, None] * self.deltas[index]
        return point.reshape(*progress_m.shape, 2)

    def tangent_at(self, progress_m: torch.Tensor) -> torch.Tensor:
        """Return unit forward tangents for the segments containing arc lengths."""
        flat = progress_m.reshape(-1).clamp(0.0, self.total_length)
        index = torch.searchsorted(self.cumulative, flat, right=True) - 1
        index = index.clamp(0, self.lengths.shape[0] - 1)
        tangent = self.deltas[index] / self.safe_lengths[index, None]
        return tangent.reshape(*progress_m.shape, 2)
