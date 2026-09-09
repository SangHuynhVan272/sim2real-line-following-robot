"""Rate validation shared by the RL control and camera clocks."""

from __future__ import annotations

import math


def integer_rate_stride(faster_hz: float, slower_hz: float, *, label: str) -> int:
    """Return the integer number of faster ticks per slower tick.

    The training lane must represent real sampling clocks exactly.  Rounding a
    non-integral ratio silently changes a measured rate, so reject it instead.
    """
    if faster_hz <= 0.0 or slower_hz <= 0.0:
        raise ValueError(f"{label} rates must be positive, got {faster_hz} and {slower_hz} Hz.")
    ratio = faster_hz / slower_hz
    stride = round(ratio)
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(
            f"{label} requires an integer rate ratio, got {faster_hz:g} / {slower_hz:g} = {ratio:g}."
        )
    return int(stride)
