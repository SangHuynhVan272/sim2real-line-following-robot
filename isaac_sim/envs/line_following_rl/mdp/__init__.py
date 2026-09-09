"""MDP terms for the line-following RL task, split one module per manager."""

from .actions import DutyAction, DutyActionCfg  # noqa: F401
from .events import randomize_scenario, reset_robot_on_track  # noqa: F401
from .observations import critic_observation, line_observation  # noqa: F401
from .rewards import (  # noqa: F401
    action_rate_penalty,
    duty_excess_penalty,
    heading_alignment,
    lateral_error_penalty,
    normalized_track_progress,
    reached_finish_bonus,
    terminal_failure_penalty,
    time_penalty,
)
from .terminations import line_lost, off_track, reached_finish, stalled, unstable_physics  # noqa: F401
