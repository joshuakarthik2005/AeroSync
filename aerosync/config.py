"""Global parameters for AeroSync.

Every tunable number used by the agents, the CSP solver and the metrics lives
here so that the code, the README and ``docs/slide_assets/facts.md`` can stay
consistent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

STEP_MIN: int = 5
"""Simulation timestep in minutes."""

BUFFER_MIN: int = 15
"""Minimum separation between consecutive flights at the same stand (hard constraint)."""

LOOKAHEAD_MIN: int = 60
"""A flight requests (or confirms) its gate this many minutes before its ETA."""

MAX_HOLD_MIN: int = 120
"""Largest on-block hold (aircraft waiting for a stand) the planners consider."""

ALPHA: float = 0.30
"""Coordination weight applied to the load reported by neighbouring gate agents."""

MIN_TURN_FRACTION: float = 0.80
"""Minimum turnaround = this fraction of the scheduled ground time (rounded up to a step)."""

MOVEMENT_CONGESTION: float = 0.30
"""Extra edge-cost multiplier per aircraft movement near an apron road node."""

MOVEMENT_WINDOW_MIN: int = 10
"""Aircraft on/off-block events within +/- this window congest the adjacent apron road."""

MIN_CONFLICTS_MAX_STEPS: int = 2000
"""Iteration cap for min-conflicts repair."""

CSP_NODE_LIMIT: int = 200_000
"""Safety cap on backtracking nodes for the initial CSP."""

SERVICE_MIN: dict[str, int] = {"fuel": 20, "bus": 15, "tug": 10}
"""Service duration once a ground vehicle reaches the stand."""

VEHICLE_SPEED_M_PER_MIN: dict[str, float] = {"tug": 250.0, "fuel": 330.0, "bus": 400.0}
"""Free-flow apron speeds (15, 20 and 24 km/h)."""

REMOTE_WALK_M: float = 350.0
"""Walking-equivalent distance for a remote stand (stairs + bus gate inside the terminal)."""


@dataclass(frozen=True)
class BidWeights:
    """Weights of the gate-agent bid cost.

    ``bid = w1*walk_m + w2*terminal_mismatch + w3*remote + w4*buffer_risk
    + w5*neighbour_pressure + w6*hold_min + w7*reassignment``
    """

    walk: float = 0.05  # w1, per metre of passenger walking (x pax/150)
    mismatch: float = 40.0  # w2, flight parked away from its preferred terminal
    remote: float = 60.0  # w3, remote stand (bus boarding)
    buffer_risk: float = 25.0  # w4, slack to neighbouring slots is thin (0..1)
    neighbour: float = 100.0  # w5, alpha * mean neighbour load (0..alpha)
    delay: float = 4.0  # w6, per minute the aircraft holds waiting for the stand
    reassign: float = 15.0  # w7, moving a flight away from its current gate

    def as_dict(self) -> dict[str, float]:
        """Return the weights as a plain dict (for logs and exports)."""
        return asdict(self)


DEFAULT_WEIGHTS = BidWeights()


def round_up_step(minutes: float) -> int:
    """Round ``minutes`` up to the next multiple of :data:`STEP_MIN`."""
    m = int(-(-minutes // STEP_MIN) * STEP_MIN)
    return m


def fmt_clock(minutes_from_start: int, start_hour: int) -> str:
    """Format simulation minutes as an ``HH:MM`` wall-clock string."""
    total = start_hour * 60 + int(minutes_from_start)
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"
