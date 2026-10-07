"""Shared scheduling model used by the CSP, min-conflicts and gate agents.

A *value* for a flight variable is ``(stand, hold)``: the stand it parks at and
how many minutes it waits after its ETA before blocking on. The occupied
interval is ``[eta + hold, max(sched_dep, eta + hold + min_turn))`` and two
flights at the same stand must be at least ``BUFFER_MIN`` apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aerosync.config import BUFFER_MIN, DEFAULT_WEIGHTS, MAX_HOLD_MIN, STEP_MIN, BidWeights
from aerosync.sim.airport import Airport, Stand

Value = tuple[str, int]


@dataclass(frozen=True)
class Task:
    """A flight as seen by a planner (only observable information)."""

    id: str
    size: str
    eta: int
    sched_dep: int
    min_turn: int
    pref_terminal: str
    pax: int = 150

    def interval(self, hold: int) -> tuple[int, int]:
        start = self.eta + hold
        return start, max(self.sched_dep, start + self.min_turn)

    def dep_delay(self, hold: int) -> int:
        return self.interval(hold)[1] - self.sched_dep


@dataclass(frozen=True)
class Block:
    """A fixed, immovable occupation of a stand (aircraft on block, or a closure)."""

    stand: str
    start: int
    end: int
    label: str = ""
    is_closure: bool = False


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int,
             buffer: int = BUFFER_MIN) -> bool:
    """True if two stand occupations violate the separation buffer."""
    return a_start < b_end + buffer and b_start < a_end + buffer


def closure_overlaps(a_start: int, a_end: int, block: Block) -> bool:
    """Closures need no buffer: the stand is simply unavailable in ``[start, end)``."""
    return a_start < block.end and block.start < a_end


def block_conflicts(start: int, end: int, block: Block, buffer: int = BUFFER_MIN) -> bool:
    if block.is_closure:
        return closure_overlaps(start, end, block)
    return overlaps(start, end, block.start, block.end, buffer)


def holds(max_hold: int = MAX_HOLD_MIN) -> list[int]:
    return list(range(0, max_hold + 1, STEP_MIN))


CostFn = Callable[[Task, str, int], float]


def static_cost_fn(airport: Airport, weights: BidWeights = DEFAULT_WEIGHTS) -> CostFn:
    """Soft cost of a value that does not depend on other flights.

    ``w1*walk*pax/150 + w2*terminal_mismatch + w3*remote + w6*hold``.
    Walking is weighted per passenger-hundred so that big aircraft get priority
    on close gates.
    """

    def cost(task: Task, stand: str, hold: int) -> float:
        st: Stand = airport.stands[stand]
        walk = airport.walk_distance(stand, task.pref_terminal)
        mismatch = 0.0 if (st.terminal == task.pref_terminal) else 1.0
        if st.is_remote:
            mismatch = 0.0
        pax_factor = task.pax / 150.0
        return (
            weights.walk * walk * pax_factor
            + weights.mismatch * mismatch
            + weights.remote * (1.0 if st.is_remote else 0.0)
            + weights.delay * hold
        )

    return cost
