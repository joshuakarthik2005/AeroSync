"""Flight Agent: represents one aircraft turnaround.

Objective: minimise its own departure delay and its passengers' walking
distance. It requests a stand (CFP via the coordinator) when it enters the
look-ahead window and re-requests after a disruption that affects it.
"""

from __future__ import annotations

from typing import Any

from aerosync.sim.flights import Flight
from aerosync.sim.message_bus import MessageBus, MsgType


class FlightAgent:
    """Lightweight agent wrapper around a :class:`Flight`."""

    def __init__(self, flight: Flight) -> None:
        self.flight = flight
        self.id = f"FLT:{flight.id}"
        self.eta = flight.sched_arr
        self.contract: str | None = None
        self.needs_request = True

    def observe_eta(self, bus: MessageBus, t: int, eta: int, coordinator_id: str) -> None:
        """New ETA information (delay revealed): tell the coordinator, re-request."""
        if eta != self.eta:
            self.eta = eta
            self.needs_request = True
            bus.send(t, MsgType.STATE_UPDATE, self.id, coordinator_id,
                     {"flight": self.flight.id, "eta": eta, "delay": eta - self.flight.sched_arr})

    def cfp(self, t: int, eta: int, current_stand: str | None) -> dict[str, Any]:
        """Build the call-for-proposals payload describing this flight."""
        f = self.flight
        return {
            "flight": f.id,
            "t": t,
            "eta": eta,
            "sched_dep": f.sched_dep,
            "min_turn": f.min_turn,
            "size": f.size,
            "pax": f.pax,
            "pref_terminal": f.pref_terminal,
            "current_stand": current_stand,
        }

    def utility(self, walk_m: float, dep_delay: float) -> float:
        """Flight's own (selfish) objective, used for reporting only."""
        return -(dep_delay * 2.0 + walk_m * 0.1 * self.flight.pax / 150.0)
