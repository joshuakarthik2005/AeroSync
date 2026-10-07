"""The three controllers compared in the experiments.

* ``fcfs``     - first come first served: each flight, when it requests a stand,
  gets the first feasible stand in fixed order (G1..G12, R1..R4). No
  coordination, no repair; first-available vehicle on the static route.
* ``greedy``   - nearest feasible stand to the preferred terminal exit. No
  negotiation, no repair; first-available vehicle on the static route.
* ``aerosync`` - CSP initial schedule + Contract Net negotiation + min-conflicts
  repair + congestion-aware A* routing and best-arrival vehicle selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aerosync.agents.coordinator import CoordinatorAgent
from aerosync.agents.vehicle_agent import GroundVehicleAgent, Route
from aerosync.ai.model import overlaps
from aerosync.config import BUFFER_MIN, LOOKAHEAD_MIN, MAX_HOLD_MIN, STEP_MIN
from aerosync.sim.message_bus import MsgType

if TYPE_CHECKING:
    from aerosync.sim.environment import FlightState, ServiceJob, Simulation

CONTROLLER_NAMES = ["fcfs", "greedy", "aerosync"]


class Controller:
    """Hook interface between the environment and a decision-making policy."""

    name = "base"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def setup(self, sim: Simulation) -> None:
        """Called once before the first step."""

    def on_step(self, sim: Simulation, t: int, changes: list[dict[str, Any]]) -> None:
        """Called every step after events/ETA updates and before execution."""

    def assign_unplanned(self, sim: Simulation, fs: FlightState, t: int) -> None:
        """A flight arrived without any stand: assign one now."""

    def resolve_arrival(self, sim: Simulation, fs: FlightState, t: int) -> str | None:
        """Planned stand unavailable at arrival: return an alternative or ``None`` to wait."""
        return None

    def choose_tow_target(self, sim: Simulation, fs: FlightState, t: int) -> str | None:
        return None

    def choose_vehicle(self, sim: Simulation, kind: str, stand: str, need: int
                       ) -> tuple[GroundVehicleAgent, Route, float]:
        raise NotImplementedError

    def redispatch(self, sim: Simulation, job: ServiceJob, broken: GroundVehicleAgent, t: int
                   ) -> tuple[GroundVehicleAgent, Route, float]:
        raise NotImplementedError


# ---------------------------------------------------------------- baselines
class BaselineController(Controller):
    """Shared logic of the two non-coordinating baselines."""

    def stand_order(self, sim: Simulation, fs: FlightState) -> list[str]:
        return list(sim.airport.stands)

    def _fits(self, sim: Simulation, stand: str, fs: FlightState, start: int, end: int) -> bool:
        if sim.is_closed(stand, start, end) or sim.is_comms_down(stand):
            return False
        if sim.last_departure[stand] + BUFFER_MIN > start:
            return False
        return not any(overlaps(start, end, s, e, BUFFER_MIN)
                       for s, e, _ in sim.committed_intervals(stand, exclude=fs.id))

    def pick(self, sim: Simulation, fs: FlightState, t: int) -> tuple[str, int]:
        f = fs.flight
        eta = max(fs.eta, t)
        order = [s for s in self.stand_order(sim, fs) if sim.airport.stands[s].accepts(f.size)]
        for s in order:
            if self._fits(sim, s, fs, eta, f.planned_departure(eta)):
                return s, eta - fs.eta
        best: tuple[int, int, str] | None = None
        for idx, s in enumerate(order):
            for h in range(STEP_MIN, MAX_HOLD_MIN + 1, STEP_MIN):
                st = eta + h
                if self._fits(sim, s, fs, st, f.planned_departure(st)):
                    if best is None or (h, idx) < (best[0], best[1]):
                        best = (h, idx, s)
                    break
        if best is not None:
            return best[2], eta + best[0] - fs.eta
        return order[0], eta - fs.eta

    def _assign(self, sim: Simulation, fs: FlightState, t: int, reason: str) -> None:
        stand, hold = self.pick(sim, fs, t)
        fs.assign(t, stand, reason)
        fs.planned_hold = hold
        fs.requested = True
        fs.published = True
        sim.bus.send(t, MsgType.AWARD, self.name.upper(), f"FLT:{fs.id}",
                     {"flight": fs.id, "stand": stand, "start": fs.planned_start(),
                      "why": reason})
        sim.decisions.append({"t": t, "kind": "assign", "flight": fs.id,
                              "inputs": {"eta": fs.eta}, "output": {"stand": stand, "hold": hold},
                              "reasoning": f"{self.name}: {reason} -> {stand} (hold {hold} min)"})

    def on_step(self, sim: Simulation, t: int, changes: list[dict[str, Any]]) -> None:
        for fs in sim.states.values():
            if (not fs.requested and fs.on_block is None
                    and fs.eta - LOOKAHEAD_MIN <= t):
                self._assign(sim, fs, t, self.reason)

    reason = "first feasible stand"

    def assign_unplanned(self, sim: Simulation, fs: FlightState, t: int) -> None:
        self._assign(sim, fs, t, self.reason)

    def _available_now(self, sim: Simulation, fs: FlightState, t: int) -> str | None:
        for s in self.stand_order(sim, fs):
            if (sim.airport.stands[s].accepts(fs.flight.size) and sim.can_block_on(s, t)
                    and not sim.is_comms_down(s)):
                return s
        return None

    def resolve_arrival(self, sim: Simulation, fs: FlightState, t: int) -> str | None:
        # No repair: only a closed stand forces a change; a busy stand means waiting.
        if fs.stand is not None and sim.is_closed(fs.stand, t):
            return self._available_now(sim, fs, t)
        return None

    def choose_tow_target(self, sim: Simulation, fs: FlightState, t: int) -> str | None:
        return self._available_now(sim, fs, t)

    def choose_vehicle(self, sim: Simulation, kind: str, stand: str, need: int
                       ) -> tuple[GroundVehicleAgent, Route, float]:
        fleet = [v for v in sim.vehicles.values() if v.kind == kind]
        veh = min(fleet, key=lambda v: (v.available_at(sim.t), v.id))
        goal = stand
        a = veh.available_at(sim.t)
        r0 = veh.plan_static(sim.airport, goal, a, sim.edge_multiplier, sim.edge_cleared)
        depart = max(float(a), need - r0.minutes)
        route = veh.plan_static(sim.airport, goal, int(depart), sim.edge_multiplier,
                                sim.edge_cleared)
        return veh, route, depart

    def redispatch(self, sim: Simulation, job: ServiceJob, broken: GroundVehicleAgent, t: int
                   ) -> tuple[GroundVehicleAgent, Route, float]:
        # Baselines wait for the broken vehicle to be repaired.
        depart = float(broken.broken_until)
        route = broken.plan_static(sim.airport, job.stand, int(depart), sim.edge_multiplier,
                                   sim.edge_cleared, start=job.route.path[0] if job.route.path
                                   else None)
        return broken, route, depart


class FCFSController(BaselineController):
    name = "fcfs"
    reason = "first feasible stand (fixed order)"


class GreedyController(BaselineController):
    name = "greedy"
    reason = "nearest feasible stand to preferred terminal"

    def stand_order(self, sim: Simulation, fs: FlightState) -> list[str]:
        term = fs.flight.pref_terminal
        return sorted(sim.airport.stands,
                      key=lambda s: (sim.airport.walk_distance(s, term), s))


# ----------------------------------------------------------------- aerosync
class AeroSyncController(Controller):
    """CSP + Contract Net + min-conflicts + A* (the proposed system)."""

    name = "aerosync"

    def setup(self, sim: Simulation) -> None:
        self.coord = CoordinatorAgent(sim, self.seed)
        self.coord.initial_plan()

    def on_step(self, sim: Simulation, t: int, changes: list[dict[str, Any]]) -> None:
        self.coord.step(t, changes)

    def assign_unplanned(self, sim: Simulation, fs: FlightState, t: int) -> None:
        self.coord.negotiate(fs, t, context="unplanned")

    def resolve_arrival(self, sim: Simulation, fs: FlightState, t: int) -> str | None:
        res = self.coord.negotiate(fs, t, context="arrival conflict", only_now=True)
        return res.winner.bidder if res and res.winner else None

    def choose_tow_target(self, sim: Simulation, fs: FlightState, t: int) -> str | None:
        res = self.coord.negotiate(fs, t, context="tow after closure", only_now=True)
        return res.winner.bidder if res and res.winner else None

    def _best(self, sim: Simulation, fleet: list[GroundVehicleAgent], stand: str, need: int
              ) -> tuple[GroundVehicleAgent, Route, float]:
        best: tuple[float, float, str, GroundVehicleAgent, Route, float] | None = None
        for v in fleet:
            a = v.available_at(sim.t)
            r0 = v.plan_astar(sim.airport, stand, a, sim.edge_multiplier)
            if not r0.path:
                continue
            depart = max(float(a), need - r0.minutes)
            r = v.plan_astar(sim.airport, stand, int(depart), sim.edge_multiplier)
            if not r.path:
                continue
            key = (depart + r.minutes, r.minutes, v.id)
            if best is None or key < best[:3]:
                best = (key[0], key[1], key[2], v, r, depart)
        if best is None:  # every route blocked: fall back to waiting on static path
            v = fleet[0]
            a = v.available_at(sim.t)
            r = v.plan_static(sim.airport, stand, a, sim.edge_multiplier, sim.edge_cleared)
            return v, r, float(a)
        return best[3], best[4], best[5]

    def choose_vehicle(self, sim: Simulation, kind: str, stand: str, need: int
                       ) -> tuple[GroundVehicleAgent, Route, float]:
        fleet = [v for v in sim.vehicles.values() if v.kind == kind]
        return self._best(sim, fleet, stand, need)

    def redispatch(self, sim: Simulation, job: ServiceJob, broken: GroundVehicleAgent, t: int
                   ) -> tuple[GroundVehicleAgent, Route, float]:
        fleet = [v for v in sim.vehicles.values() if v.kind == job.kind and v.id != broken.id]
        if not fleet:
            fleet = [broken]
        return self._best(sim, fleet, job.stand, t)


def make_controller(name: str, seed: int = 42) -> Controller:
    """Factory: ``fcfs`` | ``greedy`` | ``aerosync``."""
    table = {"fcfs": FCFSController, "greedy": GreedyController, "aerosync": AeroSyncController}
    if name not in table:
        raise ValueError(f"unknown controller {name!r}; choose from {CONTROLLER_NAMES}")
    return table[name](seed)
