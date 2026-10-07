"""Discrete-time airport environment.

Environment characteristics (Russell & Norvig):

* **Partially observable**: arrival delays are hidden until ``reveal_at``;
  gate agents only see their own and their neighbours' schedules.
* **Stochastic**: seeded random delays, breakdowns and closures.
* **Sequential**: every assignment constrains later ones.
* **Dynamic**: the world changes while agents deliberate (5-minute steps).
* **Multi-agent, cooperative**: flight, gate, vehicle and coordinator agents
  share the airport-wide objective.

The environment is controller-agnostic: a controller (fcfs, greedy or aerosync)
decides gate assignments and vehicle dispatch through a small hook interface;
the environment executes those decisions, enforces the physics (an aircraft can
only block on at a free, open stand) and records everything for the metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aerosync.agents.flight_agent import FlightAgent
from aerosync.agents.vehicle_agent import GroundVehicleAgent, Route
from aerosync.ai.model import Block, Task
from aerosync.config import (
    BUFFER_MIN,
    LOOKAHEAD_MIN,
    MOVEMENT_CONGESTION,
    MOVEMENT_WINDOW_MIN,
    SERVICE_MIN,
    STEP_MIN,
    fmt_clock,
    round_up_step,
)
from aerosync.sim.airport import Airport, build_airport
from aerosync.sim.events import Event
from aerosync.sim.flights import Flight
from aerosync.sim.message_bus import MessageBus, MsgType
from aerosync.sim.scenarios import Scenario

if TYPE_CHECKING:
    from aerosync.controllers import Controller

INF = 10**9
DRAIN_STEPS = 96  # extra steps allowed after the window so every flight departs
TOW_SETUP_MIN = 10


@dataclass
class ServiceJob:
    """A ground-vehicle trip to serve a flight."""

    kind: str
    flight: str
    vehicle: str
    stand: str
    depart: float
    arrive: float
    done: float
    route: Route

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "flight": self.flight, "vehicle": self.vehicle,
            "stand": self.stand, "depart": round(self.depart, 2),
            "arrive": round(self.arrive, 2), "done": round(self.done, 2),
            "trip_min": round(self.arrive - self.depart, 2),
            "path": self.route.path, "length_m": round(self.route.length_m, 1),
            "expanded": self.route.expanded,
        }


@dataclass
class FlightState:
    """Runtime state of a flight inside the environment."""

    flight: Flight
    eta: int
    revealed: bool = False
    stand: str | None = None
    planned_hold: int = 0
    requested: bool = False
    published: bool = False
    on_block: int | None = None
    off_block: int | None = None
    departed: bool = False
    history: list[tuple[int, str, str]] = field(default_factory=list)
    reassignments: int = 0
    conflict: bool = False
    towed: int = 0
    jobs: list[ServiceJob] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.flight.id

    @property
    def at_stand(self) -> bool:
        return self.on_block is not None and not self.departed

    @property
    def mutable(self) -> bool:
        return self.on_block is None

    def planned_start(self) -> int:
        return self.eta + self.planned_hold

    def planned_interval(self) -> tuple[int, int]:
        if self.on_block is not None:
            return self.on_block, self.off_block if self.off_block is not None else INF
        s = self.planned_start()
        return s, self.flight.planned_departure(s)

    def task(self, now: int | None = None) -> Task:
        eta = self.eta if now is None else max(self.eta, now)
        f = self.flight
        return Task(id=f.id, size=f.size, eta=eta, sched_dep=f.sched_dep, min_turn=f.min_turn,
                    pref_terminal=f.pref_terminal, pax=f.pax)

    def assign(self, t: int, stand: str, reason: str) -> bool:
        """Record a (re)assignment; returns True if the stand actually changed.

        A change only counts as a *reassignment* once the gate has been announced
        to passengers (``published``, at T-60 for every controller).
        """
        if stand == self.stand:
            return False
        if self.stand is not None and self.published:
            self.reassignments += 1
        self.stand = stand
        self.history.append((t, stand, reason))
        return True


@dataclass
class SimResult:
    """Everything recorded during a run (serialisable)."""

    scenario: str
    controller: str
    seed: int
    start_hour: int
    steps: int
    window_min: int
    end_t: int
    stands: list[str]
    flights: list[dict[str, Any]]
    jobs: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    message_counts: dict[str, int]
    decisions: list[dict[str, Any]]
    auctions: dict[str, list[dict[str, Any]]]
    repairs: list[dict[str, Any]]
    closures: list[dict[str, Any]]
    events: list[dict[str, Any]]
    compute_ms: float
    counters: dict[str, int]
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_messages: bool = True) -> dict[str, Any]:
        d = dict(self.__dict__)
        if not include_messages:
            d["messages"] = []
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SimResult:
        return SimResult(**d)

    def flight(self, fid: str) -> dict[str, Any] | None:
        return next((f for f in self.flights if f["id"] == fid), None)

    def clock(self, t: int) -> str:
        return fmt_clock(t, self.start_hour)


class Simulation:
    """Runs one scenario under one controller."""

    def __init__(self, scenario: Scenario, controller: Controller,
                 extra_events: list[Event] | None = None, airport: Airport | None = None
                 ) -> None:
        self.scenario = scenario
        self.controller = controller
        self.airport = airport or build_airport()
        self.bus = MessageBus()
        self.t = 0
        self.step_index = 0
        self.states: dict[str, FlightState] = {
            f.id: FlightState(flight=f, eta=f.sched_arr)
            for f in sorted(scenario.flights, key=lambda f: (f.sched_arr, f.id))
        }
        self.flight_agents = {fid: FlightAgent(s.flight) for fid, s in self.states.items()}
        self.events = sorted(list(scenario.events) + list(extra_events or []),
                             key=lambda e: (e.at, e.kind))
        self.closures: list[Block] = []
        self.comms_down: dict[str, int] = {}
        self.weather: list[tuple[int, int, float]] = []
        self.blocked_edges: dict[tuple[str, str], int] = {}
        self.occupant: dict[str, str | None] = {s: None for s in self.airport.stands}
        self.last_departure: dict[str, int] = {s: -INF for s in self.airport.stands}
        self.movements: dict[str, dict[str, list[int]]] = {s: {} for s in self.airport.stands}
        self.road_to_stand = {st.road_node: sid for sid, st in self.airport.stands.items()}
        self.vehicles: dict[str, GroundVehicleAgent] = {}
        for kind in ("tug", "fuel", "bus"):
            for i in range(1, scenario.fleet.get(kind, 0) + 1):
                vid = f"{kind.upper()}{i}"
                self.vehicles[vid] = GroundVehicleAgent(vid, kind, "DEPOT")
        self.jobs: list[ServiceJob] = []
        self.decisions: list[dict[str, Any]] = []
        self.auctions: dict[str, list[dict[str, Any]]] = {}
        self.repairs: list[dict[str, Any]] = []
        self.event_log: list[dict[str, Any]] = []
        self.compute_ms = 0.0
        self.counters = {"conflicts": 0, "towing": 0, "dispatches": 0, "redispatches": 0,
                         "repairs": 0, "auctions": 0}
        self._setup_done = False

    # ================================================================ queries
    @property
    def coordinator_id(self) -> str:
        return "COORD"

    def clock(self, t: int | None = None) -> str:
        return fmt_clock(self.t if t is None else t, self.scenario.start_hour)

    def is_closed(self, stand: str, t: int, until: int | None = None) -> bool:
        end = t + 1 if until is None else until
        return any(b.stand == stand and t < b.end and b.start < end for b in self.closures)

    def closure_blocks(self) -> list[Block]:
        return list(self.closures)

    def is_comms_down(self, stand: str) -> bool:
        return stand in self.comms_down

    def can_block_on(self, stand: str, t: int) -> bool:
        """Physical check: stand open, empty, and buffer since last departure respected."""
        return (not self.is_closed(stand, t) and self.occupant[stand] is None
                and self.last_departure[stand] + BUFFER_MIN <= t)

    def weather_factor(self, t: int) -> float:
        f = self.scenario.weather_factor
        for s, e, w in self.weather:
            if s <= t < e:
                f *= w
        return max(1.0, f)

    def edge_key(self, u: str, v: str) -> tuple[str, str]:
        return (u, v) if u < v else (v, u)

    def edge_multiplier(self, u: str, v: str, t: int) -> float | None:
        """Congestion multiplier (>= 1) for an apron edge, or ``None`` if blocked."""
        if self.blocked_edges.get(self.edge_key(u, v), -1) > t:
            return None
        m = self.weather_factor(t)
        moves = 0
        for node in (u, v):
            sid = self.road_to_stand.get(node)
            if sid is None:
                continue
            for times in self.movements[sid].values():
                moves += sum(1 for x in times if abs(x - t) <= MOVEMENT_WINDOW_MIN)
        return m * (1.0 + MOVEMENT_CONGESTION * moves)

    def edge_cleared(self, u: str, v: str, t: int) -> int:
        until = self.blocked_edges.get(self.edge_key(u, v), -1)
        return until if until > t else t

    def committed_intervals(self, stand: str, exclude: str | None = None
                            ) -> list[tuple[int, int, str]]:
        """Intervals of flights currently assigned to ``stand`` (planned or actual)."""
        out = []
        for fs in self.states.values():
            if fs.stand != stand or fs.departed or fs.id == exclude:
                continue
            s, e = fs.planned_interval()
            out.append((s, e, fs.id))
        return sorted(out)

    # ================================================================ running
    def setup(self) -> None:
        if self._setup_done:
            return
        t0 = time.perf_counter()
        self.controller.setup(self)
        self.compute_ms += (time.perf_counter() - t0) * 1000.0
        self._setup_done = True

    def all_done(self) -> bool:
        return all(fs.departed for fs in self.states.values())

    def run(self, steps: int | None = None, drain: bool = True) -> SimResult:
        """Run ``steps`` steps (default: the scenario window), then drain until all depart."""
        self.setup()
        n = self.scenario.steps if steps is None else steps
        for _ in range(n):
            self.step()
        if drain and steps is None:
            extra = 0
            while not self.all_done() and extra < DRAIN_STEPS:
                self.step()
                extra += 1
        return self.result()

    def step(self) -> None:
        """Advance the world by one timestep."""
        self.setup()
        t = self.t
        changes: list[dict[str, Any]] = []
        for ev in [e for e in self.events if e.at == self.step_index]:
            changes.append(self._apply_event(ev, t))
        for fs in self.states.values():
            f = fs.flight
            if (not fs.revealed and f.arr_delay > 0 and f.reveal_at <= t
                    and fs.on_block is None):
                fs.revealed = True
                fs.eta = f.actual_arr
                self._withdraw_if_slipped(fs, t)
                self.flight_agents[fs.id].observe_eta(self.bus, t, fs.eta, self.coordinator_id)
                changes.append({"kind": "delay", "flight": fs.id, "eta": fs.eta})
        for gate, until in list(self.comms_down.items()):
            if until <= t:
                del self.comms_down[gate]
                self.bus.down.discard(gate)
                changes.append({"kind": "comms_restored", "gate": gate})

        t0 = time.perf_counter()
        self.controller.on_step(self, t, changes)
        self.compute_ms += (time.perf_counter() - t0) * 1000.0

        # Departures.
        for fs in self.states.values():
            if fs.at_stand and fs.off_block is not None and fs.off_block <= t:
                fs.departed = True
                fs.off_block = t if fs.off_block < t else fs.off_block
                self.occupant[fs.stand] = None
                self.last_departure[fs.stand] = t
                self.bus.send(t, MsgType.STATE_UPDATE, f"FLT:{fs.id}", fs.stand,
                              {"flight": fs.id, "status": "off-block"})
        # Arrivals.
        arriving = sorted((fs for fs in self.states.values()
                           if fs.on_block is None and fs.flight.actual_arr <= t
                           and fs.eta <= t), key=lambda s: (s.eta, s.id))
        for fs in arriving:
            if fs.stand is None:
                t0 = time.perf_counter()
                self.controller.assign_unplanned(self, fs, t)
                self.compute_ms += (time.perf_counter() - t0) * 1000.0
                if fs.stand is None:
                    continue
            if self.can_block_on(fs.stand, t):
                self._block_on(fs, t)  # stand already free: no need to sit out a planned hold
                continue
            if t < fs.planned_start():
                continue  # planned hold, not a conflict
            if not fs.conflict:
                fs.conflict = True
                self.counters["conflicts"] += 1
            t0 = time.perf_counter()
            alt = self.controller.resolve_arrival(self, fs, t)
            self.compute_ms += (time.perf_counter() - t0) * 1000.0
            if alt and self.can_block_on(alt, t):
                fs.assign(t, alt, "re-assigned at arrival")
                self._block_on(fs, t)
        self.t += STEP_MIN
        self.step_index += 1

    # ================================================================ events
    def _apply_event(self, ev: Event, t: int) -> dict[str, Any]:
        p = ev.params
        desc = ev.describe()
        self.event_log.append({"t": t, "step": ev.at, "kind": ev.kind, "params": dict(p),
                               "description": desc})
        if ev.kind == "gate_closure":
            gate = p["gate"]
            dur = p.get("duration")
            end = INF if dur is None else t + int(dur)
            self.closures.append(Block(gate, t, end, label=f"{gate} closed", is_closure=True))
            self.bus.multicast(t, MsgType.FAULT, gate,
                               [self.coordinator_id] + self.airport.neighbours.get(gate, []),
                               {"description": desc, "gate": gate})
            occupant = self.occupant.get(gate)
            if occupant:
                self._tow(self.states[occupant], t)
        elif ev.kind == "flight_delay":
            fs = self.states[p["flight"]]
            fs.flight.arr_delay += int(p["minutes"])
            if fs.on_block is None:
                fs.revealed = True
                fs.eta = fs.flight.actual_arr
                self._withdraw_if_slipped(fs, t)
                self.flight_agents[fs.id].observe_eta(self.bus, t, fs.eta, self.coordinator_id)
            self.bus.send(t, MsgType.FAULT, f"FLT:{fs.id}", self.coordinator_id,
                          {"description": desc, "flight": fs.id})
        elif ev.kind == "vehicle_breakdown":
            vid = p["vehicle"]
            veh = self.vehicles[vid]
            dur = int(p.get("duration", 60))
            veh.broken_until = t + dur
            if len(veh.last_path) >= 2:
                self.blocked_edges[self.edge_key(veh.last_path[-2], veh.last_path[-1])] = t + dur
            self.bus.send(t, MsgType.FAULT, vid, self.coordinator_id,
                          {"description": desc, "vehicle": vid})
            self._handle_breakdown(veh, t)
        elif ev.kind == "comms_loss":
            gate = p["gate"]
            self.bus.send(t, MsgType.FAULT, gate, self.coordinator_id,
                          {"description": desc, "gate": gate})
            self.comms_down[gate] = t + int(p.get("duration", 30))
            self.bus.down.add(gate)
        elif ev.kind == "weather":
            self.weather.append((t, t + int(p.get("duration", 60)), float(p.get("factor", 1.4))))
            self.bus.send(t, MsgType.FAULT, "MET", self.coordinator_id, {"description": desc})
        return {"kind": ev.kind, **p, "t": t}

    def _withdraw_if_slipped(self, fs: FlightState, t: int) -> None:
        """A delay that pushes the ETA back outside the look-ahead window withdraws the
        gate announcement (it is re-announced at the new T-60); applies to every controller."""
        if fs.eta - LOOKAHEAD_MIN > t:
            fs.published = False

    def _tow(self, fs: FlightState, t: int) -> None:
        old = fs.stand
        assert old is not None
        t0 = time.perf_counter()
        target = self.controller.choose_tow_target(self, fs, t)
        self.compute_ms += (time.perf_counter() - t0) * 1000.0
        if target is None or not self.can_block_on(target, t):
            fs.stand = old
            return
        self.occupant[old] = None
        self.last_departure[old] = t
        if fs.stand != target:
            fs.assign(t, target, f"towed: {old} closed")
        fs.towed += 1
        self.counters["towing"] += 1
        self.occupant[target] = fs.id
        tug_node = self.airport.stands[old].road_node
        job = self._dispatch(fs, "tug", t, dest=old)
        tow_route = self.vehicles[job.vehicle].plan_astar(
            self.airport, self.airport.stands[target].road_node, t, self.edge_multiplier,
            start=tug_node)
        ready = job.arrive + TOW_SETUP_MIN + tow_route.minutes
        self.vehicles[job.vehicle].free_at = int(round_up_step(ready))
        self.vehicles[job.vehicle].location = target
        fs.off_block = max(fs.off_block or 0, round_up_step(ready + SERVICE_MIN["tug"]))
        self.movements[target][fs.id] = [t, fs.off_block]

    # ================================================================ services
    def _block_on(self, fs: FlightState, t: int) -> None:
        stand = fs.stand
        assert stand is not None
        fs.on_block = t
        self.occupant[stand] = fs.id
        self.bus.send(t, MsgType.STATE_UPDATE, f"FLT:{fs.id}", stand,
                      {"flight": fs.id, "status": "on-block", "hold": t - fs.flight.actual_arr})
        f = fs.flight
        needs: list[tuple[str, int]] = [("fuel", t + 5)]
        if self.airport.stands[stand].is_remote:
            needs.append(("bus", t))
        needs.append(("tug", t + f.min_turn - 10))
        for kind, need in needs:
            self._dispatch(fs, kind, need)
        self._update_off_block(fs)

    def _dispatch(self, fs: FlightState, kind: str, need: int, dest: str | None = None
                  ) -> ServiceJob:
        stand = dest or fs.stand
        assert stand is not None
        t0 = time.perf_counter()
        veh, route, depart = self.controller.choose_vehicle(self, kind, stand, need)
        self.compute_ms += (time.perf_counter() - t0) * 1000.0
        arrive = depart + route.minutes
        done = arrive + SERVICE_MIN[kind]
        job = ServiceJob(kind=kind, flight=fs.id, vehicle=veh.id, stand=stand, depart=depart,
                         arrive=arrive, done=done, route=route)
        veh.free_at = int(round_up_step(done))
        veh.location = stand
        veh.last_path = route.path or [stand]
        fs.jobs.append(job)
        self.jobs.append(job)
        self.counters["dispatches"] += 1
        return job

    def _update_off_block(self, fs: FlightState) -> None:
        f = fs.flight
        assert fs.on_block is not None
        ready = fs.on_block + f.min_turn
        for job in fs.jobs:
            if job.stand != fs.stand:
                continue
            ready = max(ready, job.done)
        off = round_up_step(max(f.sched_dep, ready))
        fs.off_block = max(off, fs.off_block or 0)
        tug = next((j for j in fs.jobs if j.kind == "tug" and j.stand == fs.stand), None)
        if tug is not None:
            self.vehicles[tug.vehicle].free_at = max(self.vehicles[tug.vehicle].free_at,
                                                     fs.off_block)
        assert fs.stand is not None
        self.movements[fs.stand][fs.id] = [fs.on_block, fs.off_block]

    def _handle_breakdown(self, veh: GroundVehicleAgent, t: int) -> None:
        affected = [j for j in self.jobs if j.vehicle == veh.id and j.arrive > t]
        for job in affected:
            fs = self.states[job.flight]
            if fs.departed:
                continue
            t0 = time.perf_counter()
            new_veh, route, depart = self.controller.redispatch(self, job, veh, t)
            self.compute_ms += (time.perf_counter() - t0) * 1000.0
            job.vehicle = new_veh.id
            job.route = route
            job.depart = depart
            job.arrive = depart + route.minutes
            job.done = job.arrive + SERVICE_MIN[job.kind]
            new_veh.free_at = int(round_up_step(job.done))
            new_veh.location = job.stand
            new_veh.last_path = route.path or [job.stand]
            self.counters["redispatches"] += 1
            if fs.on_block is not None:
                self._update_off_block(fs)

    # ================================================================ results
    def result(self) -> SimResult:
        from aerosync.metrics.engine import compute_metrics

        flights = []
        for fs in self.states.values():
            f = fs.flight
            stand = fs.stand
            on, off = fs.on_block, fs.off_block
            if not fs.departed and on is None:
                off = None
            walk = self.airport.walk_distance(stand, f.pref_terminal) if stand else None
            st = self.airport.stands[stand] if stand else None
            flights.append({
                "id": f.id, "airline": f.airline, "size": f.size, "pax": f.pax,
                "pref_terminal": f.pref_terminal, "sched_arr": f.sched_arr,
                "sched_dep": f.sched_dep, "arr_delay": f.arr_delay, "reveal_at": f.reveal_at,
                "eta": fs.eta, "stand": stand, "on_block": on, "off_block": off,
                "departed": fs.departed,
                "hold": (on - f.actual_arr) if on is not None else None,
                "dep_delay": (off - f.sched_dep) if off is not None else None,
                "walk_m": walk, "remote": bool(st and st.is_remote),
                "terminal_mismatch": bool(st and not st.is_remote
                                          and st.terminal != f.pref_terminal),
                "reassignments": fs.reassignments, "conflict": fs.conflict,
                "towed": fs.towed, "history": [list(h) for h in fs.history],
            })
        closures = [{"stand": b.stand, "start": b.start, "end": min(b.end, self.t)}
                    for b in self.closures]
        res = SimResult(
            scenario=self.scenario.name,
            controller=self.controller.name,
            seed=self.scenario.seed,
            start_hour=self.scenario.start_hour,
            steps=self.scenario.steps,
            window_min=self.scenario.window_min,
            end_t=self.t,
            stands=list(self.airport.stands),
            flights=flights,
            jobs=[j.to_dict() for j in self.jobs],
            messages=[m.to_dict() for m in self.bus.log],
            message_counts=self.bus.counts(),
            decisions=self.decisions,
            auctions=self.auctions,
            repairs=self.repairs,
            closures=closures,
            events=self.event_log,
            compute_ms=round(self.compute_ms, 2),
            counters=dict(self.counters),
        )
        res.metrics = compute_metrics(res, self.airport)
        return res
