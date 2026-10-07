"""Ground Vehicle Agent: tug, fuel truck or passenger bus.

The agent plans its own apron route with A* over the taxi/apron graph. Edge
costs are travel minutes ``length / speed * congestion(edge, t)`` and blocked
edges are removed, so it replans around breakdowns and congestion. The
heuristic ``euclid(n, goal) / speed`` is admissible because congestion >= 1.

A ``naive`` agent (used by the baseline controllers) follows the static
shortest-distance path and simply waits at blocked edges.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from aerosync.ai.astar import SearchResult, astar
from aerosync.config import VEHICLE_SPEED_M_PER_MIN
from aerosync.sim.airport import Airport

EdgeFn = Callable[[str, str, int], float | None]
"""``(u, v, t) -> congestion multiplier`` or ``None`` if the edge is blocked."""
BlockFn = Callable[[str, str, int], int]
"""``(u, v, t) -> minute the edge is cleared`` (``t`` if not blocked)."""


@dataclass
class Route:
    """A planned trip."""

    path: list[str]
    minutes: float
    length_m: float
    expanded: int
    base_minutes: float
    edges: list[dict[str, float | str]] = field(default_factory=list)


class GroundVehicleAgent:
    """One apron vehicle."""

    def __init__(self, vid: str, kind: str, location: str) -> None:
        self.id = vid
        self.kind = kind
        self.location = location
        self.free_at = 0
        self.broken_until = -1
        self.speed = VEHICLE_SPEED_M_PER_MIN[kind]
        self.last_path: list[str] = [location]

    def available_at(self, t: int) -> int:
        return max(self.free_at, self.broken_until, t)

    def plan_astar(self, airport: Airport, goal: str, t: int, edge_fn: EdgeFn,
                   start: str | None = None) -> Route:
        """Congestion-aware A* route from the current location (or ``start``)."""
        src = start or self.location
        speed = self.speed

        def cost(u: str, v: str, length: float) -> float | None:
            m = edge_fn(u, v, t)
            return None if m is None else length / speed * m

        def h(n: str) -> float:
            return airport.euclid(n, goal) / speed

        res = astar(airport.adjacency, src, goal, cost_fn=cost, heuristic=h)
        return self._to_route(res, speed)

    def plan_static(self, airport: Airport, goal: str, t: int, edge_fn: EdgeFn,
                    block_fn: BlockFn, start: str | None = None) -> Route:
        """Shortest-*distance* path, then actual travel time under current conditions."""
        src = start or self.location
        res = astar(airport.adjacency, src, goal,
                    heuristic=lambda n: airport.euclid(n, goal))
        clock = float(t)
        edges = []
        base = 0.0
        for u, v, length, _c in res.edge_costs:
            cleared = block_fn(u, v, int(clock))
            if cleared > clock:
                clock = float(cleared)
            m = edge_fn(u, v, int(clock))
            m = 1.0 if m is None else m
            dt = length / self.speed * m
            base += length / self.speed
            edges.append({"u": u, "v": v, "length_m": length, "minutes": round(dt, 2),
                          "congestion": round(m, 3)})
            clock += dt
        return Route(path=res.path, minutes=clock - t, length_m=res.length_m,
                     expanded=res.expanded, base_minutes=base, edges=edges)

    @staticmethod
    def _to_route(res: SearchResult, speed: float) -> Route:
        edges = []
        base = 0.0
        for u, v, length, c in res.edge_costs:
            b = length / speed
            base += b
            edges.append({"u": u, "v": v, "length_m": length, "minutes": round(c, 2),
                          "congestion": round(c / b, 3) if b else 1.0})
        return Route(path=res.path, minutes=res.cost, length_m=res.length_m,
                     expanded=res.expanded, base_minutes=base, edges=edges)
