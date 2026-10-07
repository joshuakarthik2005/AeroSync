"""A* search (and Dijkstra as the h = 0 special case) on a weighted graph.

Edge costs are supplied by a callback so that ground-vehicle agents can make
them congestion-aware and time-dependent, and can block edges (``None``).
The heuristic must never over-estimate the remaining cost (admissible); for the
apron graph we use straight-line distance divided by the vehicle's free-flow
speed, which is a lower bound because every edge is a straight segment and the
congestion multiplier is always >= 1.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

Graph = Mapping[str, Sequence[tuple[str, float]]]
CostFn = Callable[[str, str, float], float | None]
HeuristicFn = Callable[[str], float]


@dataclass
class SearchResult:
    """Outcome of a path search."""

    path: list[str]
    cost: float
    expanded: int
    edge_costs: list[tuple[str, str, float, float]] = field(default_factory=list)
    """Per edge: ``(u, v, length_m, cost)``."""

    @property
    def found(self) -> bool:
        return bool(self.path)

    @property
    def length_m(self) -> float:
        return sum(e[2] for e in self.edge_costs)


def _default_cost(_u: str, _v: str, length: float) -> float:
    return length


def astar(graph: Graph, start: str, goal: str, cost_fn: CostFn | None = None,
          heuristic: HeuristicFn | None = None) -> SearchResult:
    """Find a least-cost path from ``start`` to ``goal``.

    Pseudocode::

        open <- {start}, g[start] = 0
        while open not empty:
            n <- argmin_{n in open} g[n] + h(n)
            if n == goal: return reconstruct(n)
            for (m, len) in neighbours(n):
                c <- cost(n, m, len); skip if blocked
                if g[n] + c < g[m]: g[m] <- g[n] + c; parent[m] <- n; push m

    Returns an empty path with ``cost = inf`` if the goal is unreachable.
    """
    cost_fn = cost_fn or _default_cost
    h = heuristic or (lambda _n: 0.0)
    if start == goal:
        return SearchResult(path=[start], cost=0.0, expanded=0)
    counter = itertools.count()
    g: dict[str, float] = {start: 0.0}
    parent: dict[str, tuple[str, float, float]] = {}
    open_heap: list[tuple[float, int, str]] = [(h(start), next(counter), start)]
    closed: set[str] = set()
    expanded = 0
    while open_heap:
        _f, _, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        if node == goal:
            return _reconstruct(parent, start, goal, g[goal], expanded)
        closed.add(node)
        expanded += 1
        for nbr, length in graph.get(node, ()):
            if nbr in closed:
                continue
            c = cost_fn(node, nbr, length)
            if c is None:
                continue
            ng = g[node] + c
            if ng < g.get(nbr, float("inf")) - 1e-12:
                g[nbr] = ng
                parent[nbr] = (node, length, c)
                heapq.heappush(open_heap, (ng + h(nbr), next(counter), nbr))
    return SearchResult(path=[], cost=float("inf"), expanded=expanded)


def dijkstra(graph: Graph, start: str, goal: str, cost_fn: CostFn | None = None) -> SearchResult:
    """Uniform-cost search: A* with the zero heuristic."""
    return astar(graph, start, goal, cost_fn=cost_fn, heuristic=None)


def dijkstra_all(graph: Graph, source: str, cost_fn: CostFn | None = None) -> dict[str, float]:
    """Exact cost-to-go from ``source`` to every node (used to verify admissibility)."""
    cost_fn = cost_fn or _default_cost
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist.get(n, float("inf")):
            continue
        for m, length in graph.get(n, ()):
            c = cost_fn(n, m, length)
            if c is None:
                continue
            nd = d + c
            if nd < dist.get(m, float("inf")):
                dist[m] = nd
                heapq.heappush(heap, (nd, m))
    return dist


def _reconstruct(parent: dict[str, tuple[str, float, float]], start: str, goal: str,
                 cost: float, expanded: int) -> SearchResult:
    path = [goal]
    edges: list[tuple[str, str, float, float]] = []
    node = goal
    while node != start:
        prev, length, c = parent[node]
        edges.append((prev, node, length, c))
        path.append(prev)
        node = prev
    path.reverse()
    edges.reverse()
    return SearchResult(path=path, cost=cost, expanded=expanded, edge_costs=edges)
