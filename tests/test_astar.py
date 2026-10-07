"""A*: optimal (cross-checked with Dijkstra) and admissible heuristic."""

from __future__ import annotations

import math
import random

import pytest

from aerosync.ai.astar import astar, dijkstra, dijkstra_all


def random_geometric_graph(rng: random.Random, n: int, extra: int):
    pos = {f"n{i}": (rng.uniform(0, 1000), rng.uniform(0, 1000)) for i in range(n)}
    names = list(pos)
    adj: dict[str, list[tuple[str, float]]] = {k: [] for k in names}

    def add(u: str, v: str) -> None:
        d = math.dist(pos[u], pos[v])
        adj[u].append((v, d))
        adj[v].append((u, d))

    for i in range(1, n):  # random spanning tree keeps the graph connected
        add(names[i], names[rng.randrange(i)])
    for _ in range(extra):
        u, v = rng.sample(names, 2)
        add(u, v)
    return pos, adj


@pytest.mark.parametrize("seed", range(25))
def test_astar_matches_dijkstra_on_random_graphs(seed):
    rng = random.Random(seed)
    pos, adj = random_geometric_graph(rng, 40, 60)
    mult = {(u, v): rng.uniform(1.0, 3.0) for u in adj for v, _ in adj[u]}

    def cost(u, v, length):
        return length * mult[(u, v)]

    for _ in range(5):
        s, g = rng.sample(list(adj), 2)
        a = astar(adj, s, g, cost_fn=cost, heuristic=lambda n, g=g: math.dist(pos[n], pos[g]))
        d = dijkstra(adj, s, g, cost_fn=cost)
        assert a.cost == pytest.approx(d.cost)
        assert a.path[0] == s and a.path[-1] == g
        assert a.expanded <= d.expanded
        assert sum(e[3] for e in a.edge_costs) == pytest.approx(a.cost)


@pytest.mark.parametrize("seed", range(10))
def test_euclidean_heuristic_admissible_on_random_graphs(seed):
    rng = random.Random(100 + seed)
    pos, adj = random_geometric_graph(rng, 30, 40)
    goal = rng.choice(list(adj))
    exact = dijkstra_all(adj, goal)  # undirected: cost-to-go == cost-from-goal
    for n in adj:
        assert math.dist(pos[n], pos[goal]) <= exact[n] + 1e-9


def test_blocked_edges_are_avoided():
    adj = {"A": [("B", 1.0), ("C", 5.0)], "B": [("A", 1.0), ("D", 1.0)],
           "C": [("A", 5.0), ("D", 5.0)], "D": [("B", 1.0), ("C", 5.0)]}

    def cost(u, v, length):
        return None if {u, v} == {"B", "D"} else length

    res = astar(adj, "A", "D", cost_fn=cost)
    assert res.path == ["A", "C", "D"] and res.cost == 10.0
    assert not astar(adj, "A", "Z").found


def test_apron_heuristic_is_admissible(airport):
    speed = 250.0
    for goal in ["G7", "R3", "HANGAR", "G1"]:
        true_cost = dijkstra_all(airport.adjacency, goal, cost_fn=lambda u, v, ln: ln / speed)
        for n in airport.nodes:
            assert airport.euclid(n, goal) / speed <= true_cost[n] + 1e-9


def test_apron_route_hangar_to_g7(airport):
    res = astar(airport.adjacency, "HANGAR", "G7",
                heuristic=lambda n: airport.euclid(n, "G7"))
    ref = dijkstra(airport.adjacency, "HANGAR", "G7")
    assert res.cost == pytest.approx(ref.cost)
    assert res.expanded <= ref.expanded
