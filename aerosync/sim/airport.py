"""Airport topology: stands, terminals, walking distances and the apron/taxi graph."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from aerosync.config import REMOTE_WALK_M


@dataclass(frozen=True)
class Stand:
    """A parking position: a contact gate (G*) or a remote stand (R*)."""

    id: str
    kind: str  # "gate" | "remote"
    terminal: str  # "T1" | "T2" | "REMOTE"
    wide: bool  # can host wide-body aircraft
    x: float
    y: float
    road_node: str  # apron road node the stand connects to

    @property
    def is_remote(self) -> bool:
        return self.kind == "remote"

    def accepts(self, size: str) -> bool:
        """Hard constraint: wide-body ('W') flights only at wide-capable stands."""
        return size == "N" or self.wide


@dataclass
class Airport:
    """Static airport model shared by every agent."""

    stands: dict[str, Stand]
    nodes: dict[str, tuple[float, float]]
    adjacency: dict[str, list[tuple[str, float]]]
    exits: dict[str, tuple[float, float]]
    walk: dict[str, dict[str, float]] = field(default_factory=dict)
    gate_walk: dict[str, dict[str, float]] = field(default_factory=dict)
    neighbours: dict[str, list[str]] = field(default_factory=dict)
    taxiway_nodes: frozenset[str] = frozenset()

    @property
    def stand_ids(self) -> list[str]:
        return list(self.stands)

    @property
    def contact_gates(self) -> list[str]:
        return [s for s, st in self.stands.items() if st.kind == "gate"]

    @property
    def remote_stands(self) -> list[str]:
        return [s for s, st in self.stands.items() if st.kind == "remote"]

    def edges(self) -> list[tuple[str, str, float]]:
        """Undirected edge list ``(u, v, length_m)`` with ``u < v``."""
        out = []
        for u, nbrs in self.adjacency.items():
            for v, length in nbrs:
                if u < v:
                    out.append((u, v, length))
        return out

    def euclid(self, a: str, b: str) -> float:
        (x1, y1), (x2, y2) = self.nodes[a], self.nodes[b]
        return math.hypot(x1 - x2, y1 - y2)

    def walk_distance(self, stand: str, terminal: str) -> float:
        """Passenger walking distance (m) from ``stand`` to the exit of ``terminal``."""
        return self.walk[stand][terminal]


GATE_SPACING = 120.0
T1_GATES_X = [60.0 + GATE_SPACING * i for i in range(6)]  # G1..G6
T2_GATES_X = [860.0 + GATE_SPACING * i for i in range(6)]  # G7..G12
WIDE_GATES = {"G3", "G4", "G9", "G10"}
CONNECTOR_M = 250.0  # inter-terminal walkway
PIER_M = 80.0  # gate bridge + pier walk


def build_airport() -> Airport:
    """Build the default 12-gate, 2-terminal, 4-remote-stand airport.

    Layout (metres, y grows downwards on the apron)::

        y=0     T1 exit (360)                 T2 exit (1160)
        y=100   G1 .. G6                      G7 .. G12
        y=220   A1 .. A6 --- C (760) ---      A7 .. A12        apron road
        y=320                DEPOT
        y=420   TW0 .. TW6                                     service taxiway
        y=600   RR1 .. RR4                                     remote road
        y=720   R1 .. R4                                       remote stands
    """
    nodes: dict[str, tuple[float, float]] = {}
    stands: dict[str, Stand] = {}
    adj: dict[str, list[tuple[str, float]]] = {}

    def add_edge(u: str, v: str) -> None:
        (x1, y1), (x2, y2) = nodes[u], nodes[v]
        length = round(math.hypot(x1 - x2, y1 - y2), 1)
        adj.setdefault(u, []).append((v, length))
        adj.setdefault(v, []).append((u, length))

    for i, x in enumerate(T1_GATES_X + T2_GATES_X, start=1):
        gid, aid = f"G{i}", f"A{i}"
        nodes[gid] = (x, 100.0)
        nodes[aid] = (x, 220.0)
        stands[gid] = Stand(
            id=gid,
            kind="gate",
            terminal="T1" if i <= 6 else "T2",
            wide=gid in WIDE_GATES,
            x=x,
            y=100.0,
            road_node=aid,
        )

    nodes["C"] = (760.0, 220.0)
    nodes["DEPOT"] = (760.0, 320.0)
    nodes["HANGAR"] = (-250.0, 300.0)
    tw_x = [-100.0, 240.0, 480.0, 760.0, 1040.0, 1280.0, 1600.0]
    for i, x in enumerate(tw_x):
        nodes[f"TW{i}"] = (x, 420.0)
    rr_x = [300.0, 640.0, 980.0, 1320.0]
    for i, x in enumerate(rr_x, start=1):
        rid, rr = f"R{i}", f"RR{i}"
        nodes[rr] = (x, 600.0)
        nodes[rid] = (x, 720.0)
        stands[rid] = Stand(
            id=rid, kind="remote", terminal="REMOTE", wide=True, x=x, y=720.0, road_node=rr
        )

    # Gate bridges to apron road.
    for i in range(1, 13):
        add_edge(f"G{i}", f"A{i}")
    # Apron road chains.
    for i in range(1, 6):
        add_edge(f"A{i}", f"A{i + 1}")
    for i in range(7, 12):
        add_edge(f"A{i}", f"A{i + 1}")
    add_edge("A6", "C")
    add_edge("C", "A7")
    # Depot and hangar.
    add_edge("C", "DEPOT")
    add_edge("DEPOT", "TW3")
    add_edge("HANGAR", "TW0")
    add_edge("HANGAR", "A1")
    # Service taxiway chain.
    for i in range(6):
        add_edge(f"TW{i}", f"TW{i + 1}")
    # Apron road <-> taxiway links.
    for a, t in [("A2", "TW1"), ("A4", "TW2"), ("A5", "TW2"), ("C", "TW3"),
                 ("A8", "TW4"), ("A10", "TW5"), ("A12", "TW6")]:
        add_edge(a, t)
    # Remote road.
    for i in range(1, 5):
        add_edge(f"R{i}", f"RR{i}")
    for i in range(1, 4):
        add_edge(f"RR{i}", f"RR{i + 1}")
    for r, t in [("RR1", "TW1"), ("RR2", "TW2"), ("RR2", "TW3"), ("RR3", "TW3"),
                 ("RR3", "TW4"), ("RR4", "TW5")]:
        add_edge(r, t)

    exits = {"T1": (360.0, 0.0), "T2": (1160.0, 0.0)}
    airport = Airport(
        stands=stands,
        nodes=nodes,
        adjacency=adj,
        exits=exits,
        taxiway_nodes=frozenset(n for n in nodes if n.startswith("TW") or n.startswith("RR")),
    )
    airport.walk = _walk_matrix(airport)
    airport.gate_walk = _gate_walk_matrix(airport)
    airport.neighbours = _neighbours(airport)
    return airport


def _walk_matrix(airport: Airport) -> dict[str, dict[str, float]]:
    """Walking distance (m) from every stand to every terminal exit.

    Contact gate to its own terminal exit: pier walk + distance along the pier.
    To the other terminal: walk to own exit + inter-terminal connector + distance
    from the other exit. Remote stands: fixed bus-transfer equivalent.
    """
    out: dict[str, dict[str, float]] = {}
    for sid, st in airport.stands.items():
        row: dict[str, float] = {}
        for term, (ex, _) in airport.exits.items():
            if st.is_remote:
                row[term] = REMOTE_WALK_M
            elif st.terminal == term:
                row[term] = round(PIER_M + abs(st.x - ex), 1)
            else:
                own_ex = airport.exits[st.terminal][0]
                row[term] = round(PIER_M + abs(st.x - own_ex) + CONNECTOR_M + abs(ex - own_ex), 1)
        out[sid] = row
    return out


def _gate_walk_matrix(airport: Airport) -> dict[str, dict[str, float]]:
    """Airside walking distance between any two contact gates (connecting passengers)."""
    gates = airport.contact_gates
    out: dict[str, dict[str, float]] = {}
    for a in gates:
        out[a] = {}
        for b in gates:
            sa, sb = airport.stands[a], airport.stands[b]
            if a == b:
                d = 0.0
            elif sa.terminal == sb.terminal:
                d = abs(sa.x - sb.x)
            else:
                d = abs(sa.x - airport.exits[sa.terminal][0]) + CONNECTOR_M + abs(
                    sb.x - airport.exits[sb.terminal][0]
                ) + abs(airport.exits[sa.terminal][0] - airport.exits[sb.terminal][0])
            out[a][b] = round(d, 1)
    return out


def _neighbours(airport: Airport) -> dict[str, list[str]]:
    """1-hop neighbours used for partial observability of gate agents."""
    nb: dict[str, list[str]] = {}
    for i in range(1, 13):
        lo, hi = (1, 6) if i <= 6 else (7, 12)
        nb[f"G{i}"] = [f"G{j}" for j in (i - 1, i + 1) if lo <= j <= hi]
    for i in range(1, 5):
        nb[f"R{i}"] = [f"R{j}" for j in (i - 1, i + 1) if 1 <= j <= 4]
    return nb
