"""Initial gate schedule as a Constraint Satisfaction Problem.

* Variables: flights.
* Domains: ``(stand, hold)`` with the stand able to host the aircraft size and
  ``hold`` in ``0..MAX_HOLD`` minutes (multiples of the timestep).
* Constraints: one flight per stand at a time with a ``BUFFER_MIN`` gap;
  wide-body only at wide-capable stands (domain filter); closed stands are
  unavailable (fixed blocks).

Solved by backtracking search with
  - MRV (minimum remaining values) variable ordering,
  - degree heuristic as the MRV tie-breaker,
  - least-cost value ordering (soft cost), and
  - forward checking after every assignment.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

from aerosync.ai.model import Block, CostFn, Task, Value, block_conflicts, holds, overlaps
from aerosync.config import BUFFER_MIN, CSP_NODE_LIMIT, MAX_HOLD_MIN
from aerosync.sim.airport import Stand


@dataclass
class CSPResult:
    """Solution (possibly partial) of the gate-assignment CSP."""

    assignment: dict[str, Value]
    success: bool
    nodes: int
    backtracks: int
    pruned: int
    time_ms: float
    trace: list[dict[str, Any]] = field(default_factory=list)


class GateCSP:
    """Backtracking CSP solver with MRV, degree heuristic and forward checking."""

    def __init__(self, tasks: list[Task], stands: dict[str, Stand], cost_fn: CostFn,
                 fixed: list[Block] | None = None, max_hold: int = MAX_HOLD_MIN,
                 buffer: int = BUFFER_MIN, use_mrv: bool = True, use_degree: bool = True,
                 use_fc: bool = True, node_limit: int = CSP_NODE_LIMIT,
                 trace: bool = False) -> None:
        self.tasks = {t.id: t for t in sorted(tasks, key=lambda t: (t.eta, t.id))}
        self.stands = stands
        self.cost_fn = cost_fn
        self.fixed = fixed or []
        self.max_hold = max_hold
        self.buffer = buffer
        self.use_mrv = use_mrv
        self.use_degree = use_degree
        self.use_fc = use_fc
        self.node_limit = node_limit
        self.trace_enabled = trace
        self.trace: list[dict[str, Any]] = []
        self.nodes = 0
        self.backtracks = 0
        self.pruned = 0
        self._hold_list = holds(max_hold)
        self.domains: dict[str, set[Value]] = {}
        self.costs: dict[str, dict[Value, float]] = {}
        self.neighbours: dict[str, list[str]] = {}
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        for tid, task in self.tasks.items():
            dom: set[Value] = set()
            costs: dict[Value, float] = {}
            for sid, st in self.stands.items():
                if not st.accepts(task.size):
                    continue
                for h in self._hold_list:
                    s, e = task.interval(h)
                    if any(b.stand == sid and block_conflicts(s, e, b, self.buffer)
                           for b in self.fixed):
                        continue
                    dom.add((sid, h))
                    costs[(sid, h)] = self.cost_fn(task, sid, h)
            self.domains[tid] = dom
            self.costs[tid] = costs
        ids = list(self.tasks)
        window = {
            tid: (t.eta, t.interval(self.max_hold)[1]) for tid, t in self.tasks.items()
        }
        self.neighbours = {tid: [] for tid in ids}
        for i, a in enumerate(ids):
            a0, a1 = window[a]
            for b in ids[i + 1:]:
                b0, b1 = window[b]
                if overlaps(a0, a1, b0, b1, self.buffer):
                    self.neighbours[a].append(b)
                    self.neighbours[b].append(a)

    # ------------------------------------------------------------- heuristics
    def select_variable(self, unassigned: list[str]) -> str:
        """MRV, then degree heuristic (most constraints on unassigned vars), then ETA."""
        if not self.use_mrv:
            return unassigned[0]
        un = set(unassigned)

        def key(v: str) -> tuple[int, int, int, str]:
            degree = sum(1 for n in self.neighbours[v] if n in un) if self.use_degree else 0
            return (len(self.domains[v]), -degree, self.tasks[v].eta, v)

        return min(unassigned, key=key)

    def order_values(self, var: str) -> list[Value]:
        """Least-cost first (soft objective), ties broken by stand id and hold."""
        c = self.costs[var]
        return sorted(self.domains[var], key=lambda val: (c[val], val[1], val[0]))

    # -------------------------------------------------------------- inference
    def _consistent(self, var: str, val: Value, assignment: dict[str, Value]) -> bool:
        s, e = self.tasks[var].interval(val[1])
        for n in self.neighbours[var]:
            other = assignment.get(n)
            if other is None or other[0] != val[0]:
                continue
            os_, oe = self.tasks[n].interval(other[1])
            if overlaps(s, e, os_, oe, self.buffer):
                return False
        return True

    def _forward_check(self, var: str, val: Value, assignment: dict[str, Value]
                       ) -> list[tuple[str, Value]] | None:
        """Prune values of unassigned neighbours that clash with ``var = val``.

        Returns the list of removed ``(neighbour, value)`` pairs, or ``None`` if a
        domain was wiped out (the removals are undone before returning).
        """
        stand, hold = val
        s, e = self.tasks[var].interval(hold)
        removed: list[tuple[str, Value]] = []
        for n in self.neighbours[var]:
            if n in assignment:
                continue
            dom = self.domains[n]
            task_n = self.tasks[n]
            for h in self._hold_list:
                cand = (stand, h)
                if cand in dom:
                    ns, ne = task_n.interval(h)
                    if overlaps(s, e, ns, ne, self.buffer):
                        dom.discard(cand)
                        removed.append((n, cand))
            if not dom:
                self._restore(removed)
                return None
        self.pruned += len(removed)
        return removed

    def _restore(self, removed: list[tuple[str, Value]]) -> None:
        for n, v in removed:
            self.domains[n].add(v)

    # ----------------------------------------------------------------- search
    def solve(self) -> CSPResult:
        """Run backtracking search and return the (best effort) assignment."""
        t0 = time.perf_counter()
        assignment: dict[str, Value] = {}
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 10 * len(self.tasks) + 1000))
        try:
            ok = self._backtrack(assignment, list(self.tasks))
        finally:
            sys.setrecursionlimit(old_limit)
        return CSPResult(
            assignment=dict(assignment) if ok else dict(assignment),
            success=ok,
            nodes=self.nodes,
            backtracks=self.backtracks,
            pruned=self.pruned,
            time_ms=(time.perf_counter() - t0) * 1000.0,
            trace=self.trace,
        )

    def _backtrack(self, assignment: dict[str, Value], unassigned: list[str]) -> bool:
        if not unassigned:
            return True
        if self.nodes >= self.node_limit:
            return False
        var = self.select_variable(unassigned)
        rest = [u for u in unassigned if u != var]
        for val in self.order_values(var):
            self.nodes += 1
            if not self.use_fc and not self._consistent(var, val, assignment):
                continue
            removed: list[tuple[str, Value]] = []
            if self.use_fc:
                res = self._forward_check(var, val, assignment)
                if res is None:
                    continue
                removed = res
            assignment[var] = val
            if self.trace_enabled:
                self.trace.append({
                    "step": len(self.trace) + 1,
                    "var": var,
                    "value": val,
                    "cost": round(self.costs[var][val], 2),
                    "domain_sizes": {u: len(self.domains[u]) for u in rest},
                    "pruned": len(removed),
                })
            if self._backtrack(assignment, rest):
                return True
            del assignment[var]
            self._restore(removed)
            self.backtracks += 1
            if self.nodes >= self.node_limit:
                return False
        return False


def solve_gate_csp(tasks: list[Task], stands: dict[str, Stand], cost_fn: CostFn,
                   fixed: list[Block] | None = None, **kwargs: Any) -> CSPResult:
    """Convenience wrapper: build a :class:`GateCSP` and solve it."""
    return GateCSP(tasks, stands, cost_fn, fixed=fixed, **kwargs).solve()


def schedule_violations(tasks: dict[str, Task], assignment: dict[str, Value],
                        stands: dict[str, Stand], fixed: list[Block] | None = None,
                        buffer: int = BUFFER_MIN) -> list[str]:
    """List every hard-constraint violation in ``assignment`` (empty = valid)."""
    out: list[str] = []
    by_stand: dict[str, list[tuple[int, int, str]]] = {}
    for tid, (sid, h) in assignment.items():
        task = tasks[tid]
        if not stands[sid].accepts(task.size):
            out.append(f"{tid}: wide-body at narrow stand {sid}")
        s, e = task.interval(h)
        by_stand.setdefault(sid, []).append((s, e, tid))
        for b in fixed or []:
            if b.stand == sid and block_conflicts(s, e, b, buffer):
                out.append(f"{tid}: clashes with {b.label or 'block'} at {sid}")
    for sid, items in by_stand.items():
        items.sort()
        for i, (s1, e1, a) in enumerate(items):
            for s2, e2, b in items[i + 1:]:
                if overlaps(s1, e1, s2, e2, buffer):
                    out.append(f"{a} and {b} overlap at {sid}")
    return out
