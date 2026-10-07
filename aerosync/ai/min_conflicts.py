"""Min-conflicts local search used to repair the schedule after a disruption.

Pseudocode::

    current <- existing schedule (with the disruption applied)
    repeat up to MAX_STEPS:
        C <- mutable flights involved in a hard-constraint conflict
        if C is empty: return current
        v <- random choice from C
        current[v] <- argmin_{x in domain(v)} (conflicts(v, x), soft_cost(v, x))

Only flights that have not yet blocked on are mutable; aircraft already at a
stand and stand closures are fixed blocks. The soft cost adds a reassignment
penalty when ``x`` moves the flight away from its current stand, so the repair
prefers to shift as few flights as possible.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from aerosync.ai.model import Block, CostFn, Task, Value, block_conflicts, holds, overlaps
from aerosync.config import BUFFER_MIN, MAX_HOLD_MIN, MIN_CONFLICTS_MAX_STEPS
from aerosync.sim.airport import Stand


@dataclass
class RepairResult:
    """Outcome of a min-conflicts repair."""

    assignment: dict[str, Value]
    steps: int
    conflicts_before: int
    conflicts_after: int
    changed: dict[str, tuple[Value, Value]] = field(default_factory=dict)
    time_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.conflicts_after == 0


class MinConflictsRepair:
    """Stateful min-conflicts solver over a fixed set of tasks."""

    def __init__(self, tasks: dict[str, Task], stands: dict[str, Stand], cost_fn: CostFn,
                 fixed: list[Block] | None = None, max_hold: int = MAX_HOLD_MIN,
                 buffer: int = BUFFER_MIN, reassign_penalty: float = 15.0,
                 rng: random.Random | None = None, patience: int = 150,
                 noise: float = 0.2) -> None:
        self.tasks = tasks
        self.patience = patience
        self.noise = noise
        self.stands = stands
        self.cost_fn = cost_fn
        self.fixed = fixed or []
        self.buffer = buffer
        self.reassign_penalty = reassign_penalty
        self.rng = rng or random.Random(0)
        self._holds = holds(max_hold)
        self._fixed_by_stand: dict[str, list[Block]] = {}
        for b in self.fixed:
            self._fixed_by_stand.setdefault(b.stand, []).append(b)

    def _domain(self, task: Task) -> list[Value]:
        return [(sid, h) for sid, st in self.stands.items() if st.accepts(task.size)
                for h in self._holds]

    def conflicts_of(self, var: str, val: Value, by_stand: dict[str, dict[str, Value]]) -> int:
        """Number of hard constraints ``var = val`` violates against the rest."""
        sid, h = val
        s, e = self.tasks[var].interval(h)
        n = 0
        for b in self._fixed_by_stand.get(sid, ()):
            if block_conflicts(s, e, b, self.buffer):
                n += 1
        for other, (_osid, oh) in by_stand.get(sid, {}).items():
            if other == var:
                continue
            os_, oe = self.tasks[other].interval(oh)
            if overlaps(s, e, os_, oe, self.buffer):
                n += 1
        return n

    def repair(self, assignment: dict[str, Value], mutable: set[str],
               max_steps: int = MIN_CONFLICTS_MAX_STEPS,
               protected: set[str] | None = None) -> RepairResult:
        """Repair a copy of ``assignment`` and return the result.

        ``protected`` variables (already announced to passengers) are only picked
        when no unprotected variable is in conflict, never take random-walk moves,
        and pay three times the reassignment penalty for changing stand.
        """
        protected = protected or set()
        t0 = time.perf_counter()
        current = dict(assignment)
        original = dict(assignment)
        by_stand: dict[str, dict[str, Value]] = {}
        for var, val in current.items():
            by_stand.setdefault(val[0], {})[var] = val

        def conflicted() -> list[str]:
            return sorted(v for v in mutable if v in current
                          and self.conflicts_of(v, current[v], by_stand) > 0)

        def total_conflicts() -> int:
            return sum(self.conflicts_of(v, current[v], by_stand) for v in current)

        before = total_conflicts()
        steps = 0
        bad = conflicted()
        best_seen = len(bad)
        best_state = dict(current)
        soft_cache: dict[str, dict[Value, float]] = {}
        since_improve = 0
        while bad and steps < max_steps and since_improve < self.patience:
            steps += 1
            free = [v for v in bad if v not in protected]
            var = self.rng.choice(free or bad)
            task = self.tasks[var]
            orig_stand = original[var][0]
            pen = self.reassign_penalty * (3.0 if var in protected else 1.0)
            if var not in soft_cache:
                soft_cache[var] = {
                    val: round(self.cost_fn(task, val[0], val[1])
                               + (pen if val[0] != orig_stand else 0.0), 6)
                    for val in self._domain(task)
                }
            best: list[Value] = []
            best_key: tuple[int, float] | None = None
            for val, soft in soft_cache[var].items():
                if best_key is not None and soft > best_key[1] and best_key[0] == 0:
                    continue
                c = self.conflicts_of(var, val, by_stand)
                key = (c, soft)
                if best_key is None or key < best_key:
                    best_key, best = key, [val]
                elif key == best_key:
                    best.append(val)
            if (best_key and best_key[0] > 0 and var not in protected
                    and self.rng.random() < self.noise):
                # Random walk: escape plateaus where no single move removes the conflict.
                new_val = self.rng.choice(sorted(soft_cache[var]))
            else:
                new_val = best[0] if len(best) == 1 else self.rng.choice(sorted(best))
            old_val = current[var]
            del by_stand[old_val[0]][var]
            current[var] = new_val
            by_stand.setdefault(new_val[0], {})[var] = new_val
            bad = conflicted()
            if len(bad) < best_seen:
                best_seen, best_state, since_improve = len(bad), dict(current), 0
            else:
                since_improve += 1

        if bad and len(bad) > best_seen:
            # Stagnated: fall back to the best schedule seen.
            current = best_state
            by_stand = {}
            for var, val in current.items():
                by_stand.setdefault(val[0], {})[var] = val
        changed = {v: (original[v], current[v]) for v in current if current[v] != original[v]}
        return RepairResult(
            assignment=current,
            steps=steps,
            conflicts_before=before,
            conflicts_after=total_conflicts(),
            changed=changed,
            time_ms=(time.perf_counter() - t0) * 1000.0,
        )


def count_conflicts(tasks: dict[str, Task], assignment: dict[str, Value],
                    fixed: list[Block] | None = None, buffer: int = BUFFER_MIN) -> int:
    """Number of conflicting pairs / block clashes in a schedule."""
    n = 0
    by_stand: dict[str, list[tuple[int, int]]] = {}
    for tid, (sid, h) in assignment.items():
        s, e = tasks[tid].interval(h)
        for b in fixed or []:
            if b.stand == sid and block_conflicts(s, e, b, buffer):
                n += 1
        by_stand.setdefault(sid, []).append((s, e))
    for items in by_stand.values():
        for i, (s1, e1) in enumerate(items):
            for s2, e2 in items[i + 1:]:
                if overlaps(s1, e1, s2, e2, buffer):
                    n += 1
    return n
