"""Min-conflicts repairs a closed-gate event with zero remaining conflicts."""

from __future__ import annotations

import random

from aerosync.ai.csp import schedule_violations, solve_gate_csp
from aerosync.ai.min_conflicts import MinConflictsRepair, count_conflicts
from aerosync.ai.model import Block
from aerosync.experiments import run_single

from .conftest import scenario_tasks


def test_min_conflicts_repairs_gate_closure(airport, cost_fn):
    tasks = scenario_tasks("gate_failure")
    by_id = {t.id: t for t in tasks}
    plan = solve_gate_csp(tasks, airport.stands, cost_fn).assignment
    closure = [Block("G4", 120, 10_000, is_closure=True)]
    assert count_conflicts(by_id, plan, closure) > 0  # the closure breaks the plan
    # Flights that leave before the closure cannot conflict; everyone else may move.
    mutable = {fid for fid, t in by_id.items() if t.interval(plan[fid][1])[1] > 100}
    solver = MinConflictsRepair(by_id, airport.stands, cost_fn, fixed=closure,
                                rng=random.Random(3), patience=1000)
    res = solver.repair(plan, mutable)
    assert res.conflicts_after == 0
    assert schedule_violations(by_id, res.assignment, airport.stands, closure) == []
    # Only a minority of flights has to move.
    assert 0 < len(res.changed) < len(tasks) // 2


def test_repair_is_noop_on_consistent_schedule(airport, cost_fn):
    tasks = scenario_tasks("rush_hour")
    by_id = {t.id: t for t in tasks}
    plan = solve_gate_csp(tasks, airport.stands, cost_fn).assignment
    res = MinConflictsRepair(by_id, airport.stands, cost_fn).repair(plan, set(by_id))
    assert res.steps == 0 and res.changed == {} and res.conflicts_after == 0


def test_gate_failure_simulation_never_parks_at_closed_gate():
    res = run_single("gate_failure", "aerosync", 42)
    closure = next(c for c in res.closures if c["stand"] == "G4")
    for f in res.flights:
        if f["stand"] == "G4" and f["on_block"] is not None:
            assert f["on_block"] < closure["start"]
    assert res.metrics["hard_violations"] == 0
    assert any(r["trigger"].startswith("gate_closure") for r in res.repairs)
