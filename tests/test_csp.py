"""CSP: hard constraints are never violated; MRV + forward checking solve every scenario."""

from __future__ import annotations

import pytest

from aerosync.ai.csp import GateCSP, schedule_violations, solve_gate_csp
from aerosync.ai.model import Block, Task
from aerosync.sim.scenarios import SCENARIO_NAMES

from .conftest import scenario_tasks


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_mrv_forward_checking_valid_schedule_on_all_scenarios(name, airport, cost_fn):
    tasks = scenario_tasks(name)
    res = solve_gate_csp(tasks, airport.stands, cost_fn)
    assert res.success
    assert len(res.assignment) == len(tasks)
    assert schedule_violations({t.id: t for t in tasks}, res.assignment, airport.stands) == []


@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_csp_never_violates_hard_constraints(seed, airport, cost_fn):
    tasks = scenario_tasks("rush_hour", seed)
    closure = [Block("G4", 0, 10_000, is_closure=True), Block("G9", 60, 200, is_closure=True)]
    res = solve_gate_csp(tasks, airport.stands, cost_fn, fixed=closure)
    by_id = {t.id: t for t in tasks}
    assert res.success
    assert schedule_violations(by_id, res.assignment, airport.stands, closure) == []
    for fid, (stand, _h) in res.assignment.items():
        assert airport.stands[stand].accepts(by_id[fid].size)


def test_wide_body_only_on_wide_stands(airport, cost_fn):
    tasks = [Task(f"W{i}", "W", 0, 120, 100, "T2") for i in range(8)]
    res = solve_gate_csp(tasks, airport.stands, cost_fn)
    assert res.success
    assert all(airport.stands[s].wide for s, _ in res.assignment.values())


def test_mrv_picks_most_constrained_variable(airport, cost_fn):
    tasks = [Task("N1", "N", 0, 60, 50, "T1"), Task("W1", "W", 0, 120, 100, "T2")]
    csp = GateCSP(tasks, airport.stands, cost_fn)
    # The wide-body has fewer legal stands, hence fewer remaining values.
    assert csp.select_variable(["N1", "W1"]) == "W1"


def test_degree_heuristic_breaks_mrv_ties(airport, cost_fn):
    # Same domain sizes; "B" overlaps both others, so it constrains the most.
    tasks = [Task("A", "N", 0, 60, 50, "T1"), Task("B", "N", 50, 110, 50, "T1"),
             Task("C", "N", 100, 160, 50, "T1")]
    csp = GateCSP(tasks, airport.stands, cost_fn, max_hold=0)
    degrees = {v: len(csp.neighbours[v]) for v in csp.tasks}
    assert degrees == {"A": 1, "B": 2, "C": 1}
    assert csp.select_variable(["A", "B", "C"]) == "B"


def test_forward_checking_prunes_neighbour_domains(airport, cost_fn):
    tasks = [Task("A", "N", 0, 60, 50, "T1"), Task("B", "N", 10, 70, 50, "T1")]
    csp = GateCSP(tasks, airport.stands, cost_fn)
    before = len(csp.domains["B"])
    removed = csp._forward_check("A", ("G1", 0), {})
    assert removed and len(csp.domains["B"]) < before
    assert all(v[0] == "G1" for _, v in removed)


def test_overconstrained_problem_reports_failure(airport, cost_fn):
    # 9 simultaneous wide-bodies, only 8 wide-capable stands and no hold allowed.
    tasks = [Task(f"W{i}", "W", 0, 120, 100, "T2") for i in range(9)]
    res = solve_gate_csp(tasks, airport.stands, cost_fn, max_hold=0)
    assert not res.success


def test_plain_backtracking_without_heuristics_is_still_valid(airport, cost_fn):
    tasks = scenario_tasks("gate_failure")[:25]
    res = solve_gate_csp(tasks, airport.stands, cost_fn, use_mrv=False, use_fc=False)
    assert res.success
    assert schedule_violations({t.id: t for t in tasks}, res.assignment, airport.stands) == []
