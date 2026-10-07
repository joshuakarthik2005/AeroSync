"""Coordinator Agent (auctioneer).

Responsibilities:

1. Build the initial provisional schedule with the CSP (MRV + degree +
   forward checking) from the published timetable.
2. Run a Contract Net round when a flight enters its look-ahead window (or
   re-requests after a disruption) and award the stand to the cheapest bid.
3. Detect conflicts after faults / new ETAs and repair with min-conflicts.
4. Log every decision together with the inputs that produced it.

It holds no global ground truth: ETAs come from flight-agent STATE_UPDATEs,
stand availability from gate-agent bids/refusals and FAULT messages.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

from aerosync.agents.gate_agent import GateAgent
from aerosync.ai.contract_net import AuctionResult, Bid, run_contract_net
from aerosync.ai.csp import GateCSP
from aerosync.ai.min_conflicts import MinConflictsRepair, RepairResult, count_conflicts
from aerosync.ai.model import Block, Task, static_cost_fn
from aerosync.config import ALPHA, BUFFER_MIN, DEFAULT_WEIGHTS, LOOKAHEAD_MIN, BidWeights
from aerosync.sim.message_bus import MsgType

if TYPE_CHECKING:
    from aerosync.sim.environment import FlightState, Simulation


class CoordinatorAgent:
    """Auctioneer + planner + repairer."""

    def __init__(self, sim: Simulation, seed: int, weights: BidWeights = DEFAULT_WEIGHTS,
                 alpha: float = ALPHA) -> None:
        self.id = "COORD"
        self.sim = sim
        self.weights = weights
        self.alpha = alpha
        self.rng = random.Random(seed * 7919 + 17)
        self.gates = {sid: GateAgent(sid, sim.airport, weights, alpha)
                      for sid in sim.airport.stands}
        self.cost_fn = static_cost_fn(sim.airport, weights)
        self.contracted: set[str] = set()
        self.known_closures = 0
        self._residual = 0

    # ------------------------------------------------------------- planning
    def initial_plan(self) -> None:
        sim = self.sim
        tasks = [fs.task() for fs in sim.states.values()]
        csp = GateCSP(tasks, sim.airport.stands, self.cost_fn, fixed=sim.closure_blocks())
        res = csp.solve()
        for fid, (stand, hold) in sorted(res.assignment.items()):
            fs = sim.states[fid]
            fs.assign(0, stand, "CSP initial plan")
            fs.planned_hold = hold
        self._sync_gates()
        for gate in self.gates.values():
            gate.publish_state(sim.bus, 0)
        sim.bus.send(0, MsgType.STATE_UPDATE, self.id, "*",
                     {"plan": "CSP", "flights": len(res.assignment), "nodes": res.nodes,
                      "backtracks": res.backtracks})
        sim.decisions.append({
            "t": 0, "kind": "csp_plan",
            "inputs": {"flights": len(tasks), "stands": len(sim.airport.stands),
                       "heuristics": "MRV + degree + forward checking, least-cost values"},
            "output": {"assigned": len(res.assignment), "success": res.success,
                       "nodes": res.nodes, "backtracks": res.backtracks, "pruned": res.pruned,
                       "time_ms": round(res.time_ms, 1)},
            "reasoning": (f"CSP assigned {len(res.assignment)}/{len(tasks)} flights with "
                          f"{res.nodes} nodes, {res.backtracks} backtracks and {res.pruned} "
                          "values pruned by forward checking."),
        })

    # --------------------------------------------------------------- helpers
    def _sync_gates(self) -> None:
        """Each gate agent learns (only) its own commitments from the awarded plan."""
        per: dict[str, dict[str, tuple[int, int]]] = {g: {} for g in self.gates}
        firm: set[str] = set()
        for fs in self.sim.states.values():
            if fs.stand is None:
                continue
            if fs.id in self.contracted or fs.on_block is not None:
                firm.add(fs.id)
            if fs.departed:
                # A just-departed aircraft still blocks the stand for the buffer.
                if fs.off_block is not None and fs.off_block + BUFFER_MIN > self.sim.t:
                    per[fs.stand][fs.id] = (fs.off_block - 1, fs.off_block)
                continue
            per[fs.stand][fs.id] = fs.planned_interval()
        for g, agent in self.gates.items():
            agent.set_commitments(per[g], firm)

    def _fixed_blocks(self) -> list[Block]:
        blocks = list(self.sim.closure_blocks())
        for fs in self.sim.states.values():
            if fs.at_stand and fs.stand is not None:
                s, e = fs.planned_interval()
                blocks.append(Block(fs.stand, s, e, label=f"{fs.id} on block"))
            elif fs.departed and fs.stand is not None and fs.off_block is not None:
                blocks.append(Block(fs.stand, fs.off_block - 1, fs.off_block,
                                    label=f"{fs.id} departed"))
        return blocks

    def _mutable(self, t: int) -> list[FlightState]:
        return [fs for fs in self.sim.states.values()
                if fs.on_block is None and fs.stand is not None]

    # ---------------------------------------------------------------- repair
    def detect_and_repair(self, t: int, trigger: str) -> None:
        """Find hard-constraint conflicts in the current plan and repair them."""
        sim = self.sim
        mutable = self._mutable(t)
        if not mutable:
            return
        tasks: dict[str, Task] = {}
        assignment: dict[str, tuple[str, int]] = {}
        for fs in mutable:
            task = fs.task(now=t)
            tasks[fs.id] = task
            start = max(fs.planned_start(), task.eta)
            assignment[fs.id] = (fs.stand, start - task.eta)  # type: ignore[arg-type]
        fixed = self._fixed_blocks()
        before = count_conflicts(tasks, assignment, fixed)
        if before > 0 and before <= self._residual and trigger == "plan drift":
            before = 0  # same unresolvable residue as last step; nothing new to repair
        if before == 0:
            # Planned starts may have been shifted by the max(eta, now) clamp; keep them.
            for fs in mutable:
                fs.planned_hold = tasks[fs.id].eta + assignment[fs.id][1] - fs.eta
            return
        solver = MinConflictsRepair(tasks, sim.airport.stands, self.cost_fn, fixed=fixed,
                                    reassign_penalty=self.weights.reassign, rng=self.rng)
        # Provisional (un-announced) slots are always movable; confirmed contracts
        # only when they themselves clash with something fixed or confirmed.
        firm_ids = {fs.id for fs in mutable if fs.id in self.contracted}
        firm_by_stand: dict[str, dict[str, tuple[str, int]]] = {}
        for fid in firm_ids:
            firm_by_stand.setdefault(assignment[fid][0], {})[fid] = assignment[fid]
        movable = {fs.id for fs in mutable if fs.id not in firm_ids}
        movable |= {fid for fid in firm_ids
                    if solver.conflicts_of(fid, assignment[fid], firm_by_stand) > 0}
        res = solver.repair(assignment, mutable=movable, protected=firm_ids)
        method = "min-conflicts"
        if res.conflicts_after > 0:
            replan = self._csp_replan(tasks, assignment, movable, fixed)
            if replan is not None:
                res, method = replan, "min-conflicts + CSP re-plan"
        self._residual = res.conflicts_after
        snapshot_before = {fid: list(v) for fid, v in assignment.items()
                           if fid in res.changed}
        changed_rows = []
        for fid, (old, new) in sorted(res.changed.items()):
            fs = sim.states[fid]
            fs.assign(t, new[0], f"{method} ({trigger})")
            fs.planned_hold = tasks[fid].eta + new[1] - fs.eta
            sim.bus.send(t, MsgType.AWARD, self.id, new[0],
                         {"flight": fid, "stand": new[0], "start": tasks[fid].eta + new[1],
                          "why": f"repair after {trigger}"})
            changed_rows.append({"flight": fid, "from": old[0], "to": new[0],
                                 "old_start": tasks[fid].eta + old[1],
                                 "new_start": tasks[fid].eta + new[1]})
        for fs in mutable:
            if fs.id not in res.changed:
                fs.planned_hold = tasks[fs.id].eta + assignment[fs.id][1] - fs.eta
        sim.counters["repairs"] += 1
        agents = sorted({"COORD"} | {r["from"] for r in changed_rows}
                        | {r["to"] for r in changed_rows})
        record = {
            "t": t, "trigger": trigger, "steps": res.steps,
            "conflicts_before": res.conflicts_before, "conflicts_after": res.conflicts_after,
            "changed": changed_rows, "before": snapshot_before, "agents": agents,
            "time_ms": round(res.time_ms, 2), "method": method,
        }
        sim.repairs.append(record)
        sim.decisions.append({
            "t": t, "kind": "repair", "inputs": {"trigger": trigger,
                                                 "conflicts": res.conflicts_before,
                                                 "mutable_flights": len(mutable)},
            "output": {"moved": len(changed_rows), "steps": res.steps,
                       "conflicts_after": res.conflicts_after},
            "reasoning": (f"{method} repaired {res.conflicts_before} conflict(s) after "
                          f"{trigger} ({res.conflicts_after} left) in {res.steps} step(s), "
                          f"moving {len(changed_rows)} flight(s)."),
        })
        self._sync_gates()

    def _csp_replan(self, tasks: dict[str, Task], assignment: dict[str, tuple[str, int]],
                    movable: set[str], fixed: list[Block]) -> RepairResult | None:
        """Fallback when local search stalls: re-solve the movable flights as a CSP.

        Everything that is not movable becomes a fixed block; value ordering adds the
        reassignment penalty so the new plan stays close to the old one.
        """
        t0 = time.perf_counter()
        blocks = list(fixed)
        for fid, (stand, hold) in assignment.items():
            if fid not in movable:
                s, e = tasks[fid].interval(hold)
                blocks.append(Block(stand, s, e, label=fid))
        base = self.cost_fn
        penalty = self.weights.reassign

        def cost(task: Task, stand: str, hold: int) -> float:
            return base(task, stand, hold) + (penalty if stand != assignment[task.id][0] else 0.0)

        csp = GateCSP([tasks[f] for f in sorted(movable)], self.sim.airport.stands, cost,
                      fixed=blocks, node_limit=20_000)
        out = csp.solve()
        if not out.success:
            return None
        new = dict(assignment)
        new.update(out.assignment)
        changed = {f: (assignment[f], new[f]) for f in new if new[f] != assignment[f]}
        return RepairResult(assignment=new, steps=out.nodes,
                            conflicts_before=count_conflicts(tasks, assignment, fixed),
                            conflicts_after=count_conflicts(tasks, new, fixed), changed=changed,
                            time_ms=(time.perf_counter() - t0) * 1000.0)

    # ----------------------------------------------------------- negotiation
    def negotiate(self, fs: FlightState, t: int, context: str = "request",
                  only_now: bool = False) -> AuctionResult | None:
        """Run one Contract Net round for ``fs`` and apply the award."""
        sim = self.sim
        self._sync_gates()
        agent = sim.flight_agents[fs.id]
        eta = max(fs.eta, t)
        cfp = agent.cfp(t, eta, fs.stand)
        if only_now:
            cfp["allowed_holds"] = [0]
        contractors = [g for g in self.gates.values()]
        if only_now:
            contractors = [g for g in contractors if sim.can_block_on(g.id, t)]
        result = run_contract_net(sim.bus, t, self.id, cfp, contractors,
                                  requester_id=agent.id, context=context)
        sim.counters["auctions"] += 1
        result.reasoning = explain_auction(result, sim)
        sim.auctions.setdefault(fs.id, []).append(result.to_dict())
        sim.decisions.append({
            "t": t, "kind": "award", "flight": fs.id,
            "inputs": {"context": context, "eta": eta, "bids": len(result.bids),
                       "refusals": len(result.refusals)},
            "output": {"stand": result.winner.bidder if result.winner else None,
                       "cost": round(result.winner.cost, 2) if result.winner else None},
            "reasoning": result.reasoning,
        })
        win = result.winner
        if win is None:
            return result
        fs.assign(t, win.bidder, f"Contract Net award ({context})")
        fs.planned_hold = win.start - fs.eta
        fs.published = True
        agent.contract = win.bidder
        agent.needs_request = False
        self.contracted.add(fs.id)
        self._sync_gates()
        return result

    def step(self, t: int, changes: list[dict[str, Any]]) -> None:
        sim = self.sim
        for g in self.gates.values():
            g.receive(sim.bus)
        sim.bus.drain(self.id)
        for c in changes:
            if c["kind"] == "gate_closure":
                b = sim.closures[-1]
                self.gates[c["gate"]].add_closure(b.start, b.end)
            if c["kind"] in ("delay", "flight_delay"):
                fid = c.get("flight")
                if fid:
                    self.contracted.discard(fid)
        triggers = sorted({c["kind"] for c in changes if c["kind"] != "comms_restored"})
        self.detect_and_repair(t, " + ".join(triggers) if triggers else "plan drift")
        due = sorted((fs for fs in self._mutable(t)
                      if fs.id not in self.contracted and fs.eta - LOOKAHEAD_MIN <= t),
                     key=lambda s: (s.eta, s.id))
        displaced = False
        for fs in due:
            ctx = "re-request" if sim.auctions.get(fs.id) else "request"
            res = self.negotiate(fs, t, context=ctx)
            displaced |= bool(res and res.winner and res.winner.displaced)
        if displaced:
            self.detect_and_repair(t, "displacement")
        self._sync_gates()
        for g in self.gates.values():
            g.publish_state(sim.bus, t)


def explain_auction(result: AuctionResult, sim: Simulation) -> str:
    """Human-readable reasoning for a Contract Net award."""
    cfp = result.cfp
    f = sim.states[cfp["flight"]].flight
    size = "wide-body" if f.size == "W" else "narrow-body"
    head = (f"Flight {f.id} ({size}, {f.pax} pax, prefers {f.pref_terminal}) issued a CFP at "
            f"{sim.clock(result.t)} for ETA {sim.clock(cfp['eta'])} ({result.context}). "
            f"{len(result.bids)} stand(s) bid, {len(result.refusals)} refused.")
    if result.winner is None:
        return head + " No feasible bid: the flight keeps its current plan and holds."
    w = result.winner
    parts = ", ".join(f"{k} {v:.1f}" for k, v in w.breakdown.items() if abs(v) > 0.05)
    text = (f"{head} Winner {w.bidder} with cost {w.cost:.1f} ({parts or 'all terms zero'}); "
            f"on-block {sim.clock(w.start)}, hold {w.hold} min.")
    ranked = sorted(result.bids, key=lambda b: (b.cost, b.bidder))
    if len(ranked) > 1:
        r = ranked[1]
        diffs = {k: r.breakdown.get(k, 0) - w.breakdown.get(k, 0) for k in w.breakdown}
        key, val = max(diffs.items(), key=lambda kv: kv[1])
        text += (f" Runner-up {r.bidder} cost {r.cost:.1f} (+{r.cost - w.cost:.1f}), "
                 f"mainly higher {key.replace('_', ' ')} (+{val:.1f}).")
    return text


def best_bid(bids: list[Bid]) -> Bid | None:
    return min(bids, key=lambda b: (b.cost, b.bidder)) if bids else None
