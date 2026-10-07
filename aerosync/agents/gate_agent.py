"""Gate Agent: one per stand, bids in the Contract Net.

PEAS
----
* **Performance**: low passenger walking, few remote/terminal-mismatch parkings,
  no buffer violations, low propagated delay, balanced load with neighbours.
* **Environment**: its own stand schedule, the 1-hop neighbouring stands
  (via STATE_UPDATE messages only), closures and the CFPs it receives.
* **Actuators**: BID / REFUSE replies, STATE_UPDATE broadcasts to neighbours.
* **Sensors**: CFP messages, AWARD messages, neighbour STATE_UPDATEs, FAULTs.

The agent is *partially observable*: it never sees the full airport schedule.
"""

from __future__ import annotations

from typing import Any

from aerosync.ai.contract_net import Bid, Refusal
from aerosync.ai.model import holds, overlaps
from aerosync.config import (
    ALPHA,
    BUFFER_MIN,
    DEFAULT_WEIGHTS,
    LOOKAHEAD_MIN,
    MAX_HOLD_MIN,
    BidWeights,
)
from aerosync.sim.airport import Airport
from aerosync.sim.message_bus import MessageBus, MsgType

BUFFER_RISK_HORIZON = 30  # minutes of slack beyond the buffer that count as "safe"


class GateAgent:
    """Autonomous agent managing a single stand."""

    def __init__(self, stand_id: str, airport: Airport, weights: BidWeights = DEFAULT_WEIGHTS,
                 alpha: float = ALPHA) -> None:
        self.id = stand_id
        self.airport = airport
        self.stand = airport.stands[stand_id]
        self.weights = weights
        self.alpha = alpha
        self.neighbour_ids = list(airport.neighbours.get(stand_id, []))
        self.commitments: dict[str, tuple[int, int]] = {}
        self.firm: set[str] = set()
        self.closures: list[tuple[int, int]] = []
        self.neighbour_busy: dict[str, list[tuple[int, int]]] = {n: [] for n in self.neighbour_ids}
        self._last_published: tuple[tuple[int, int], ...] | None = None

    # ------------------------------------------------------------ perception
    def receive(self, bus: MessageBus) -> None:
        """Read neighbour STATE_UPDATEs (the only view of other stands it has)."""
        for msg in bus.drain(self.id):
            if msg.type == MsgType.STATE_UPDATE and msg.sender in self.neighbour_busy:
                self.neighbour_busy[msg.sender] = [tuple(x) for x in msg.payload.get("busy", [])]

    def set_commitments(self, commitments: dict[str, tuple[int, int]],
                        firm: set[str] | None = None) -> None:
        """Replace the agent's own schedule.

        ``firm`` lists confirmed contracts (announced or on block); the rest are
        provisional CSP slots that a better-placed request may displace.
        """
        self.commitments = dict(commitments)
        self.firm = set(commitments) if firm is None else {f for f in firm if f in commitments}

    def add_closure(self, start: int, end: int) -> None:
        self.closures.append((start, end))

    def busy(self) -> list[tuple[int, int]]:
        return sorted(self.commitments.values())

    def publish_state(self, bus: MessageBus, t: int) -> bool:
        """Send STATE_UPDATE to neighbours if the schedule changed."""
        busy = tuple(self.busy())
        if busy == self._last_published:
            return False
        delivered = bus.multicast(
            t, MsgType.STATE_UPDATE, self.id, self.neighbour_ids,
            {"stand": self.id, "flights": len(busy), "busy": [list(b) for b in busy]},
        )
        if delivered == len(self.neighbour_ids):
            self._last_published = busy
        return True

    # --------------------------------------------------------------- bidding
    def _closed(self, start: int, end: int) -> bool:
        return any(start < ce and cs < end for cs, ce in self.closures)

    def _slot(self, start: int, end: int, exclude: str) -> tuple[bool, list[str]]:
        """Is ``[start, end)`` free of firm commitments? Also list displaced provisional ones."""
        if self._closed(start, end):
            return False, []
        displaced = []
        for fid, (s, e) in self.commitments.items():
            if fid == exclude or not overlaps(start, end, s, e, BUFFER_MIN):
                continue
            if fid in self.firm:
                return False, []
            displaced.append(fid)
        return True, sorted(displaced)

    def _slack(self, start: int, end: int, ignore: set[str]) -> float:
        gaps = []
        for fid, (s, e) in self.commitments.items():
            if fid in ignore:
                continue
            if e <= start:
                gaps.append(start - e - BUFFER_MIN)
            elif s >= end:
                gaps.append(s - end - BUFFER_MIN)
        return min(gaps) if gaps else float("inf")

    def neighbour_load(self, start: int, end: int) -> float:
        """Mean share of the window [start-60, end+60] occupied at neighbour stands."""
        if not self.neighbour_ids:
            return 0.0
        w0, w1 = start - LOOKAHEAD_MIN, end + LOOKAHEAD_MIN
        span = w1 - w0
        shares = []
        for n in self.neighbour_ids:
            occ = sum(max(0, min(e, w1) - max(s, w0)) for s, e in self.neighbour_busy.get(n, []))
            shares.append(min(1.0, occ / span))
        return sum(shares) / len(shares)

    def propose(self, cfp: dict[str, Any]) -> Bid | Refusal:
        """Answer a CFP with a costed bid or a refusal.

        ``bid = w1*walk*pax/150 + w2*terminal_mismatch + w3*remote + w4*buffer_risk
        + w5*(alpha * neighbour_load) + w6*hold_min
        + w7*(reassignment + displaced_provisional)``

        Only *firm* commitments make a slot infeasible; overlapping provisional
        CSP slots may be displaced at a price (they are re-homed by repair).
        """
        fid = cfp["flight"]
        if not self.stand.accepts(cfp["size"]):
            return Refusal(self.id, "narrow-body stand cannot host wide-body")
        eta = int(cfp["eta"])
        sched_dep, min_turn = int(cfp["sched_dep"]), int(cfp["min_turn"])
        allowed = cfp.get("allowed_holds") or holds(MAX_HOLD_MIN)
        w = self.weights
        chosen = None
        best = float("inf")
        for h in allowed:
            if w.delay * h >= best:
                break
            s = eta + h
            e = max(sched_dep, s + min_turn)
            ok, displaced = self._slot(s, e, fid)
            if not ok:
                continue
            score = w.delay * h + w.reassign * len(displaced)
            if score < best:
                best, chosen = score, (h, s, e, displaced)
        if chosen is None:
            if self._closed(eta, max(sched_dep, eta + min_turn)):
                return Refusal(self.id, "stand closed")
            return Refusal(self.id, f"no free slot within {allowed[-1]} min hold")
        h, s, e, displaced = chosen
        walk = self.airport.walk_distance(self.id, cfp["pref_terminal"])
        same_terminal = self.stand.terminal == cfp["pref_terminal"]
        mismatch = 0.0 if (self.stand.is_remote or same_terminal) else 1.0
        remote = 1.0 if self.stand.is_remote else 0.0
        slack = self._slack(s, e, {fid, *displaced})
        risk = 0.0 if slack == float("inf") else max(0.0, min(1.0, (BUFFER_RISK_HORIZON - slack)
                                                             / BUFFER_RISK_HORIZON))
        pressure = self.alpha * self.neighbour_load(s, e)
        delay = float(h)
        current = cfp.get("current_stand")
        reassign = 1.0 if (current and current != self.id) else 0.0
        pax_factor = cfp.get("pax", 150) / 150.0
        breakdown = {
            "walk": w.walk * walk * pax_factor,
            "terminal_mismatch": w.mismatch * mismatch,
            "remote": w.remote * remote,
            "buffer_risk": w.buffer_risk * risk,
            "neighbour_pressure": w.neighbour * pressure,
            "delay": w.delay * delay,
            "reassignment": w.reassign * reassign,
            "displacement": w.reassign * len(displaced),
        }
        terms = {"walk_m": walk, "terminal_mismatch": mismatch, "remote": remote,
                 "buffer_risk": risk, "neighbour_pressure": pressure, "hold_min": delay,
                 "reassignment": reassign, "displaced": float(len(displaced)),
                 "slack_min": -1.0 if slack == float("inf") else slack}
        return Bid(bidder=self.id, cost=sum(breakdown.values()), start=s, end=e, hold=h,
                   breakdown=breakdown, terms=terms, displaced=displaced)
