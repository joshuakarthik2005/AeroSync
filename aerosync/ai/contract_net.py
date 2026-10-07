"""Contract Net Protocol (Smith, 1980) for negotiating stands.

1. The manager (coordinator) broadcasts a CFP describing the task (a flight).
2. Each contractor (gate agent) either REFUSEs (infeasible) or replies with a
   BID carrying its cost estimate and a cost breakdown.
3. The manager AWARDs the contract to the minimum-cost bid (ties broken by
   contractor id) and notifies the requesting flight agent.

Contractors whose communication link is down never receive the CFP and are
recorded as refusals with reason ``no response (comms loss)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from aerosync.sim.message_bus import MessageBus, MsgType


@dataclass
class Bid:
    """A contractor's proposal."""

    bidder: str
    cost: float
    start: int
    end: int
    hold: int
    breakdown: dict[str, float] = field(default_factory=dict)
    terms: dict[str, float] = field(default_factory=dict)
    displaced: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bidder": self.bidder,
            "cost": round(self.cost, 2),
            "start": self.start,
            "end": self.end,
            "hold": self.hold,
            "breakdown": {k: round(v, 2) for k, v in self.breakdown.items()},
            "terms": {k: round(v, 3) for k, v in self.terms.items()},
            "displaced": list(self.displaced),
        }


@dataclass
class Refusal:
    """A contractor declining the CFP."""

    bidder: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"bidder": self.bidder, "reason": self.reason}


@dataclass
class AuctionResult:
    """Full record of one Contract Net round (used for explainability)."""

    task_id: str
    t: int
    cfp: dict[str, Any]
    bids: list[Bid]
    refusals: list[Refusal]
    winner: Bid | None
    reasoning: str = ""
    context: str = "request"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "t": self.t,
            "context": self.context,
            "cfp": self.cfp,
            "bids": [b.to_dict() for b in sorted(self.bids, key=lambda b: (b.cost, b.bidder))],
            "refusals": [r.to_dict() for r in self.refusals],
            "winner": self.winner.to_dict() if self.winner else None,
            "reasoning": self.reasoning,
        }


class Contractor(Protocol):
    """Anything that can answer a call for proposals."""

    id: str

    def propose(self, cfp: dict[str, Any]) -> Bid | Refusal: ...


def select_winner(bids: list[Bid]) -> Bid | None:
    """Award rule: minimum cost, ties broken by lexicographic contractor id."""
    if not bids:
        return None
    return min(bids, key=lambda b: (round(b.cost, 9), b.bidder))


def run_contract_net(bus: MessageBus, t: int, manager_id: str, cfp: dict[str, Any],
                     contractors: list[Contractor], requester_id: str | None = None,
                     context: str = "request") -> AuctionResult:
    """Execute one full CFP -> BID/REFUSE -> AWARD round over the message bus."""
    bids: list[Bid] = []
    refusals: list[Refusal] = []
    for c in contractors:
        delivered = bus.send(t, MsgType.CFP, manager_id, c.id, cfp)
        if delivered is None:
            refusals.append(Refusal(c.id, "no response (comms loss)"))
            continue
        reply = c.propose(cfp)
        if isinstance(reply, Bid):
            bids.append(reply)
            bus.send(t, MsgType.BID, c.id, manager_id,
                     {"flight": cfp["flight"], "cost": reply.cost, "start": reply.start})
        else:
            refusals.append(reply)
            bus.send(t, MsgType.REFUSE, c.id, manager_id,
                     {"flight": cfp["flight"], "reason": reply.reason})
    winner = select_winner(bids)
    if winner is not None:
        payload = {"flight": cfp["flight"], "stand": winner.bidder, "start": winner.start,
                   "cost": round(winner.cost, 2), "why": f"min cost of {len(bids)} bids"}
        bus.send(t, MsgType.AWARD, manager_id, winner.bidder, payload)
        if requester_id:
            bus.send(t, MsgType.AWARD, manager_id, requester_id, payload)
    return AuctionResult(task_id=cfp["flight"], t=t, cfp=dict(cfp), bids=bids,
                         refusals=refusals, winner=winner, context=context)
