"""Contract Net awards the minimum-cost feasible bid."""

from __future__ import annotations

from dataclasses import dataclass

from aerosync.agents.gate_agent import GateAgent
from aerosync.ai.contract_net import Bid, Refusal, run_contract_net, select_winner
from aerosync.sim.message_bus import MessageBus, MsgType


@dataclass
class FakeContractor:
    id: str
    cost: float | None

    def propose(self, cfp):
        if self.cost is None:
            return Refusal(self.id, "infeasible")
        return Bid(self.id, self.cost, 0, 60, 0)


CFP = {"flight": "XX1", "eta": 0}


def test_awards_minimum_cost_feasible_bid():
    bus = MessageBus()
    cs = [FakeContractor("G1", 50.0), FakeContractor("G2", None), FakeContractor("G3", 12.5),
          FakeContractor("G4", 30.0)]
    res = run_contract_net(bus, 0, "COORD", CFP, cs, requester_id="FLT:XX1")
    assert res.winner is not None and res.winner.bidder == "G3"
    assert {r.bidder for r in res.refusals} == {"G2"}
    counts = bus.counts()
    assert counts["CFP"] == 4 and counts["BID"] == 3 and counts["REFUSE"] == 1
    assert counts["AWARD"] == 2  # winner + requesting flight


def test_no_award_when_everyone_refuses():
    bus = MessageBus()
    res = run_contract_net(bus, 0, "COORD", CFP, [FakeContractor("G1", None)])
    assert res.winner is None and bus.counts()["AWARD"] == 0


def test_ties_broken_by_contractor_id():
    assert select_winner([Bid("G9", 10.0, 0, 1, 0), Bid("G2", 10.0, 0, 1, 0)]).bidder == "G2"


def test_comms_loss_counts_as_refusal():
    bus = MessageBus()
    bus.down.add("G3")
    res = run_contract_net(bus, 0, "COORD", CFP, [FakeContractor("G3", 1.0),
                                                   FakeContractor("G5", 9.0)])
    assert res.winner.bidder == "G5"
    assert res.refusals[0].reason.startswith("no response")


def test_gate_agent_bid_formula_and_refusal(airport):
    g4 = GateAgent("G4", airport)
    cfp = {"flight": "AI1", "eta": 100, "sched_dep": 160, "min_turn": 50, "size": "N",
           "pax": 150, "pref_terminal": "T1", "current_stand": None}
    bid = g4.propose(cfp)
    w = g4.weights
    assert isinstance(bid, Bid) and bid.hold == 0
    assert bid.breakdown["walk"] == w.walk * airport.walk_distance("G4", "T1")
    assert bid.cost == sum(bid.breakdown.values())
    g1 = GateAgent("G1", airport)
    assert isinstance(g1.propose({**cfp, "size": "W"}), Refusal)
    # A firm commitment forces a hold; a provisional one can be displaced at a price.
    g4.set_commitments({"OTHER": (90, 150)}, firm={"OTHER"})
    held = g4.propose(cfp)
    assert held.hold > 0 and held.breakdown["delay"] == w.delay * held.hold
    g4.set_commitments({"OTHER": (90, 150)}, firm=set())
    disp = g4.propose(cfp)
    assert disp.hold == 0 and disp.displaced == ["OTHER"]
    assert disp.breakdown["displacement"] == w.reassign


def test_neighbour_pressure_uses_only_neighbour_state(airport):
    g5 = GateAgent("G5", airport)
    assert g5.neighbour_ids == ["G4", "G6"]
    bus = MessageBus()
    g4 = GateAgent("G4", airport)
    g4.set_commitments({"F": (0, 600)})
    g4.publish_state(bus, 0)
    assert bus.log[0].type == MsgType.STATE_UPDATE
    g5.receive(bus)
    assert g5.neighbour_load(100, 200) == 0.5  # G4 fully busy, G6 empty
