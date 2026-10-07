"""Typed message bus used for all inter-agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MsgType(StrEnum):
    """Performatives of the AeroSync protocol (Contract Net + state sharing)."""

    CFP = "CFP"  # call for proposals (coordinator -> gate agents)
    BID = "BID"  # proposal with a cost (gate agent -> coordinator)
    REFUSE = "REFUSE"  # gate agent cannot host the flight
    AWARD = "AWARD"  # contract awarded (coordinator -> gate + flight)
    STATE_UPDATE = "STATE_UPDATE"  # schedule/ETA/load broadcast
    FAULT = "FAULT"  # disruption notification


@dataclass
class Message:
    """One message on the bus."""

    t: int
    type: MsgType
    sender: str
    receiver: str  # agent id or "*" for broadcast
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int = 0

    def summary(self) -> str:
        p = self.payload
        if self.type == MsgType.CFP:
            return f"CFP flight={p.get('flight')} eta={p.get('eta')} size={p.get('size')}"
        if self.type == MsgType.BID:
            return f"BID flight={p.get('flight')} cost={p.get('cost'):.1f} start={p.get('start')}"
        if self.type == MsgType.REFUSE:
            return f"REFUSE flight={p.get('flight')} reason={p.get('reason')}"
        if self.type == MsgType.AWARD:
            return f"AWARD flight={p.get('flight')} stand={p.get('stand')} ({p.get('why', '')})"
        if self.type == MsgType.FAULT:
            return f"FAULT {p.get('description', '')}"
        return "STATE_UPDATE " + ", ".join(f"{k}={v}" for k, v in p.items() if k != "busy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "t": self.t,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "summary": self.summary(),
        }


class MessageBus:
    """Delivers messages to per-agent inboxes and keeps an append-only log.

    Agents that have lost communication (``down``) neither send nor receive:
    their messages are dropped and counted.
    """

    def __init__(self) -> None:
        self.log: list[Message] = []
        self._inbox: dict[str, list[Message]] = {}
        self.down: set[str] = set()
        self.dropped = 0
        self._seq = 0

    def send(self, t: int, type_: MsgType, sender: str, receiver: str,
             payload: dict[str, Any] | None = None) -> Message | None:
        """Send a message; returns ``None`` if it was dropped by a comms fault."""
        if sender in self.down or receiver in self.down:
            self.dropped += 1
            return None
        self._seq += 1
        msg = Message(t=t, type=type_, sender=sender, receiver=receiver,
                      payload=payload or {}, seq=self._seq)
        self.log.append(msg)
        self._inbox.setdefault(receiver, []).append(msg)
        return msg

    def multicast(self, t: int, type_: MsgType, sender: str, receivers: list[str],
                  payload: dict[str, Any] | None = None) -> int:
        """Send the same payload to several receivers; returns how many were delivered."""
        return sum(
            1 for r in receivers if self.send(t, type_, sender, r, payload) is not None
        )

    def drain(self, agent_id: str) -> list[Message]:
        """Return and clear the inbox of ``agent_id``."""
        return self._inbox.pop(agent_id, [])

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {m.value: 0 for m in MsgType}
        for m in self.log:
            out[m.type.value] += 1
        return out
