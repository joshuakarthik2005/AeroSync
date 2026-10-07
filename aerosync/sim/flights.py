"""Flight model and deterministic, seeded flight-schedule generator."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from aerosync.config import MIN_TURN_FRACTION, STEP_MIN, round_up_step

DOMESTIC_AIRLINES = ["6E", "SG", "UK", "AI", "QP"]
INTERNATIONAL_AIRLINES = ["EK", "BA", "LH", "QR", "SQ", "AI"]
INTERNATIONAL_TERMINAL = "T2"
DOMESTIC_TERMINAL = "T1"


@dataclass
class Flight:
    """A scheduled turnaround (arrival + departure of one aircraft).

    Times are minutes from the scenario start. ``arr_delay`` is hidden ground
    truth: agents only learn it at ``reveal_at`` (partial observability).
    """

    id: str
    airline: str
    size: str  # "N" narrow-body | "W" wide-body
    sched_arr: int
    sched_dep: int
    pax: int
    pref_terminal: str
    arr_delay: int = 0
    reveal_at: int = 0

    @property
    def ground_time(self) -> int:
        """Scheduled turnaround duration in minutes."""
        return self.sched_dep - self.sched_arr

    @property
    def min_turn(self) -> int:
        """Shortest feasible turnaround (lets small holds be absorbed)."""
        return round_up_step(self.ground_time * MIN_TURN_FRACTION)

    @property
    def actual_arr(self) -> int:
        return self.sched_arr + self.arr_delay

    def planned_departure(self, on_block: int) -> int:
        """Earliest departure if the aircraft blocks on at ``on_block``."""
        return max(self.sched_dep, on_block + self.min_turn)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Flight:
        return Flight(**d)


def _round5(x: float) -> int:
    return int(round(x / STEP_MIN) * STEP_MIN)


def generate_flights(gen: dict[str, Any], seed: int, featured: list[dict[str, Any]] | None = None
                     ) -> list[Flight]:
    """Generate a deterministic flight list from generator parameters and a seed.

    Parameters (keys of ``gen``): ``n_flights``, ``wide_ratio``, ``last_arrival_min``,
    ``peaks`` (list of ``[centre_min, sd_min, weight]``, optional), ``delay_prob``,
    ``delay_min`` (``[lo, hi]``), ``reveal_lead_min`` (``[lo, hi]``),
    ``international_ratio``.
    """
    rng = random.Random(seed)
    featured = featured or []
    used_ids = {f["id"] for f in featured}
    n = int(gen["n_flights"]) - len(featured)
    last_arr = int(gen["last_arrival_min"])
    peaks = gen.get("peaks")
    flights: list[Flight] = []

    for _ in range(n):
        if peaks:
            weights = [p[2] for p in peaks]
            centre, sd, _w = rng.choices(peaks, weights=weights)[0]
            arr = rng.gauss(centre, sd)
        else:
            arr = rng.uniform(0, last_arr)
        arr = min(max(_round5(arr), 0), last_arr)

        wide = rng.random() < gen.get("wide_ratio", 0.25)
        intl = wide or rng.random() < gen.get("international_ratio", 0.3)
        airline = rng.choice(INTERNATIONAL_AIRLINES if intl else DOMESTIC_AIRLINES)
        if wide:
            ground = rng.choice(range(90, 151, STEP_MIN))
            pax = rng.randint(250, 380)
        else:
            ground = rng.choice(range(45, 76, STEP_MIN))
            pax = rng.randint(120, 186)
        while True:
            fid = f"{airline}{rng.randint(100, 999)}"
            if fid not in used_ids:
                used_ids.add(fid)
                break
        flights.append(
            Flight(
                id=fid,
                airline=airline,
                size="W" if wide else "N",
                sched_arr=arr,
                sched_dep=arr + ground,
                pax=pax,
                pref_terminal=INTERNATIONAL_TERMINAL if intl else DOMESTIC_TERMINAL,
            )
        )

    for f in featured:
        fields = Flight.__dataclass_fields__
        flights.append(Flight.from_dict({k: v for k, v in f.items() if k in fields}))

    # Hidden stochastic arrival delays (revealed shortly before arrival).
    delay_prob = gen.get("delay_prob", 0.0)
    dlo, dhi = gen.get("delay_min", [5, 30])
    rlo, rhi = gen.get("reveal_lead_min", [20, 60])
    flights.sort(key=lambda f: (f.sched_arr, f.id))
    for f in flights:
        if f.id in {x["id"] for x in featured}:
            continue
        if rng.random() < delay_prob:
            f.arr_delay = _round5(rng.uniform(dlo, dhi))
            f.reveal_at = max(0, _round5(f.sched_arr - rng.uniform(rlo, rhi)))
    return flights
