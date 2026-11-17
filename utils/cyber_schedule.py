"""Deterministic threat-propagation and participation schedules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import math

import numpy as np


@dataclass(frozen=True)
class ThreatMilestone:
    round: int
    prevalence: float


class ThreatPropagationSchedule:
    """Controls which clients have been exposed to the emerging attack.

    The client exposure order is sampled once from ``seed``. Therefore all
    algorithms can be run with exactly the same spatial/temporal threat path.
    """

    def __init__(
        self,
        num_clients: int,
        milestones: Sequence[Sequence[float | int]],
        seed: int = 0,
        client_ids: Iterable[int] | None = None,
    ) -> None:
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        self.num_clients = int(num_clients)
        self.client_ids = (
            list(range(self.num_clients))
            if client_ids is None
            else sorted(set(int(i) for i in client_ids))
        )
        if not self.client_ids:
            raise ValueError("client_ids must contain at least one eligible client")
        self.milestones = sorted(
            [ThreatMilestone(int(r), float(p)) for r, p in milestones],
            key=lambda m: m.round,
        )
        for m in self.milestones:
            if not (0.0 <= m.prevalence <= 1.0):
                raise ValueError("Threat prevalence must be in [0, 1]")

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self.client_ids)).tolist()
        self.exposure_order = [self.client_ids[i] for i in perm]

    def prevalence(self, round_id: int) -> float:
        p = 0.0
        for milestone in self.milestones:
            if round_id >= milestone.round:
                p = milestone.prevalence
            else:
                break
        return p

    def exposed_clients(self, round_id: int) -> Set[int]:
        p = self.prevalence(round_id)
        n = int(math.ceil(p * len(self.client_ids))) if p > 0 else 0
        return set(self.exposure_order[:n])

    @property
    def first_drift_round(self) -> int | None:
        positive = [m.round for m in self.milestones if m.prevalence > 0]
        return min(positive) if positive else None


class ParticipationScheduler:
    """Deterministic per-round availability schedule.

    ``exposed_prob`` can be made smaller than ``normal_prob`` to model hidden
    drift: threat-exposed sites are less likely to contribute updates.
    """

    def __init__(
        self,
        num_clients: int,
        normal_prob: float,
        exposed_prob: float | None = None,
        min_selected: int = 3,
        seed: int = 0,
    ) -> None:
        self.num_clients = int(num_clients)
        self.normal_prob = float(normal_prob)
        self.exposed_prob = float(exposed_prob if exposed_prob is not None else normal_prob)
        self.min_selected = int(min_selected)
        self.seed = int(seed)
        for value in (self.normal_prob, self.exposed_prob):
            if not (0.0 <= value <= 1.0):
                raise ValueError("participation probability must be in [0, 1]")

    def select(
        self,
        round_id: int,
        exposed_clients: Set[int],
        eligible_clients: Iterable[int] | None = None,
    ) -> List[int]:
        eligible = sorted(
            set(range(self.num_clients))
            if eligible_clients is None
            else set(int(i) for i in eligible_clients)
        )
        if not eligible:
            return []

        # Separate generator per round makes the schedule reproducible and
        # independent of how much randomness an FL algorithm consumes.
        rng = np.random.default_rng(self.seed + 1_000_003 * int(round_id))
        selected: List[int] = []
        scores: List[Tuple[float, int]] = []
        for client_id in eligible:
            p = self.exposed_prob if client_id in exposed_clients else self.normal_prob
            u = float(rng.random())
            scores.append((u / max(p, 1e-12), client_id))
            if u < p:
                selected.append(client_id)

        # FedCCFA's original MADD clustering needs >=3 clients for its exact
        # distance expression. Force a small minimum when enough clients exist.
        required = min(self.min_selected, len(eligible))
        if len(selected) < required:
            for _, client_id in sorted(scores):
                if client_id not in selected:
                    selected.append(client_id)
                if len(selected) >= required:
                    break
        return sorted(selected)
