
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ArchSnapshot:
    step: int
    timestamp: float
    n_layers: int
    heads_per_layer: list[int]
    params_per_layer: list[int]
    total_params: int
    connections: list[tuple[int, int]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> ArchSnapshot:
        d = json.loads(s)
        d["connections"] = [tuple(c) for c in d["connections"]]
        return cls(**d)


@dataclass
class RewireEvent:
    step: int
    event_type: str
    layer_idx: int
    head_idx: Optional[int] = None
    reason: str = ""
    importance_score: Optional[float] = None
    loss_before: float = 0.0
    loss_after: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> RewireEvent:
        return cls(**json.loads(s))


@dataclass
class ImportanceSnapshot:
    step: int
    scores: list[list[float]]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> ImportanceSnapshot:
        return cls(**json.loads(s))


@dataclass
class MetricSnapshot:
    step: int
    loss: float
    grad_norm: float
    learning_rate: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> MetricSnapshot:
        return cls(**json.loads(s))


@dataclass
class Timeline:
    snapshots: list[ArchSnapshot] = field(default_factory=list)
    events: list[RewireEvent] = field(default_factory=list)
    importance: list[ImportanceSnapshot] = field(default_factory=list)
    metrics: list[MetricSnapshot] = field(default_factory=list)

    @property
    def steps(self) -> list[int]:
        return sorted({s.step for s in self.snapshots})

    @property
    def event_steps(self) -> list[int]:
        return sorted({e.step for e in self.events})

    def snapshot_at(self, step: int) -> Optional[ArchSnapshot]:
        best = None
        for s in self.snapshots:
            if s.step <= step:
                if best is None or s.step > best.step:
                    best = s
        return best

    def importance_at(self, step: int) -> Optional[ImportanceSnapshot]:
        best = None
        for s in self.importance:
            if s.step <= step:
                if best is None or s.step > best.step:
                    best = s
        return best

    def events_between(self, start: int, end: int) -> list[RewireEvent]:
        return [e for e in self.events if start <= e.step <= end]
