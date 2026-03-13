
from __future__ import annotations

import os
import time
from pathlib import Path

from .schema import (
    ArchSnapshot,
    ImportanceSnapshot,
    MetricSnapshot,
    RewireEvent,
    Timeline,
)


class Recorder:

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, object] = {}

    def _get_file(self, name: str):
        if name not in self._files:
            self._files[name] = open(self.log_dir / name, "a")
        return self._files[name]

    def attach(self, model) -> ArchSnapshot:
        snap = ArchSnapshot(
            step=0,
            timestamp=time.time(),
            n_layers=model.n_layers,
            heads_per_layer=list(model.heads_per_layer),
            params_per_layer=list(model.params_per_layer),
            total_params=model.total_params,
            connections=list(getattr(model, "connections", [])),
        )
        self.log_snapshot_obj(snap)
        return snap

    def log_snapshot(self, step: int, model) -> ArchSnapshot:
        snap = ArchSnapshot(
            step=step,
            timestamp=time.time(),
            n_layers=model.n_layers,
            heads_per_layer=list(model.heads_per_layer),
            params_per_layer=list(model.params_per_layer),
            total_params=model.total_params,
            connections=list(getattr(model, "connections", [])),
        )
        self.log_snapshot_obj(snap)
        return snap

    def log_snapshot_obj(self, snap: ArchSnapshot):
        f = self._get_file("snapshots.jsonl")
        f.write(snap.to_json() + "\n")
        f.flush()

    def log_rewire(self, event: RewireEvent):
        f = self._get_file("events.jsonl")
        f.write(event.to_json() + "\n")
        f.flush()

    def log_importance(self, snap: ImportanceSnapshot):
        f = self._get_file("importance.jsonl")
        f.write(snap.to_json() + "\n")
        f.flush()

    def log_metrics(self, step: int, loss: float, grad_norm: float, lr: float):
        m = MetricSnapshot(step=step, loss=loss, grad_norm=grad_norm, learning_rate=lr)
        f = self._get_file("metrics.jsonl")
        f.write(m.to_json() + "\n")
        f.flush()

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def load(log_dir: str | Path) -> Timeline:
        log_dir = Path(log_dir)
        tl = Timeline()

        def _read_jsonl(filename, cls):
            path = log_dir / filename
            if not path.exists():
                return []
            items = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(cls.from_json(line))
            return items

        tl.snapshots = _read_jsonl("snapshots.jsonl", ArchSnapshot)
        tl.events = _read_jsonl("events.jsonl", RewireEvent)
        tl.importance = _read_jsonl("importance.jsonl", ImportanceSnapshot)
        tl.metrics = _read_jsonl("metrics.jsonl", MetricSnapshot)
        return tl
