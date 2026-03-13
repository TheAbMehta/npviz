
from __future__ import annotations

from abc import ABC, abstractmethod


class AdapterBase(ABC):

    @property
    @abstractmethod
    def n_layers(self) -> int:
        pass

    @property
    @abstractmethod
    def heads_per_layer(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def params_per_layer(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def total_params(self) -> int:
        pass

    @property
    def connections(self) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(self.n_layers - 1)]

    def importance_scores(self) -> list[list[float]]:
        raise NotImplementedError("This adapter does not compute importance scores")

    def attach_recorder(self, recorder) -> None:
        recorder.attach(self)

    def snapshot(self, recorder, step: int) -> None:
        recorder.log_snapshot(step, self)
