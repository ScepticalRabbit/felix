# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from abc import ABC, abstractmethod
import numpy as np
from felix.sensorsim.enums import EErrType, EErrDep
from felix.sensorsim.sensordata import SensorData


class IErrSimulator(ABC):
    @abstractmethod
    def get_error_type(self) -> EErrType:
        pass

    @abstractmethod
    def get_error_dep(self) -> EErrDep:
        pass

    @abstractmethod
    def set_error_dep(self, dependence: EErrDep) -> None:
        pass

    @abstractmethod
    def reseed(self, seed: int | None = None) -> None:
        pass

    @abstractmethod
    def sim_errs(
        self,
        err_basis: np.ndarray,
        sens_data: SensorData,
    ) -> tuple[np.ndarray, SensorData]:
        pass

    @abstractmethod
    def to_spec_dict(self) -> dict:
        pass
