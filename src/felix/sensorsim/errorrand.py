# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import numpy as np
from felix.sensorsim.enums import EErrType, EErrDep
from felix.sensorsim.errorsimulator import IErrSimulator
from felix.sensorsim.generatorsrandom import IGenRandom
from felix.sensorsim.sensordata import SensorData


class ErrRandGen(IErrSimulator):
    __slots__ = ("_generator", "_err_dep")

    def __init__(
        self,
        generator: IGenRandom,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        self._generator = generator
        self._err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.RANDOM

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def reseed(self, seed: int | None = None) -> None:
        self._generator.reseed(seed)

    def sim_errs(
        self,
        err_basis: np.ndarray,
        sens_data: SensorData,
    ) -> tuple[np.ndarray, SensorData]:
        rand_errs = self._generator.generate(shape=err_basis.shape)
        return (rand_errs, sens_data)

    def to_spec_dict(self) -> dict:
        spec = self._generator.to_spec_dict()
        spec["kind"] = 4
        spec["err_type"] = 1
        spec["err_dep"] = 1 if self._err_dep == EErrDep.DEPENDENT else 0
        return spec


class ErrRandGenPercent(IErrSimulator):
    __slots__ = ("_generator", "_err_dep")

    def __init__(
        self,
        generator: IGenRandom,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        self._generator = generator
        self._err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.RANDOM

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def reseed(self, seed: int | None = None) -> None:
        self._generator.reseed(seed)

    def sim_errs(
        self,
        err_basis: np.ndarray,
        sens_data: SensorData,
    ) -> tuple[np.ndarray, SensorData]:
        rand_errs = self._generator.generate(shape=err_basis.shape)
        return ((rand_errs / 100.0) * err_basis, sens_data)

    def to_spec_dict(self) -> dict:
        spec = self._generator.to_spec_dict()
        spec["kind"] = 5
        spec["err_type"] = 1
        spec["err_dep"] = 1 if self._err_dep == EErrDep.DEPENDENT else 0
        return spec
