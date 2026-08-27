# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import numpy as np
from felix.sensorsim.errorsimulator import IErrSimulator, EErrType, EErrDep
from felix.sensorsim.generatorsrandom import IGenRandom
from felix.sensorsim.sensordata import SensorData


class ErrSysOffset(IErrSimulator):
    __slots__ = ("_offset", "_err_dep")

    def __init__(
        self,
        offset: float,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        self._offset = offset
        self._err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def reseed(self, seed: int | None = None) -> None:
        pass

    def sim_errs(
        self,
        err_basis: np.ndarray,
        sens_data: SensorData,
    ) -> tuple[np.ndarray, SensorData]:
        return (self._offset * np.ones_like(err_basis), sens_data)

    def to_spec_dict(self) -> dict:
        return {
            "kind": 0,
            "err_type": 0,
            "err_dep": 1 if self._err_dep == EErrDep.DEPENDENT else 0,
            "dist_type": 0,
            "param0": float(self._offset),
            "param1": 0.0,
            "param2": 0.0,
        }


class ErrSysOffsetPercent(IErrSimulator):
    __slots__ = ("_offset_percent", "_err_dep")

    def __init__(
        self,
        offset_percent: float,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        self._offset_percent = offset_percent
        self._err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def reseed(self, seed: int | None = None) -> None:
        pass

    def sim_errs(
        self,
        err_basis: np.ndarray,
        sens_data: SensorData,
    ) -> tuple[np.ndarray, SensorData]:
        return ((self._offset_percent / 100.0) * err_basis, sens_data)

    def to_spec_dict(self) -> dict:
        return {
            "kind": 1,
            "err_type": 0,
            "err_dep": 1 if self._err_dep == EErrDep.DEPENDENT else 0,
            "dist_type": 0,
            "param0": float(self._offset_percent),
            "param1": 0.0,
            "param2": 0.0,
        }


class ErrSysGen(IErrSimulator):
    __slots__ = ("_generator", "_err_dep")

    def __init__(
        self,
        generator: IGenRandom,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        self._generator = generator
        self._err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

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
        num_sensors = err_basis.shape[0]
        rand_errs_sens = self._generator.generate(shape=(num_sensors, 1, 1))
        rand_errs = rand_errs_sens * np.ones_like(err_basis)
        return (rand_errs, sens_data)

    def to_spec_dict(self) -> dict:
        spec = self._generator.to_spec_dict()
        spec["kind"] = 2
        spec["err_type"] = 0
        spec["err_dep"] = 1 if self._err_dep == EErrDep.DEPENDENT else 0
        return spec


class ErrSysGenPercent(IErrSimulator):
    __slots__ = ("_generator", "_err_dep")

    def __init__(
        self,
        generator: IGenRandom,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        self._generator = generator
        self._err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

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
        num_sensors = err_basis.shape[0]
        rand_errs_sens = self._generator.generate(shape=(num_sensors, 1, 1))
        rand_errs = (rand_errs_sens / 100.0) * err_basis
        return (rand_errs, sens_data)

    def to_spec_dict(self) -> dict:
        spec = self._generator.to_spec_dict()
        spec["kind"] = 3
        spec["err_type"] = 0
        spec["err_dep"] = 1 if self._err_dep == EErrDep.DEPENDENT else 0
        return spec
