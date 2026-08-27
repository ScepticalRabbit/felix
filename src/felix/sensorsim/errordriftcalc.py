# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from abc import ABC, abstractmethod
import numpy as np
from felix.sensorsim.enums import EErrType, EErrDep
from felix.sensorsim.errorsimulator import IErrSimulator
from felix.sensorsim.sensordata import SensorData


class IDriftCalculator(IErrSimulator, ABC):
    @abstractmethod
    def calc_drift(self, time_steps: np.ndarray) -> np.ndarray:
        pass


class DriftConstant(IDriftCalculator):
    __slots__ = ("_offset", "_err_dep")

    def __init__(
        self,
        offset: float = 0.0,
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

    def calc_drift(self, time_steps: np.ndarray) -> np.ndarray:
        return self._offset * np.ones_like(time_steps)

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


class DriftLinear(IDriftCalculator):
    __slots__ = ("_rate", "_time_start", "_offset", "_err_dep")

    def __init__(
        self,
        rate: float,
        time_start: float = 0.0,
        offset: float = 0.0,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        self._rate = rate
        self._time_start = time_start
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

    def calc_drift(self, time_steps: np.ndarray) -> np.ndarray:
        return self._rate * (time_steps - self._time_start) + self._offset

    def sim_errs(
        self,
        err_basis: np.ndarray,
        sens_data: SensorData,
    ) -> tuple[np.ndarray, SensorData]:
        t_steps = sens_data.sample_times
        if t_steps is None:
            t_steps = np.arange(err_basis.shape[2], dtype=np.float64)
        drift = self.calc_drift(t_steps)[None, None, :]
        return (drift * np.ones_like(err_basis), sens_data)

    def to_spec_dict(self) -> dict:
        return {
            "kind": 10,
            "err_type": 0,
            "err_dep": 1 if self._err_dep == EErrDep.DEPENDENT else 0,
            "dist_type": 0,
            "param0": float(self._rate),
            "param1": float(self._time_start),
            "param2": float(self._offset),
        }


class DriftPolynomial(IDriftCalculator):
    __slots__ = ("_coeffs", "_time_start", "_err_dep")

    def __init__(
        self,
        coeffs: tuple[float, ...],
        time_start: float = 0.0,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        self._coeffs = np.array(coeffs, dtype=np.float64)
        self._time_start = time_start
        self._err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def reseed(self, seed: int | None = None) -> None:
        pass

    def calc_drift(self, time_steps: np.ndarray) -> np.ndarray:
        dt = time_steps - self._time_start
        poly = np.polynomial.Polynomial(self._coeffs)
        return poly(dt)

    def sim_errs(
        self,
        err_basis: np.ndarray,
        sens_data: SensorData,
    ) -> tuple[np.ndarray, SensorData]:
        t_steps = sens_data.sample_times
        if t_steps is None:
            t_steps = np.arange(err_basis.shape[2], dtype=np.float64)
        drift = self.calc_drift(t_steps)[None, None, :]
        return (drift * np.ones_like(err_basis), sens_data)

    def to_spec_dict(self) -> dict:
        return {
            "kind": 11,
            "err_type": 0,
            "err_dep": 1 if self._err_dep == EErrDep.DEPENDENT else 0,
            "dist_type": 0,
            "param0": float(self._time_start),
            "param1": 0.0,
            "param2": 0.0,
            "poly_coeffs": self._coeffs,
        }
