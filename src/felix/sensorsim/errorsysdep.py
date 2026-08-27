# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import numpy as np
from felix.sensorsim.enums import ERoundMethod, EErrType, EErrDep
from felix.sensorsim.errorsimulator import IErrSimulator
from felix.sensorsim.sensordata import SensorData


def _select_round_method(method: ERoundMethod):
    if method == ERoundMethod.FLOOR:
        return np.floor
    if method == ERoundMethod.CEIL:
        return np.ceil
    return np.round


class ErrSysRoundOff(IErrSimulator):
    __slots__ = ("_base", "_method", "_method_enum", "_err_dep")

    def __init__(
        self,
        method: ERoundMethod = ERoundMethod.ROUND,
        base: float = 1.0,
        err_dep: EErrDep = EErrDep.DEPENDENT,
    ) -> None:
        self._base = base
        self._method_enum = method
        self._method = _select_round_method(method)
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
        rounded = self._base * self._method(err_basis / self._base)
        return (rounded - err_basis, sens_data)

    def to_spec_dict(self) -> dict:
        m_val = 0
        if self._method_enum == ERoundMethod.FLOOR:
            m_val = 1
        elif self._method_enum == ERoundMethod.CEIL:
            m_val = 2
        return {
            "kind": 6,
            "err_type": 0,
            "err_dep": 1 if self._err_dep == EErrDep.DEPENDENT else 0,
            "dist_type": 0,
            "param0": float(self._base),
            "param1": float(m_val),
            "param2": 0.0,
        }


class ErrSysDigitisation(IErrSimulator):
    __slots__ = ("_units_per_bit", "_method", "_method_enum", "_err_dep")

    def __init__(
        self,
        bits_per_unit: float | None = None,
        units_per_bit: float | None = None,
        method: ERoundMethod = ERoundMethod.ROUND,
        err_dep: EErrDep = EErrDep.DEPENDENT,
    ) -> None:
        if bits_per_unit is not None:
            self._units_per_bit = 1.0 / bits_per_unit
        elif units_per_bit is not None:
            self._units_per_bit = units_per_bit
        else:
            self._units_per_bit = 1.0

        self._method_enum = method
        self._method = _select_round_method(method)
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
        digitised = self._units_per_bit * self._method(
            err_basis / self._units_per_bit
        )
        return (digitised - err_basis, sens_data)

    def to_spec_dict(self) -> dict:
        m_val = 0
        if self._method_enum == ERoundMethod.FLOOR:
            m_val = 1
        elif self._method_enum == ERoundMethod.CEIL:
            m_val = 2
        return {
            "kind": 7,
            "err_type": 0,
            "err_dep": 1 if self._err_dep == EErrDep.DEPENDENT else 0,
            "dist_type": 0,
            "param0": float(self._units_per_bit),
            "param1": float(m_val),
            "param2": 0.0,
        }


class ErrSysSaturation(IErrSimulator):
    __slots__ = ("_meas_min", "_meas_max", "_err_dep")

    def __init__(
        self,
        meas_min: float = -np.inf,
        meas_max: float = np.inf,
        err_dep: EErrDep = EErrDep.DEPENDENT,
    ) -> None:
        self._meas_min = meas_min
        self._meas_max = meas_max
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
        sat = np.clip(err_basis, self._meas_min, self._meas_max)
        return (sat - err_basis, sens_data)

    def to_spec_dict(self) -> dict:
        return {
            "kind": 8,
            "err_type": 0,
            "err_dep": 1 if self._err_dep == EErrDep.DEPENDENT else 0,
            "dist_type": 0,
            "param0": float(self._meas_min),
            "param1": float(self._meas_max),
            "param2": 0.0,
        }
