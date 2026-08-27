# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from collections.abc import Callable
import numpy as np
from felix.sensorsim.enums import EErrType, EErrDep
from felix.sensorsim.errorsimulator import IErrSimulator
from felix.sensorsim.sensordata import SensorData


class ErrSysCalibration(IErrSimulator):
    __slots__ = (
        "_assumed_calib",
        "_truth_calib",
        "_truth_calib_prime",
        "_cal_range",
        "_n_cal_divs",
        "_use_newton",
        "_tol",
        "_max_iter",
        "_err_dep",
        "_truth_cal_table",
    )

    def __init__(
        self,
        assumed_calib: Callable[[np.ndarray], np.ndarray],
        truth_calib: Callable[[np.ndarray], np.ndarray],
        cal_range: tuple[float, float],
        n_cal_divs: int = 10000,
        use_newton: bool = False,
        truth_calib_prime: Callable[[np.ndarray], np.ndarray] | None = None,
        tol: float = 1e-8,
        max_iter: int = 50,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
    ) -> None:
        self._assumed_calib = assumed_calib
        self._truth_calib = truth_calib
        self._truth_calib_prime = truth_calib_prime
        self._cal_range = cal_range
        self._n_cal_divs = n_cal_divs
        self._use_newton = use_newton
        self._tol = tol
        self._max_iter = max_iter
        self._err_dep = err_dep

        self._truth_cal_table = np.zeros((n_cal_divs, 2), dtype=np.float64)
        self._truth_cal_table[:, 0] = np.linspace(
            cal_range[0], cal_range[1], n_cal_divs
        )
        self._truth_cal_table[:, 1] = self._truth_calib(
            self._truth_cal_table[:, 0]
        )

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
        raw_signals = np.interp(
            err_basis,
            self._truth_cal_table[:, 1],
            self._truth_cal_table[:, 0],
        )
        calib_measurements = self._assumed_calib(raw_signals)
        return (calib_measurements - err_basis, sens_data)

    def to_spec_dict(self) -> dict:
        # Invert table so column 0 is y (truth physical), column 1 is assumed(x) (measured physical)
        raw_x = np.linspace(
            self._cal_range[0], self._cal_range[1], self._n_cal_divs
        )
        truth_y = self._truth_calib(raw_x)
        assumed_y = self._assumed_calib(raw_x)

        # Sort by truth_y
        sort_idx = np.argsort(truth_y)
        table_inv = np.zeros((self._n_cal_divs, 2), dtype=np.float64)
        table_inv[:, 0] = truth_y[sort_idx]
        table_inv[:, 1] = assumed_y[sort_idx]

        return {
            "kind": 9,
            "err_type": 0,
            "err_dep": 1 if self._err_dep == EErrDep.DEPENDENT else 0,
            "dist_type": 0,
            "param0": 0.0,
            "param1": 0.0,
            "param2": 0.0,
            "table": table_inv,
        }
