# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import copy
from dataclasses import dataclass
import numpy as np
from felix.sensorsim.enums import EErrType, EErrDep
from felix.sensorsim.errorsimulator import IErrSimulator
from felix.sensorsim.sensordata import SensorData


@dataclass(slots=True)
class ErrIntOpts:
    force_dependence: EErrDep | None = None
    store_all_errs: bool = False


class ErrIntegrator:
    __slots__ = (
        "_err_chain",
        "_meas_shape",
        "_errs_by_chain",
        "_errs_systematic",
        "_errs_random",
        "_errs_total",
        "_sens_data_by_chain",
        "_err_int_opts",
        "_sens_data_accumulated",
        "_sens_data_initial",
    )

    def __init__(
        self,
        err_chain: list[IErrSimulator],
        sensor_data_initial: SensorData,
        meas_shape: tuple[int, int, int],
        err_int_opts: ErrIntOpts | None = None,
    ) -> None:
        if err_int_opts is None:
            self._err_int_opts = ErrIntOpts()
        else:
            self._err_int_opts = err_int_opts

        self.set_error_chain(err_chain)
        self._meas_shape = meas_shape

        self._sens_data_initial = copy.deepcopy(sensor_data_initial)
        self._sens_data_accumulated = copy.deepcopy(sensor_data_initial)

        if self._err_int_opts.store_all_errs:
            self._sens_data_by_chain = []
            self._errs_by_chain = np.zeros(
                (len(self._err_chain),) + self._meas_shape
            )
        else:
            self._sens_data_by_chain = None
            self._errs_by_chain = None

        self._errs_systematic = np.zeros(meas_shape)
        self._errs_random = np.zeros(meas_shape)
        self._errs_total = np.zeros(meas_shape)

    def reseed_error_chain(self, seed: int | None = None) -> None:
        for ee in self._err_chain:
            ee.reseed(seed)

    def set_error_chain(self, err_chain: list[IErrSimulator]) -> None:
        self._err_chain = err_chain
        if self._err_int_opts.force_dependence is not None:
            for ee in self._err_chain:
                ee.set_error_dep(self._err_int_opts.force_dependence)

    def get_err_chain(self) -> list[IErrSimulator]:
        return self._err_chain

    def get_sens_data_accumulated(self) -> SensorData:
        return self._sens_data_accumulated

    def get_sens_data_initial(self) -> SensorData:
        return self._sens_data_initial

    def get_errs_systematic(self) -> np.ndarray:
        return self._errs_systematic

    def get_errs_random(self) -> np.ndarray:
        return self._errs_random

    def get_errs_total(self) -> np.ndarray:
        return self._errs_total

    def get_errs_by_chain(self) -> np.ndarray | None:
        return self._errs_by_chain

    def get_sens_data_by_chain(self) -> list[SensorData] | None:
        return self._sens_data_by_chain

    def calc_errors_from_chain(
        self,
        truth: np.ndarray,
        sens_data_override: SensorData | None = None,
    ) -> np.ndarray:
        initial_data = (
            sens_data_override
            if sens_data_override is not None
            else self._sens_data_initial
        )
        self._sens_data_accumulated = copy.deepcopy(initial_data)

        self._errs_total = np.zeros_like(truth)
        self._errs_systematic = np.zeros_like(truth)
        self._errs_random = np.zeros_like(truth)

        if self._err_int_opts.store_all_errs:
            self._sens_data_by_chain = []
            self._errs_by_chain = np.zeros(
                (len(self._err_chain),) + self._meas_shape
            )

        for ii, ee in enumerate(self._err_chain):
            dep_val = ee.get_error_dep()
            is_dep = (
                dep_val == EErrDep.DEPENDENT
                or getattr(dep_val, "name", None) == "DEPENDENT"
            )
            if is_dep:
                error_array, updated_data = ee.sim_errs(
                    truth + self._errs_total,
                    self._sens_data_accumulated,
                )
                self._sens_data_accumulated = copy.deepcopy(updated_data)
            else:
                error_array, updated_data = ee.sim_errs(
                    truth,
                    initial_data,
                )

            if self._err_int_opts.store_all_errs:
                self._sens_data_by_chain.append(updated_data)
                self._errs_by_chain[ii, :, :, :] = error_array

            type_val = ee.get_error_type()
            is_sys = (
                type_val == EErrType.SYSTEMATIC
                or getattr(type_val, "name", None) == "SYSTEMATIC"
            )
            if is_sys:
                self._errs_systematic += error_array
            else:
                self._errs_random += error_array

            self._errs_total += error_array

        return self._errs_total

    def to_specs_list(self) -> list[dict]:
        specs = []
        for ee in self._err_chain:
            spec = ee.to_spec_dict()
            if spec:
                specs.append(spec)
        return specs
