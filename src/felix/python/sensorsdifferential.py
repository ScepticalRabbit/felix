# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Differential multi-anchor sensor array in Felix."""

from __future__ import annotations

from typing import Callable
import numpy as np

from felix.python.enums import EDifferentialMode
from felix.python.errgraph import ErrGraph
from felix.python.errspecs import ErrIntOpts, IErrSimulator
from felix.python.fieldspecs import IField
from felix.python.sensordata import SensorData
from felix.python.sensordescriptor import SensorDescriptor
from felix.python.sensorspoint import ErrIntegrator, ISensorArray


class SensorsDifferential(ISensorArray):
    """Differential sensor array measuring relative differences or strain."""

    __slots__ = (
        "_sensor_a",
        "_sensor_b",
        "_mode",
        "_gauge_lengths",
        "_direction_vectors",
        "_custom_func",
        "_descriptor",
        "_truth",
        "_measurements",
        "_error_integrator",
    )

    def __init__(
        self,
        sensor_a: ISensorArray,
        sensor_b: ISensorArray,
        mode: EDifferentialMode = EDifferentialMode.STRAIN,
        custom_func: (
            Callable[[np.ndarray, np.ndarray], np.ndarray] | None
        ) = None,
        descriptor: SensorDescriptor | None = None,
    ) -> None:
        self._sensor_a = sensor_a
        self._sensor_b = sensor_b
        self._mode = mode
        self._custom_func = custom_func

        if descriptor is None:
            descriptor = SensorDescriptor(
                name="Differential Sensor",
                tag="DIFF",
            )
        self._descriptor = descriptor

        self._error_integrator = None
        self._truth = None
        self._measurements = None

        pos_a = sensor_a.get_sensor_data().positions
        pos_b = sensor_b.get_sensor_data().positions
        diff_pos = pos_b - pos_a
        lengths = np.linalg.norm(diff_pos, axis=1, keepdims=True)
        lengths = np.where(lengths == 0.0, 1.0, lengths)
        self._gauge_lengths = lengths.flatten()
        self._direction_vectors = diff_pos / lengths

    def get_sensor_a(self) -> ISensorArray:
        return self._sensor_a

    def get_sensor_b(self) -> ISensorArray:
        return self._sensor_b

    def get_mode(self) -> EDifferentialMode:
        return self._mode

    def get_gauge_lengths(self) -> np.ndarray:
        return self._gauge_lengths

    def get_direction_vectors(self) -> np.ndarray:
        return self._direction_vectors

    def get_descriptor(self) -> SensorDescriptor:
        return self._descriptor

    def get_field(self) -> IField:
        return self._sensor_b.get_field()

    def get_sensor_data(self) -> SensorData:
        return self._sensor_b.get_sensor_data()

    def get_sensor_data_nominal(self) -> SensorData:
        return self._sensor_b.get_sensor_data_nominal()

    def get_all_components(self) -> tuple[str, ...]:
        if self._mode == EDifferentialMode.STRAIN:
            return ("engineering_strain",)
        if self._mode == EDifferentialMode.DIFFERENCE:
            in_comps = self._sensor_b.get_field().get_all_components()
            return tuple(f"diff_{c}" for c in in_comps)
        if self._mode == EDifferentialMode.RATIO:
            in_comps = self._sensor_b.get_field().get_all_components()
            return tuple(f"ratio_{c}" for c in in_comps)
        return ("custom_diff",)

    def get_measurement_shape(self) -> tuple[int, int, int]:
        n_sensors = self._sensor_b.get_sensor_data().positions.shape[0]
        n_comps = len(self.get_all_components())
        n_times = self.get_sample_times().shape[0]
        return (n_sensors, n_comps, n_times)

    def get_sample_times(self) -> np.ndarray:
        return self._sensor_b.get_sample_times()

    def _reduce_measurements(
        self, meas_a: np.ndarray, meas_b: np.ndarray
    ) -> np.ndarray:
        if self._mode == EDifferentialMode.DIFFERENCE:
            return meas_b - meas_a
        if self._mode == EDifferentialMode.RATIO:
            return meas_b / np.where(meas_a == 0.0, 1e-12, meas_a)
        if self._mode == EDifferentialMode.CUSTOM:
            if self._custom_func is None:
                raise ValueError("Custom reduction func required.")
            return self._custom_func(meas_a, meas_b)

        diff_u = meas_b - meas_a
        n_comps = diff_u.shape[1]
        e_ab = self._direction_vectors[:, :n_comps, np.newaxis]
        delta_l = np.sum(diff_u * e_ab, axis=1, keepdims=True)
        l_0 = self._gauge_lengths[:, np.newaxis, np.newaxis]
        return delta_l / l_0

    def calc_truth(self) -> np.ndarray:
        truth_a = self._sensor_a.get_truth()
        truth_b = self._sensor_b.get_truth()
        self._truth = self._reduce_measurements(truth_a, truth_b)
        return self._truth

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth()
        return self._truth

    def set_error_chain(
        self,
        err_chain: list[IErrSimulator] | ErrGraph | None,
        err_int_opts: ErrIntOpts | None = None,
    ) -> None:
        if err_chain is None:
            self._error_integrator = None
            return None

        if isinstance(err_chain, ErrGraph):
            self._error_integrator = err_chain
            return None

        self._error_integrator = ErrIntegrator(
            err_chain=err_chain,
            sensor_data_initial=self.get_sensor_data(),
            meas_shape=self.get_measurement_shape(),
            err_int_opts=err_int_opts,
        )
        return None

    def set_error_graph(self, err_graph: ErrGraph | None) -> None:
        self._error_integrator = err_graph

    def sim_measurements(self) -> np.ndarray:
        truth = self.get_truth()
        if self._error_integrator is None:
            self._measurements = truth
            return self._measurements

        if isinstance(self._error_integrator, ErrGraph):
            total_err = self._error_integrator.calc_errors_from_graph(truth)
        else:
            total_err = self._error_integrator.calc_errors_from_chain(truth)

        self._measurements = truth + total_err
        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.sim_measurements()
        return self._measurements

    def get_errors_systematic(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_systematic()

    def get_errors_random(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_random()

    def get_errors_total(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_total()
