# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import numpy as np
from felix.sensorsim.field import IField
from felix.sensorsim.fieldtensor import FieldTensor
from felix.sensorsim.errorintegrator import ErrIntegrator, ErrIntOpts
from felix.sensorsim.errorsimulator import IErrSimulator
from felix.sensorsim.sensordescriptor import SensorDescriptor
from felix.sensorsim.sensordata import SensorData
from felix.sensorsim.simtools import sample_simdata_field


class SensorsPoint:
    __slots__ = (
        "_field",
        "_descriptor",
        "_sensor_data",
        "_truth",
        "_measurements",
        "_error_integrator",
    )

    def __init__(
        self,
        sensor_data: SensorData,
        field: IField,
        descriptor: SensorDescriptor | None = None,
    ) -> None:
        self._sensor_data = sensor_data
        self._field = field
        self._error_integrator = None

        self._descriptor = (
            descriptor if descriptor is not None else SensorDescriptor()
        )
        self._truth = None
        self._measurements = None

    @property
    def field(self) -> IField:
        return self._field

    @property
    def descriptor(self) -> SensorDescriptor:
        return self._descriptor

    @property
    def sensor_data(self) -> SensorData:
        return self._sensor_data

    @property
    def truth(self) -> np.ndarray:
        return self.get_truth()

    @property
    def measurements(self) -> np.ndarray:
        return self.get_measurements()

    @property
    def errors_systematic(self) -> np.ndarray | None:
        return self.get_errors_systematic()

    @property
    def errors_random(self) -> np.ndarray | None:
        return self.get_errors_random()

    @property
    def errors_total(self) -> np.ndarray | None:
        return self.get_errors_total()

    def get_sensor_data(self) -> SensorData:
        return self._sensor_data

    def get_sensor_data_nominal(self) -> SensorData:
        return self._sensor_data

    def get_sensor_data_perturbed(self) -> SensorData:
        if (
            self._error_integrator is not None
            and hasattr(self._error_integrator, "_sens_data_accumulated")
            and self._error_integrator._sens_data_accumulated is not None
        ):
            return self._error_integrator._sens_data_accumulated
        return self._sensor_data

    def get_descriptor(self) -> SensorDescriptor:
        return self._descriptor

    def get_sample_times(self) -> np.ndarray:
        if self._sensor_data.sample_times is None:
            return self._field.get_time_steps()
        return self._sensor_data.sample_times

    def get_measurement_shape(self) -> tuple[int, int, int]:
        n_sensors = (
            self._sensor_data.positions.shape[0]
            if self._sensor_data.positions is not None
            else 0
        )
        n_comps = len(self._field.get_all_components())
        n_times = self.get_sample_times().shape[0]
        return (n_sensors, n_comps, n_times)

    def get_field(self) -> IField:
        return self._field

    def calc_truth(self) -> np.ndarray:
        is_tensor = isinstance(self._field, FieldTensor)
        truth, _, _, _, _ = sample_simdata_field(
            sim_data=self._field.get_sim_data(),
            comp_keys=self._field.get_all_components(),
            spatial_dims=self._field._spatial_dims,
            points=self._sensor_data.positions,
            times=self._sensor_data.sample_times,
            angles=self._sensor_data.angles,
            is_tensor=is_tensor,
            error_specs=None,
        )
        self._truth = truth
        return self._truth

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth()
        return self._truth

    def get_error_integrator(self) -> ErrIntegrator | None:
        return self._error_integrator

    def set_error_chain(
        self,
        err_chain: list[IErrSimulator] | None,
        err_int_opts: ErrIntOpts | None = None,
    ) -> None:
        if err_chain is None:
            self._error_integrator = None
            return

        if err_int_opts is None:
            err_int_opts = ErrIntOpts()

        self._error_integrator = ErrIntegrator(
            err_chain,
            self._sensor_data,
            self.get_measurement_shape(),
            err_int_opts,
        )

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

    def sim_measurements(self) -> np.ndarray:
        truth = self.get_truth()
        if self._error_integrator is None:
            self._measurements = truth
        else:
            total_errs = self._error_integrator.calc_errors_from_chain(
                truth, self._sensor_data
            )
            self._measurements = truth + total_errs

        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.sim_measurements()
        return self._measurements
