# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from copy import deepcopy

import numpy as np

from felix.cython import felix as fc
from felix.python.errspecs import ErrIntOpts, IErrSimulator
from felix.python.fieldspecs import FieldSpec
from felix.python.sensordata import SensorData


class ErrIntegrator:
    """Compatibility view over error results calculated by Zig."""

    __slots__ = (
        "_err_chain",
        "_err_int_opts",
        "_errs_systematic",
        "_errs_random",
        "_errs_total",
        "_errs_by_chain",
        "_sens_data_accumulated",
        "_sens_data_by_chain",
        "_sens_data_initial",
    )

    def __init__(
        self,
        err_chain: list[IErrSimulator],
        sensor_data: SensorData | None = None,
        err_int_opts: ErrIntOpts | None = None,
        sensor_data_initial: SensorData | None = None,
        meas_shape: tuple[int, int, int] | None = None,
    ) -> None:
        self._err_chain = err_chain
        self._err_int_opts = (
            err_int_opts if err_int_opts is not None else ErrIntOpts()
        )
        s_data = (
            sensor_data
            if sensor_data is not None
            else (
                sensor_data_initial
                if sensor_data_initial is not None
                else SensorData()
            )
        )
        self._errs_systematic = None
        self._errs_random = None
        self._errs_total = None
        self._errs_by_chain = None
        self._sens_data_by_chain = None
        self._sens_data_initial = deepcopy(s_data)
        self._sens_data_accumulated = deepcopy(s_data)

    def get_sens_data_accumulated(self) -> SensorData:
        return self._sens_data_accumulated

    def get_sens_data_initial(self) -> SensorData:
        return self._sens_data_initial

    def get_sens_data_by_chain(self) -> list[SensorData] | None:
        return self._sens_data_by_chain

    def get_errs_by_chain(self) -> np.ndarray | None:
        return self._errs_by_chain

    def get_errs_systematic(self) -> np.ndarray | None:
        return self._errs_systematic

    def get_errs_random(self) -> np.ndarray | None:
        return self._errs_random

    def get_errs_total(self) -> np.ndarray | None:
        return self._errs_total

    def reseed_error_chain(self, seed: int | None = None) -> None:
        for error in self._err_chain:
            error.reseed(seed)

    def reseed(self, seed: int | None = None) -> None:
        self.reseed_error_chain(seed)

    def calc_errors_from_chain(self, truth: np.ndarray) -> np.ndarray:
        from felix.python.errgraph import err_chain_to_graph

        graph = err_chain_to_graph(
            self._err_chain,
            truth.shape,
            self._sens_data_accumulated,
        )
        self._errs_total = graph.calc_errors_from_graph(truth)
        self._errs_systematic = graph.get_errs_systematic()
        self._errs_random = graph.get_errs_random()
        return self._errs_total


class SensorsPoint:
    """Thin compatibility adapter over the Zig point-sensor pipeline."""

    __slots__ = (
        "_sensor_data",
        "_field",
        "_descriptor",
        "_error_integrator",
        "_err_graph",
        "_truth",
        "_measurements",
    )

    def __init__(
        self,
        sensor_data: SensorData,
        field: FieldSpec,
        descriptor: object | None = None,
    ) -> None:
        self._sensor_data = sensor_data
        self._field = field
        self._descriptor = descriptor
        self._error_integrator = None
        self._err_graph = None
        self._truth = None
        self._measurements = None

    def get_field(self) -> FieldSpec:
        return self._field

    def get_sensor_data(self) -> SensorData:
        return self._sensor_data

    def get_sensor_data_nominal(self) -> SensorData:
        return self._sensor_data

    def get_sensor_data_perturbed(self) -> SensorData:
        if self._error_integrator is None:
            return self._sensor_data
        return self._error_integrator._sens_data_accumulated

    def get_descriptor(self) -> object | None:
        return self._descriptor

    def get_sample_times(self) -> np.ndarray:
        if self._sensor_data.sample_times is None:
            return self._field.get_time_steps()
        return self._sensor_data.sample_times

    def get_measurement_shape(self) -> tuple[int, int, int]:
        positions = self._sensor_data.positions
        num_sensors = 0 if positions is None else positions.shape[0]
        return (
            num_sensors,
            len(self._field.get_all_components()),
            self.get_sample_times().shape[0],
        )

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
        opts = err_int_opts if err_int_opts is not None else ErrIntOpts()
        if opts.force_dependence is not None:
            for error in err_chain:
                error.set_error_dep(opts.force_dependence)
        self._error_integrator = ErrIntegrator(
            err_chain,
            self._sensor_data,
            opts,
        )
        self._err_graph = None

    def set_error_graph(self, err_graph: object | None) -> None:
        self._err_graph = err_graph
        self._error_integrator = None

    def calc_truth(self) -> np.ndarray:
        self._truth = self._simulate(None)[0]
        return self._truth

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            return self.calc_truth()
        return self._truth

    def sim_measurements(self) -> np.ndarray:
        if self._err_graph is not None:
            graph_dict = self._err_graph.to_spec_dict()
            if self._sensor_data.positions is None:
                raise ValueError("SensorData.positions must be provided")
            result = fc.sample_field_config_graph(
                self._field,
                self._sensor_data.positions,
                graph_dict,
                self._sensor_data.sample_times,
                self._sensor_data.angles,
            )
            self._truth = result[0]
            self._measurements = result[1]
            self._err_graph._errs_systematic = result[2]
            self._err_graph._errs_random = result[3]
            self._err_graph._errs_total = result[4]
            return self._measurements

        specs = None
        if self._error_integrator is not None:
            specs = [
                _convert_error_spec(error)
                for error in self._error_integrator._err_chain
            ]
        result = self._simulate(specs)
        self._truth = result[0]
        self._measurements = result[1]
        if self._error_integrator is not None:
            self._error_integrator._errs_systematic = result[2]
            self._error_integrator._errs_random = result[3]
            self._error_integrator._errs_total = result[4]
        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            return self.sim_measurements()
        return self._measurements

    def get_errors_systematic(self) -> np.ndarray | None:
        if self._error_integrator is not None:
            return self._error_integrator.get_errs_systematic()
        if self._err_graph is not None:
            return self._err_graph.get_errs_systematic()
        return None

    def get_errors_random(self) -> np.ndarray | None:
        if self._error_integrator is not None:
            return self._error_integrator.get_errs_random()
        if self._err_graph is not None:
            return self._err_graph.get_errs_random()
        return None

    def get_errors_total(self) -> np.ndarray | None:
        if self._error_integrator is not None:
            return self._error_integrator.get_errs_total()
        if self._err_graph is not None:
            return self._err_graph.get_errs_total()
        return None

    def sim_experiments(
        self,
        num_experiments: int,
        seed: int = 0,
    ) -> tuple[np.ndarray, ...]:
        if self._sensor_data.positions is None:
            raise ValueError("SensorData.positions must be provided")

        if self._err_graph is not None:
            graph_dict = self._err_graph.to_spec_dict()
            return fc.sample_field_config_graph(
                self._field,
                self._sensor_data.positions,
                graph_dict,
                self._sensor_data.sample_times,
                self._sensor_data.angles,
                num_experiments,
                seed,
            )

        specs = None
        if self._error_integrator is not None:
            specs = [
                _convert_error_spec(error)
                for error in self._error_integrator._err_chain
            ]
        return fc.sample_field_config(
            self._field,
            self._sensor_data.positions,
            self._sensor_data.sample_times,
            self._sensor_data.angles,
            specs,
            num_experiments,
            seed,
        )

    def _simulate(self, specs: list[dict] | None) -> tuple[np.ndarray, ...]:
        if self._sensor_data.positions is None:
            raise ValueError("SensorData.positions must be provided")
        from felix.python.fieldtransforms import FieldTransformed

        if isinstance(self._field, FieldTransformed):
            truth = self._field.sample_field(
                points=self._sensor_data.positions,
                times=self._sensor_data.sample_times,
                angles=self._sensor_data.angles,
            )
            if specs is None or len(specs) == 0:
                return (truth, truth, None, None, None)
            from felix.python.errgraph import err_chain_to_graph
            graph = err_chain_to_graph(
                self._error_integrator._err_chain,
                truth.shape,
                self._sensor_data,
            )
            tot = graph.calc_errors_from_graph(truth)
            return (
                truth,
                truth + tot,
                graph.get_errs_systematic(),
                graph.get_errs_random(),
                tot,
            )

        return fc.sample_field_config(
            self._field,
            self._sensor_data.positions,
            self._sensor_data.sample_times,
            self._sensor_data.angles,
            specs,
        )


ISensorArray = SensorsPoint


def _convert_error_spec(error: object) -> dict:
    to_spec = getattr(error, "to_spec_dict", None)
    if to_spec is not None:
        spec = to_spec()
        if spec:
            return spec
    raise TypeError(
        f"Error adapter is not supported by the Zig pipeline: {type(error)}"
    )
