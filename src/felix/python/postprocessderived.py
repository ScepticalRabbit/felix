# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Multi-sensor fusion processors and derived quantity calculators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np
from scipy import integrate

from felix.python.measurementdata import MeasurementData
from felix.python.sensordescriptor import SensorDescriptor


class IMeasurementProcessor(ABC):
    """Abstract interface for measurement data post-processing operators."""

    @abstractmethod
    def get_source_keys(self) -> tuple[str, ...]:
        """Returns input sensor keys required by this processor."""

    @abstractmethod
    def get_output_components(
        self, input_metadata: dict[str, MeasurementData]
    ) -> tuple[str, ...]:
        """Returns derived output component names."""

    @abstractmethod
    def process(
        self, inputs: dict[str, MeasurementData]
    ) -> MeasurementData:
        """Processes input measurement datasets into derived output dataset."""


class ProcessStiffness(IMeasurementProcessor):
    """Calculates dynamic secant stiffness k(t) = F(t) / u(t)."""

    __slots__ = ("_force_key", "_disp_key", "_eps", "_units")

    def __init__(
        self,
        force: str,
        disp: str,
        eps: float = 1e-9,
        units: str = "N/mm",
    ) -> None:
        self._force_key = force
        self._disp_key = disp
        self._eps = float(eps)
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._force_key, self._disp_key)

    def get_output_components(
        self, input_metadata: dict[str, MeasurementData]
    ) -> tuple[str, ...]:
        return ("stiffness",)

    def process(
        self, inputs: dict[str, MeasurementData]
    ) -> MeasurementData:
        f_data = inputs[self._force_key]
        u_data = inputs[self._disp_key]

        f_vals = f_data.values
        u_vals = u_data.values

        denom = np.where(np.abs(u_vals) < self._eps, self._eps, u_vals)
        k_vals = f_vals / denom

        desc = SensorDescriptor(
            name="Dynamic Stiffness", tag="STIFF", units=self._units
        )

        return MeasurementData(
            values=k_vals,
            sample_times=f_data.sample_times,
            positions=f_data.positions,
            components=("stiffness",),
            units=self._units,
            descriptor=desc,
        )


class ProcessWork(IMeasurementProcessor):
    """Calculates mechanical work W(t) = integral F(t) u_dot(t) dt."""

    __slots__ = ("_force_key", "_disp_key", "_units")

    def __init__(
        self,
        force: str,
        disp: str,
        units: str = "mJ",
    ) -> None:
        self._force_key = force
        self._disp_key = disp
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._force_key, self._disp_key)

    def get_output_components(
        self, input_metadata: dict[str, MeasurementData]
    ) -> tuple[str, ...]:
        return ("work",)

    def process(
        self, inputs: dict[str, MeasurementData]
    ) -> MeasurementData:
        f_data = inputs[self._force_key]
        u_data = inputs[self._disp_key]

        f_vals = f_data.values
        u_vals = u_data.values
        times = f_data.sample_times

        t_1d = times[0] if times.ndim > 1 else times
        u_dot = np.gradient(u_vals, t_1d, axis=-1)

        power = f_vals * u_dot
        work = integrate.cumulative_trapezoid(
            power, x=t_1d, axis=-1, initial=0.0
        )

        desc = SensorDescriptor(
            name="Mechanical Work", tag="WORK", units=self._units
        )

        return MeasurementData(
            values=work,
            sample_times=f_data.sample_times,
            positions=f_data.positions,
            components=("work",),
            units=self._units,
            descriptor=desc,
        )


class ProcessRelativeDifference(IMeasurementProcessor):
    """Calculates relative difference delta_y(t) = y_B(t) - y_A(t)."""

    __slots__ = ("_source_a", "_source_b", "_label", "_units")

    def __init__(
        self,
        source_a: str,
        source_b: str,
        label: str = "difference",
        units: str | None = None,
    ) -> None:
        self._source_a = source_a
        self._source_b = source_b
        self._label = label
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source_a, self._source_b)

    def get_output_components(
        self, input_metadata: dict[str, MeasurementData]
    ) -> tuple[str, ...]:
        in_a = input_metadata[self._source_a]
        return tuple(f"{self._label}_{c}" for c in in_a.components)

    def process(
        self, inputs: dict[str, MeasurementData]
    ) -> MeasurementData:
        data_a = inputs[self._source_a]
        data_b = inputs[self._source_b]

        diff_vals = data_b.values - data_a.values
        out_comps = tuple(f"{self._label}_{c}" for c in data_a.components)
        units = self._units if self._units is not None else data_a.units

        desc = SensorDescriptor(
            name=f"Relative Difference ({self._label})",
            tag="DIFF",
            units=units,
        )

        return MeasurementData(
            values=diff_vals,
            sample_times=data_a.sample_times,
            positions=data_b.positions,
            components=out_comps,
            units=units,
            descriptor=desc,
        )
